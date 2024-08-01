import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Import Sionna
import sionna

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

import matplotlib.pyplot as plt
import numpy as np
import pickle

import time
import random
from datetime import datetime

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.nn import relu


from tensorflow.keras.layers import (
    Layer,
    Dense,
    LayerNormalization,
    MultiHeadAttention,
)

from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, UMa
from sionna.channel import OFDMChannel
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper
from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber
from sionna.utils import sim_ber

############################################
## Channel configuration
carrier_frequency = 28e9 # Hz
delay_spread = 266e-9 # s
cdl_model = "C" # CDL model to use
# speed = 27.0 # Speed for evaluation and training [m/s]
# SNR range for evaluation and training [dB]
ebno_db_min = -5.0
ebno_db_max = 15.0

############################################
## OFDM waveform configuration
subcarrier_spacing = 240e3 # Hz
fft_size = 128 # Number of subcarriers forming the resource grid, including the null-subcarrier and the guard bands
num_ofdm_symbols = 14 # Number of OFDM symbols forming the resource grid
dc_null = True # Null the DC subcarrier
num_guard_carriers = [5, 6] # Number of guard carriers on each side
pilot_pattern = "kronecker" # Pilot pattern
pilot_ofdm_symbol_indices = [2, 11] # Index of OFDM symbols carrying pilots
cyclic_prefix_length = 0 # Simulation in frequency domain. This is useless

############################################
## Modulation and coding configuration
num_bits_per_symbol = 6 # QPSK
coderate = 0.5 # Coderate for LDPC code

############################################
## Neural receiver configuration
embed_dim = 128
num_heads = 4
ff_dim = 128

############################################
## Training configuration
num_training_iterations = 375000 # Number of training iterations
training_batch_size = 32 # Training batch size
model_weights_path = "trans_model_weights" # Location to save the neural receiver weights once training is done

############################################
## Evaluation configuration
# results_filename = "neural_receiver_results" # Location to save the results

num_bs_ant = 2
num_ut = 1
num_ut_ant = 1


# Create an RX-TX association matrix
# rx_tx_association[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change. 
bs_ut_association = np.zeros([1, num_ut])
bs_ut_association[0, :] = 1
rx_tx_association = bs_ut_association
num_tx = num_ut
num_streams_per_tx = num_ut_ant


stream_manager = StreamManagement(rx_tx_association, num_streams_per_tx)               # One stream per transmitter

resource_grid = ResourceGrid(num_ofdm_symbols = num_ofdm_symbols,
                             fft_size = fft_size,
                             subcarrier_spacing = subcarrier_spacing,
                             num_tx = 1,
                             num_streams_per_tx = 1,
                             cyclic_prefix_length = cyclic_prefix_length,
                             dc_null = dc_null,
                             pilot_pattern = pilot_pattern,
                             pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices,
                             num_guard_carriers = num_guard_carriers)


# Codeword length. It is calculated from the total number of databits carried by the resource grid, and the number of bits transmitted per resource element
n = int(resource_grid.num_data_symbols*num_bits_per_symbol)
# Number of information bits per codeword
k = int(n*coderate)

# Configure antenna arrays
ut_array = AntennaArray(num_rows=1,
                        num_cols=1,
                        polarization="single",
                        polarization_type="V",
                        antenna_pattern="omni",
                        carrier_frequency=carrier_frequency)

bs_array = AntennaArray(num_rows=1,
                        # num_cols=1,
                        num_cols=int(num_bs_ant/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)


#transformer implementation

class TransformerBlock(Layer):
    def build(self, input_shape):
        self._att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self._layer_norm_1 = LayerNormalization(axis=(-1,-2))
        self._ffn_1 = Dense(ff_dim)
        self._ffn_2 = Dense(embed_dim)       
        self._layer_norm_2 = LayerNormalization(axis=(-1, -2))

    def call(self, inputs):    
        attn_output = self._att(inputs, inputs)
        out1 = inputs + attn_output
        out1 = self._layer_norm_1(out1)
        ffn_output = self._ffn_1(out1)
        ffn_output = relu(ffn_output)
        ffn_output = self._ffn_2(ffn_output)
        trans_output = self._layer_norm_2(out1 + ffn_output)
        return trans_output

class NeuralReceiver(Layer):

    def build(self, input_shape):
        #trasformer implementation
        self._input_dense = Dense(embed_dim)

        self._transformer_block1 = TransformerBlock()
        self._transformer_block2 = TransformerBlock()
        self._transformer_block3 = TransformerBlock()
        self._transformer_block4 = TransformerBlock()

        self._output_dense = Dense(num_bits_per_symbol)
    
    def call(self, inputs):
        y, no = inputs
        # Feeding the noise power in log10 scale helps with the performance
        no = log10(no)
        # Stacking the real and imaginary components of the different antennas along the 'channel' dimension
        y = tf.transpose(y, [0, 2, 3, 1]) # Putting antenna dimension last
        no = insert_dims(no, 3, 1)
        no = tf.tile(no, [1, y.shape[1], y.shape[2], 1])
        # z : [batch size, num ofdm symbols, num subcarriers, 2*num rx antenna + 1]
        z = tf.concat([tf.math.real(y),
                       tf.math.imag(y),
                       no], axis=-1)
        
        
        # transformer implementation
        z = tf.reshape(z, (-1, z.shape[1]*z.shape[2], z.shape[3]))
        z = self._input_dense(z)

        z = self._transformer_block1(z)
        z = self._transformer_block2(z)
        z = self._transformer_block3(z)
        z = self._transformer_block4(z)

        z = self._output_dense(z)
        z = tf.reshape(z, (-1, num_ofdm_symbols, fft_size, num_bits_per_symbol))

        return z
    

class E2ESystem(Model):
    def __init__(self, system, training=False):
        super().__init__()
        self._system = system
        self._training = training

        ######################################
        ## Transmitter
        self._binary_source = BinarySource()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._encoder = LDPC5GEncoder(k, n)
        self._mapper = Mapper("qam", num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(resource_grid)

        cdl = CDL(cdl_model, delay_spread, carrier_frequency,
                  ut_array, bs_array, "uplink", min_speed=27)
        self._channel_model = OFDMChannel(cdl, resource_grid, normalize_channel=True, return_channel=True)

        
        #####################################
        # Channel
        # A 3GPP CDL channel model is used
        self._channel = OFDMChannel(self._channel_model, resource_grid, add_awgn=True,
                                    normalize_channel=True, return_channel=True)

        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of `system`
        if "baseline" in system:
            if system == 'baseline-perfect-csi': # Perfect CSI
                self._removed_null_subc = RemoveNulledSubcarriers(resource_grid)
            elif system == 'baseline-ls-estimation': # LS estimation
                self._ls_est = LSChannelEstimator(resource_grid, interpolation_type="nn")
            # Components required by both baselines
            self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager)
            self._demapper = Demapper("app", "qam", num_bits_per_symbol)
        elif system == "neural-receiver": # Neural receiver
            self._neural_receiver = NeuralReceiver()
            self._rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) # Used to extract data-carrying resource elements
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)


    @tf.function
    def call(self, batch_size, ebno_db):
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        # Outer coding is only performed if not training
        if self._training:
            c = self._binary_source([batch_size, num_tx, num_streams_per_tx, n])
        else:
            b = self._binary_source([batch_size, num_tx, num_streams_per_tx, k])
            print("b: " + str(b))
            c = self._encoder(b)
            
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y,h = self._channel([x_rg, no_])

        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of ``system``
        if "baseline" in self._system:
            if self._system == 'baseline-perfect-csi':
                h_hat = self._removed_null_subc(h) # Extract non-null subcarriers
                err_var = 0.0 # No channel estimation error when perfect CSI knowledge is assumed
            elif self._system == 'baseline-ls-estimation':
                h_hat, err_var = self._ls_est([y, no]) # LS channel estimation with nearest-neighbor
            x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no]) # LMMSE equalization
            no_eff_= expand_to_rank(no_eff, tf.rank(x_hat))
            llr = self._demapper([x_hat, no_eff_]) # Demapping
        elif self._system == "neural-receiver":
            # The neural receover computes LLRs from the frequency domain received symbols and N0
            y = tf.squeeze(y, axis=1)
            llr = self._neural_receiver([y, no])
            llr = insert_dims(llr, 2, 1) # Reshape the input to fit what the resource grid demapper is expected
            llr = self._rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded
            llr = tf.reshape(llr, [batch_size, 1, 1, n]) # Reshape the LLRs to fit what the outer decoder is expected

        # Outer coding is not needed if the information rate is returned
        if self._training:
            # Compute and return BMD rate (in bit), which is known to be an achievable
            # information rate for BICM systems.
            # Training aims at maximizing the BMD rate

            predictions = tf.sigmoid(llr)
            # Define a threshold to classify predictions as either 0 or 1
            threshold = 0.7
            predicted_classes = tf.cast(tf.greater(predictions, threshold), tf.float32)
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_classes, c), tf.float32))

            # Calculate Mean Absolute Error (MAE)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(c, llr)
            bce = tf.reduce_mean(bce)
            rate = tf.constant(1.0, tf.float32) - bce/tf.math.log(2.)

            return rate, accuracy, bce


        else:
            # Outer decoding
            b_hat = self._decoder(llr)
            return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation
        

# Range of SNRs over which the systems are evaluated
ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                     ebno_db_max, # Max SNR for evaluation
                     0.5) # Step


model = E2ESystem('neural-receiver')

# Run one inference to build the layers and loading the weights
model(1, tf.constant(10.0, tf.float32))
with open(model_weights_path, 'rb') as f:
    weights = pickle.load(f)
model.set_weights(weights)

# ebno_db = tf.constant([[5]], dtype=tf.float32)
# b,b_hat = model(100, ebno_db)

BLER={}
BER={}
# # Evaluations
ber,bler = sim_ber(model, ebno_dbs, batch_size=12, num_target_block_errors=100, max_mc_iter=100)
BLER['neural-receiver'] = bler.numpy()
BER['neural-receiver'] = ber.numpy()

model = E2ESystem('baseline-perfect-csi')
ber,bler = sim_ber(model, ebno_dbs, batch_size=12, num_target_block_errors=100, max_mc_iter=100)
BLER['baseline-perfect-csi'] = bler.numpy()
BER['baseline-perfect-csi'] = ber.numpy()

model = E2ESystem('baseline-ls-estimation')
ber,bler = sim_ber(model, ebno_dbs, batch_size=12, num_target_block_errors=100, max_mc_iter=100)
BLER['baseline-ls-estimation'] = bler.numpy()
BER['baseline-ls-estimation'] = ber.numpy()




plt.figure(figsize=(10,6))
# Baseline - Perfect CSI
plt.semilogy(ebno_dbs, BLER['baseline-perfect-csi'], 'o-', c=f'C0', label=f'Baseline - Perfect CSI')
# Baseline - LS Estimation
plt.semilogy(ebno_dbs, BLER['baseline-ls-estimation'], 'x--', c=f'C1', label=f'Baseline - LS Estimation')
# Neural receiver
plt.semilogy(ebno_dbs, BLER['neural-receiver'], 's-.', c=f'C2', label=f'Neural receiver')
#
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()
plt.savefig('BLER_proposed')


plt.figure(figsize=(10,6))
# Baseline - Perfect CSI
plt.semilogy(ebno_dbs, BER['baseline-perfect-csi'], 'o-', c=f'C0', label=f'Baseline - Perfect CSI')
# Baseline - LS Estimation
plt.semilogy(ebno_dbs, BER['baseline-ls-estimation'], 'x--', c=f'C1', label=f'Baseline - LS Estimation')
# Neural receiver
plt.semilogy(ebno_dbs, BER['neural-receiver'], 's-.', c=f'C2', label=f'Neural receiver')
#
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()
plt.savefig('BER_proposed')


