import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt

from linear_decoder import LinearDecoder  # Import the LinearDecoder class from the linear_decoder module


# Read command-line arguments for V_th_std
V_th_std = float(sys.argv[1])  # Threshold standard deviation (V_th_std)

n_neurons = 10000  # Set the number of neurons
duration = 200  # Set the duration of the simulation in seconds
dt = 0.1  # Set the time step (delta t) in seconds

spikes_trials_all = []  # Initialize an empty list to store spike times for all trials
n_trials = 20  # Set the number of trials to simulate
for trial in range(n_trials):  # Loop through each trial
    filename = f"/Users/frazi/scripts/HLIF_7_mac/data/spikes_sine_V_th_std_{V_th_std}_trial_{trial}.pkl"
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    spikes_trials_all.append(data["spikes"])  # Add the generated spike times to the list of all trials


time = np.arange(duration/dt)  # Create a time array from 0 to duration with step size dt
signal = np.zeros((1,int(duration/dt)), dtype=float)  # Initialize a 2D array for two signals
signal[0] = 2*np.sin(5*np.pi*time/10000)  # Set the first signal as a sine wave with frequency 5π/10000


tau = 10  # Set the time constant for exponential kernel in milliseconds
lambda_reg = 1e-3  # Set the regularization parameter for ridge regression
rng_decode = np.random.default_rng(np.random.SeedSequence(entropy=654321, spawn_key=(0, 0, 3, int(100*V_th_std))))  # Initialize a random number generator

decoder_network = LinearDecoder(dt, tau, lambda_reg, rng_decode)  # Create a LinearDecoder object with specified parameters

# Preprocess spike data using the decoder's method, applying filtering to all trials
filtered_spikes = decoder_network.preprocess_data(spikes_trials_all, n_neurons, duration)
train_errors_network, test_errors_network, all_weights_network = decoder_network.stratified_cv(filtered_spikes, signal, n_splits=10)


decoder_stim = LinearDecoder(dt, tau, lambda_reg, rng_decode)  # Create a LinearDecoder object with specified parameters for the stimulated subset
# Perform stratified cross-validation
train_errors_stim, test_errors_stim, all_weights_stim = decoder_stim.stratified_cv(filtered_spikes[:,:,data["input_1_neurons"]], signal, n_splits=10)

data_to_save = {
    "test_rmse_stim": test_errors_stim[: , 0],
    "test_rmse_network": test_errors_network[: , 0],
    "weights_stim": np.array(all_weights_stim)[:, :, 0],
    "weights_network": np.array(all_weights_network)[:, :, 0],
    "ground_truth_signal": signal[0],
    "decoded_test_signal_stim": decoder_stim.example_predicted_test[:, 0],
    "decoded_test_signal_network": decoder_network.example_predicted_test[:, 0],
    "stim_neurons_id": data["input_1_neurons"]

}

# Save simulation results
filename = f'/Users/frazi/scripts/HLIF_7_mac/data/decoding_V_th_std_{V_th_std}.pkl'
with open(filename, 'wb') as f:
    pickle.dump(data_to_save, f)
print(f"Data saved to {filename}")
