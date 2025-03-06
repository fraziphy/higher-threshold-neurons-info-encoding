import numpy as np
import pickle

def spikes_to_matrix(spike_list, n_steps, N, step_size):
    """
    Convert spike data into a spike matrix.

    Args:
        spike_list (list): List of spikes [(time, neuron_id), ...].
        n_steps (int): Number of time steps.
        N (int): Number of neurons.
        step_size (float): Time step size in ms.

    Returns:
        numpy.ndarray: Spike matrix of shape (n_steps, N).
    """
    spike_matrix = np.zeros((n_steps, N))
    for spike_time, neuron_id in spike_list:
        time_bin = int(spike_time / step_size)
        if 0 <= time_bin < n_steps and 0 <= neuron_id < N:
            spike_matrix[time_bin, neuron_id] += 1
    return spike_matrix


def compute_participation_ratio(spike_matrix):
    # Center the data
    centered_matrix = spike_matrix - np.mean(spike_matrix, axis=0, keepdims=True)

    # Compute SVD
    _, s, _ = np.linalg.svd(centered_matrix, full_matrices=False)

    # Compute participation ratio
    participation_ratio = (np.sum(s**2)**2) / np.sum(s**4)

    return participation_ratio



def generate_heterogeneous_thresholds(V_th_mean, V_th_std, N, rng):
    """
    Generate heterogeneous threshold potentials for N neurons.
    """
    if V_th_std == 0:
        V_th = np.full(N, V_th_mean)
    else:
        V_th = rng.uniform(V_th_mean - V_th_std * np.sqrt(3),
                           V_th_mean + V_th_std * np.sqrt(3), N)
    return V_th



def fft_convolution_with_padding(signal, kernel):
    """
    Perform linear convolution using FFT with proper zero-padding.

    Args:
        signal (numpy.ndarray): Input signal.
        kernel (numpy.ndarray): Impulse response or kernel.

    Returns:
        numpy.ndarray: Linearly convolved signal.
    """
    padded_length = len(signal) + len(kernel) - 1
    padded_signal = np.pad(signal, (0, padded_length - len(signal)))
    padded_kernel = np.pad(kernel, (0, padded_length - len(kernel)))

    convolved = np.fft.ifft(np.fft.fft(padded_signal) * np.fft.fft(padded_kernel))

    return np.real(convolved[:len(signal)])


def generate_input_bumps(duration_ms, n_input_neurons, dt, noise_std=0.2):
    """
    Generate a potential with two large bumps at the edges and a smaller, lower bump in the middle.

    Parameters:
    x: array of x values
    a: controls the position of the bumps
    b: controls the width of the potential
    max_height: height of the outer peaks
    mid_height: height of the middle peak (should be less than max_height)
    dip: how much to pull down the middle section
    """
    a=1.0
    b=2.0
    max_height=3.0
    mid_height=0.8
    dip=0.3

    num_samples = int(duration_ms / dt)
    x = np.linspace(-2, 2, num_samples)
    # Ensure mid_height is less than max_height
    mid_height = min(mid_height, max_height - 0.5)

    # Create the basic shape with three bumps
    V = max_height * (np.exp(-(x+a)**2/0.4) + np.exp(-(x-a)**2/0.2)) + mid_height * np.exp(-x**2/0.4)

    # Pull down the middle section
    # V -= dip * np.exp(-x**2/40)

    # Ensure it goes to zero at the edges
    V *= (1 - (x/b)**2)**2

    # Normalize to ensure max_height is reached
    V *= max_height / np.max(V)

    # Ensure no negative values
    V = np.maximum(V, 0)

    current = V

    return current



def generate_input_sine(duration_ms, n_input_neurons, dt, amplitude=2, noise_std=0.2):
    """
    Generate sinusoidal current with Gaussian noise for input neurons.

    Parameters:
    duration_ms (float): Duration of the simulation in milliseconds
    input_neurons_dyn (numpy.ndarray): Array of neuron indices receiving dynamic input
    amplitude (float): Amplitude of the sine wave (default: 0.5)
    noise_std (float): Standard deviation of the Gaussian noise (default: 0.1)

    Returns:
    numpy.ndarray: Normalized current for each input neuron

    Global variables used:
    dt (float): Time step in milliseconds
    """
    num_samples = int(duration_ms / dt)
    time = np.arange(num_samples) * dt

    # Generate base sine wave with a period of 400 ms
    frequency = 2.5  # 2.5 Hz for 400 ms period
    base_sine = amplitude * np.sin(2 * np.pi * frequency * time / 1000)

    # Keep only the upper half of the sine wave
    current = np.maximum(base_sine, 0)

    # current = np.repeat(current.reshape(1,-1), n_input_neurons, axis=0)

     # Generate unique Gaussian noise for each neuron and add to sine wave
    current = current


    return current








data_to_save = {}
data_to_save["sine"] = generate_input_sine(200, 1, 0.1)
data_to_save["bump"] = generate_input_bumps(300, 1, 0.1)


V_th_std_all = np.linspace(0, 2, 9)


# Initialize lists to store values
firing_means = []
firing_stds = []

dimensionality_values = []

stim_rmse_means = []
stim_rmse_stds = []
network_rmse_means = []
network_rmse_stds = []

decoded_test_signal_network = None
decoded_test_signal_stim = None

V_th_repeated = None
weights_flattened = None

stim_rmse_means_gen = []
stim_rmse_stds_gen = []
network_rmse_means_gen = []
network_rmse_stds_gen = []

decoded_test_signal_network_gen = None
decoded_test_signal_stim_gen = None

for V_th_std in V_th_std_all:
    filename = f'/Users/frazi/scripts/HLIF_7_mac/data/spikes_V_th_std_{V_th_std}_trial_{0}.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # Extract spike data
    spikes = data['spikes']
    stim_neuron_ids = data["input_1_neurons"]

    spike_matrix = spikes_to_matrix(spikes, 20000, 10000, 0.1)
    mean_rate = (spike_matrix.sum(axis=0) / 2).mean()
    std_rate = (spike_matrix.sum(axis=0) / 2).std()

    firing_means.append(mean_rate)
    firing_stds.append(std_rate)

    # Compute participation ratio
    n_bins = 250  # Number of bins to sum over
    spike_matrix_reduced = spike_matrix.reshape(-1, n_bins, spike_matrix.shape[1]).sum(axis=1)
    dimensionality = compute_participation_ratio(spike_matrix_reduced)
    dimensionality_values.append(dimensionality)



    filename = f"/Users/frazi/scripts/HLIF_7_mac/data/DECODING_V_th_std_{V_th_std}.pkl"
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    stim_rmse_means.append(np.mean(data['test_rmse_stim']))
    stim_rmse_stds.append(np.std(data['test_rmse_stim']))
    network_rmse_means.append(np.mean(data['test_rmse_network']))
    network_rmse_stds.append(np.std(data['test_rmse_network']))



    weights_stim = data["weights_stim"]
    weights_network = data["weights_network"]




    if V_th_std == 2.0:
        decoded_test_signal_stim = data["decoded_test_signal_stim"]
        decoded_test_signal_network = data["decoded_test_signal_network"]



        rng = [np.random.default_rng(np.random.SeedSequence(entropy=654321, spawn_key=(0, 0, 0, i))) for i in range(4)]
        # Generate thresholds
        V_th_mean = -55
        V_th = generate_heterogeneous_thresholds(V_th_mean, V_th_std, 10000, rng[0])
        # Repeat V_th for each trial
        V_th_repeated = np.tile(V_th, 10)

        weights_flattened = weights_network.flatten()




    # Load generalization data
    filename = f"/Users/frazi/scripts/HLIF_7_mac/data/spikes_generalization_V_th_std_{V_th_std}_trial_0.pkl"
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    spike_data = data["spikes"]

    # Convert spike data into spike matrices
    spike_matrix = spikes_to_matrix(spike_data, 3000, 10000, 0.1)

    # Apply convolution with an exponential kernel to smooth spikes
    tau = 10
    kernel = np.exp(-np.arange(0, 5 * tau, 0.1) / tau)
    filtered_spikes = np.array([fft_convolution_with_padding(spike_matrix[:, j], kernel) for j in range(10000)]).T

    # Compute generalization RMSE
    X_stim_gen = filtered_spikes[:, stim_neuron_ids]
    X_network_gen = filtered_spikes

    rmse_stim = []
    rmse_network = []

    for i in range(10):
        y_pred_stim_gen = X_stim_gen.dot(weights_stim[i])
        y_pred_network_gen = X_network_gen.dot(weights_network[i])

        rmse_stim.append(np.sqrt(np.mean((y_pred_stim_gen - data_to_save["bump"]) ** 2)))
        rmse_network.append(np.sqrt(np.mean((y_pred_network_gen - data_to_save["bump"]) ** 2)))


    stim_rmse_means_gen.append(np.mean(rmse_stim))
    stim_rmse_stds_gen.append(np.std(rmse_stim))
    network_rmse_means_gen.append(np.mean(rmse_network))
    network_rmse_stds_gen.append(np.std(rmse_network))


    if V_th_std == 2.0:

        decoded_test_signal_stim_gen = y_pred_stim_gen
        decoded_test_signal_network_gen = y_pred_network_gen





data_to_save["V_th_std_all"] = V_th_std_all
data_to_save["firing_means"] = firing_means
data_to_save["firing_stds"] = firing_stds


data_to_save["dimensionality_values"] = dimensionality_values


data_to_save["stim_rmse_means"] = stim_rmse_means
data_to_save["stim_rmse_stds"] = stim_rmse_stds
data_to_save["network_rmse_means"] = network_rmse_means
data_to_save["network_rmse_stds"] = network_rmse_stds
data_to_save["decoded_test_signal_stim"] = decoded_test_signal_stim
data_to_save["decoded_test_signal_network"] = decoded_test_signal_network


data_to_save["V_th_repeated"] = V_th_repeated
data_to_save["weights_flattened"] = weights_flattened


data_to_save["stim_rmse_means_gen"] = stim_rmse_means_gen
data_to_save["stim_rmse_stds_gen"] = stim_rmse_stds_gen
data_to_save["network_rmse_means_gen"] = network_rmse_means_gen
data_to_save["network_rmse_stds_gen"] = network_rmse_stds_gen
data_to_save["decoded_test_signal_stim_gen"] = decoded_test_signal_stim_gen
data_to_save["decoded_test_signal_network_gen"] = decoded_test_signal_network_gen



# Save the plot data
with open('./data/plot_data.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)
