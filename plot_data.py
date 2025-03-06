import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Wedge
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import pickle
from matplotlib.ticker import ScalarFormatter


class LeftAlignedLineHandler_panelD(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        x_offset = -1.  # Move the marker to the left edge
        artists = super().create_artists(legend, orig_handle, x_offset, ydescent, width, height, fontsize, trans)
        return artists


# Load the saved data
with open('./data/plot_data.pkl', 'rb') as f:
    data_to_plot = pickle.load(f)


V_th_std_all = data_to_plot["V_th_std_all"]
firing_means = data_to_plot["firing_means"]
firing_stds = data_to_plot["firing_stds"]


dimensionality_values = data_to_plot["dimensionality_values"]


stim_rmse_means = data_to_plot["stim_rmse_means"]
stim_rmse_stds = data_to_plot["stim_rmse_stds"]
network_rmse_means = data_to_plot["network_rmse_means"]
network_rmse_stds = data_to_plot["network_rmse_stds"]
decoded_test_signal_stim = data_to_plot["decoded_test_signal_stim"]
decoded_test_signal_network = data_to_plot["decoded_test_signal_network"]


V_th_repeated = data_to_plot["V_th_repeated"]
weights_flattened = data_to_plot["weights_flattened"]


stim_rmse_means_gen = data_to_plot["stim_rmse_means_gen"]
stim_rmse_stds_gen = data_to_plot["stim_rmse_stds_gen"]
network_rmse_means_gen = data_to_plot["network_rmse_means_gen"]
network_rmse_stds_gen = data_to_plot["network_rmse_stds_gen"]
decoded_test_signal_stim_gen = data_to_plot["decoded_test_signal_stim_gen"]
decoded_test_signal_network_gen = data_to_plot["decoded_test_signal_network_gen"]


def create_figure(fig_format):
    # Set random seed for consistency
    np.random.seed(42)
    def exp_function(x, a=1, b=1, c=0):
        return a * np.exp(b * x) + c
    # Set the desired pixel dimensions and DPI
    width_pixels = 830  # Increased width
    height_pixels = 600
    dpi = 100

    # Calculate figure size in inches
    width_inches = width_pixels / dpi
    height_inches = height_pixels / dpi

    # Create figure with calculated dimensions
    fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)

    # Create Panel A (top half)
    ax_a = fig.add_axes([0.1, -0.4+0., 0.4, 0.4+0.62])  # Reduced width for Panel A  [left, bottom, width, height]
    ax_a.set_xlim(0, 7.2)  # Adjusted x limits
    ax_a.set_ylim(-3, 4.6)
    ax_a.axis('off')

    result_axes = []
    for i in range(2):
        for j in range(2):
            if i ==1 and j ==0:
                ax_sub = fig.add_axes([0.585 + j*0.375, -0.4+0.07 + i*0.54, 0.23, 0.2+0.19])  # Adjust these values as needed  [left, bottom, width, height]
                result_axes.append(ax_sub)
            else:
                ax_sub = fig.add_axes([0.585 + j*0.375, -0.4+0.07 + i*0.54, 0.28, 0.2+0.19])  # Adjust these values as needed  [left, bottom, width, height]
                result_axes.append(ax_sub)
    result_axes = np.array(result_axes).reshape(-1,2)




    # Panel A content
    # Upper part of sine wave using np.maximum()
    x = np.linspace(0, 1, 100)
    y = np.maximum(np.sin(1 * np.pi * x), 0) / 2 + 2.7
    ax_a.plot(x, y, 'k-', linewidth=2)

    # Arrow from sine wave to rectangle
    ax_a.add_patch(FancyArrowPatch((1.1, 3), (1.9, 3),
                                   arrowstyle='->', mutation_scale=20, color='k'))

    # Frozen input
    ax_a.add_patch(Rectangle((2.1, 2), 1, 2, fill=False))

    # Noisy signals inside frozen input
    for i in range(3):
        y_noise = np.maximum(np.sin(1 * np.pi * x), 0) / 2 + np.random.normal(0, 0.1, x.shape)
        ax_a.plot(-0.4+0.15+x/2 + 2.6, y_noise/2.6 + 2.35 + i*0.6, 'k-', linewidth=0.7, alpha=0.7)

    # Two sets of three vertical dots between signals
    for i in range(2):
        ax_a.plot(np.array([-0.4+3, -0.4+3, -0.4+3]), 0.1+np.array([2.6 + i*0.6, 2.7 + i*0.6, 2.8 + i*0.6]), 'k.', markersize=1.5)

    # Curved arrows from rectangle to neural network
    # arrow_paths = [((3.8, 2.9), (4.9, 3.), (4.95, 2.5)),  # bottom to top
    #                ((3.8, 3), (4.9, 3), (4.4, 3)),        # straight
    #                ((3.8, 3.1), (4.9, 3.), (4.95, 3.5))]  # top to bottom

    # for i, path in enumerate(arrow_paths):
    #     if i == 1:  # straight arrow
    #         ax_a.add_patch(FancyArrowPatch(path[0], path[1],
    #                                        arrowstyle='->', mutation_scale=20, color='g'))
    #     else:  # curved arrows
    #         ax_a.add_patch(FancyArrowPatch(path[0], path[2],
    #                                        connectionstyle=f"arc3,rad={0.3 if i == 0 else -0.3}",
    #                                        arrowstyle='->', mutation_scale=20, color='g'))
    ax_a.add_patch(FancyArrowPatch((3.35, 2.4), (4.15, 2.4),
                                   arrowstyle='->', mutation_scale=20, color='r'))
    ax_a.add_patch(FancyArrowPatch((3.35, 3), (4.15, 3),
                                   arrowstyle='->', mutation_scale=20, color='r'))
    ax_a.add_patch(FancyArrowPatch((3.35, 3.6), (4.15, 3.6),
                                   arrowstyle='->', mutation_scale=20, color='r'))

    # Two sets of three vertical dots between signals
    for i in range(2):
        ax_a.plot(np.array([3.75, 3.75, 3.75]), 0.1+np.array([2.6 + i*0.6, 2.7 + i*0.6, 2.8 + i*0.6]), 'k.', markersize=1.5)





    ax_a.add_patch(Rectangle((4.34, 1.53), 2.6, 2.51, facecolor=(0.92, 0.92, 0.92),edgecolor=(0.62, 0.62, 0.62)))
    ax_a.add_patch(Rectangle((4.27, 1.63), 2.6, 2.51, facecolor=(0.92, 0.92, 0.92),edgecolor=(0.62, 0.62, 0.62)))
    ax_a.add_patch(Rectangle((4.2, 1.73), 2.6, 2.51, facecolor=(0.92, 0.92, 0.92),edgecolor=(0.62, 0.62, 0.62)))

    ax_a.plot(np.array([4.05, 4.15, 4.25]), 0.1+np.array([1.5 , 1.4 , 1.3 ]), 'k.', markersize=1.5)
    ax_a.text(3.7, 1.4, 'Trials', ha='center', fontsize='medium')





    # Neural network (big circle with neurons)
    network_circle = Circle((5.5, 3), 1.2, fill=False)  # Shifted X position
    ax_a.add_artist(network_circle)

    # Add shaded area for stimulated neurons
    start_angle = 180 - 54
    end_angle = 180 + 54
    stimulated_area = Wedge((5.5, 3), 1.2, start_angle, end_angle, color='r', alpha=0.5) # Shifted X position
    ax_a.add_artist(stimulated_area)

    # Add label for stimulated neural subset
    label_angle = np.radians((start_angle + end_angle) / 2)
    label_x = 5.3 + 1.3 * np.cos(label_angle) # Shifted X position
    label_y = 2. + 1.3 * np.sin(label_angle)
    ax_a.text(+0.05+label_x, -0.03+label_y, 'Stim.\nSubset', ha='center', va='center', fontsize=8.9, color='r')

    # Create neurons inside the circle
    n_neurons = 40
    neuron_positions = []
    radius = 1.12  # Slightly smaller than the circle radius to ensure all neurons are inside

    for i in range(n_neurons):
        # Use golden ratio to distribute points evenly
        theta = 2 * np.pi * i / (1 + np.sqrt(5))
        r = radius * np.sqrt(i / n_neurons)

        # Add small random variation
        r += np.random.normal(0, 0.03)
        theta += np.random.normal(0, 0.1)

        # Convert to Cartesian coordinates
        x = 5.5 + r * np.cos(theta) # Shifted X position
        y = 3 + r * np.sin(theta)

        neuron_positions.append((x, y))
        ax_a.add_artist(Circle((x, y), 0.03, fill=True, color='k', alpha=0.6))

    # Add connections with probability 0.05
    connection_probability = 0.04
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i != j and np.random.random() < connection_probability:
                ax_a.add_patch(FancyArrowPatch(neuron_positions[i], neuron_positions[j],
                                               connectionstyle=f"arc3,rad={np.random.uniform(-0.3, 0.3)}",
                                               arrowstyle='->', mutation_scale=8, color='brown', alpha=0.3))





    ax_a.add_patch(Rectangle((4.34, -2.53- 0.24), 2.6, 2.51, facecolor=(0.92, 0.92, 0.92),edgecolor=(0.62, 0.62, 0.62)))
    ax_a.add_patch(Rectangle((4.27, -2.63- 0.24), 2.6, 2.51, facecolor=(0.92, 0.92, 0.92),edgecolor=(0.62, 0.62, 0.62)))
    ax_a.add_patch(Rectangle((4.2, -2.73- 0.24), 2.6, 2.51, facecolor=(0.92, 0.92, 0.92),edgecolor=(0.62, 0.62, 0.62)))







    # --- Example Spikes (Raster Plot) ---
    spike_x_start = 5  # Below the network
    spike_y_start = -2.5
    n_neurons_to_plot = 10

    # Add red background for the first three neurons
    ax_a.add_patch(Rectangle((spike_x_start - 0.6, spike_y_start + 1.28),  # x, y (lower-left corner)
                             2.3, 3 * 0.2 + 0.06,                  # width, height
                             facecolor='r', alpha=0.5,
                             edgecolor='none'))

    for i in range(n_neurons_to_plot):
        # Higher spiking rate for the first three neurons
        if i > 6:
            min_spikes = 10  # Increased minimum spikes
            max_spikes = 20  # Increased maximum spikes
        else:
            min_spikes = 3
            max_spikes = 8

        num_spikes = np.random.randint(min_spikes, max_spikes)
        spike_times = np.random.uniform(spike_x_start - 0.5, spike_x_start + 1.5, num_spikes)
        spike_times.sort()

        # Plot each spike as a vertical line
        for spike_time in spike_times:
            ax_a.plot([spike_time, spike_time], [spike_y_start + i * 0.2 - 0.06, spike_y_start + i * 0.2 + 0.06], 'k-', linewidth=1)




    # ax_a.add_patch(Rectangle((3.95, -0.185), 3.03, 0.95,
    #                      facecolor='none',
    #                      edgecolor="k",
    #                      linestyle='--',
    #                         linewidth =0.3))
    # ax_a.add_patch(Rectangle((3.95, -1.98), 3.03, 1.77,
    #                      facecolor='none',
    #                      edgecolor="k",
    #                      linestyle='--',
    #                         linewidth =0.3))






    # Add arrow down to spikes
    ax_a.add_patch(FancyArrowPatch((5.5, 1.2), (5.5, 0.3),
                                   arrowstyle='->', mutation_scale=15, color='k'))




    ax_a.add_patch(FancyArrowPatch((3.35, -1.7), (4.15, -1.7),
                                   arrowstyle='<-', mutation_scale=20, color='k'))
    # ax_a.add_patch(FancyArrowPatch((3.35, 0.15), (4.15, -0.15),
    #                                arrowstyle='<-', mutation_scale=20, color='k'))
    # ax_a.add_patch(FancyArrowPatch((3.35, -1.3), (4.15, -1.),
    #                                arrowstyle='<-', mutation_scale=20, color='k'))





    # --- Decoder and Reconstructed Signal (Moved to Left) ---
    # Decoder (square) - Shifted BELOW the network
    decoder_rect = Rectangle((2.1, -2.16), 1, 1, fill=False) #Position changed
    ax_a.add_patch(decoder_rect)

    # decoder_rect = Rectangle((2.1, -0.5), 1, 1, fill=False) #Position changed
    # ax_a.add_patch(decoder_rect)
    # decoder_rect = Rectangle((2.1, -1.95), 1, 1, fill=False) #Position changed
    # ax_a.add_patch(decoder_rect)

    # Add decaying exponential function inside decoder
    def exp_decay(x, a=1, b=1, c=0):
        return a * np.exp(-b * x) + c

    x_exp = np.linspace(0, 1, 100)
    y_exp = exp_decay(x_exp, a=0.8, b=3, c=0.1)

    # Plot the decaying exponential
    ax_a.plot(-0.09+x_exp * 0.8 + 2.3, -2.3+y_exp * 0.8 + 0.1, 'k-', linewidth=1.5) #Position Changed
    ax_a.text(0.08+2.7, -1.7, 'Exp.\nKer.', ha='center',fontsize=8) #Position Changed

    # # Plot the decaying exponential
    # ax_a.plot(-0.09+x_exp * 0.8 + 2.3, -0.5+y_exp * 0.8 + 0.1, 'k-', linewidth=1.5) #Position Changed
    # ax_a.plot(-0.09+x_exp * 0.8 + 2.3, -1.95 + y_exp * 0.8 + 0.1, 'k-', linewidth=1.5) #Position Changed

    # # Add Textual Abbreviation and equation
    # ax_a.text(0.08+2.7, 0., 'Exp.\nKer.', ha='center',fontsize=8) #Position Changed
    # ax_a.text(0.08+2.7, -1.95 + 0.5, 'Exp.\nKer.', ha='center',fontsize=8) #Position Changed




    # Arrow from sine wave to rectangle
    ax_a.add_patch(FancyArrowPatch((1.1, -1.7), (1.9, -1.7),
                                   arrowstyle='<-', mutation_scale=20, color='k'))
    # # Arrow from sine wave to rectangle
    # ax_a.add_patch(FancyArrowPatch((1.1, 0.), (1.9, 0.),
    #                                arrowstyle='<-', mutation_scale=20, color='k'))
    # # Arrow from sine wave to rectangle
    # ax_a.add_patch(FancyArrowPatch((1.1, -1.45), (1.9, -1.45),
    #                                arrowstyle='<-', mutation_scale=20, color='k'))




    # Reconstructed signal (Shifted Below the Hidden Signal)
    x = np.linspace(0, 1, 100)
    y = np.maximum(np.sin(1 * np.pi * x), 0) / 2 + 0.1 #Position Changed
    # Add noise
    noise = np.random.normal(0, 0.05, x.shape)
    y_noisy = -0.5+y + noise
    # Create dents (reconstructed signal effect)
    dent_positions = [20, 25, 26, 28, 30, 40, 50, 55, 60, 80]  # Positions where dents occur
    dent_size = 0.1  # Size of the dents
    for pos in dent_positions:
        y_noisy[pos-2:pos+2] -= dent_size

    # Plot the resulting signal
    ax_a.plot(x, -1.8+0.18+y_noisy, 'k-', linewidth=1)

    # noise = np.random.normal(0, 0.05, x.shape)
    # y_noisy = -1.79 + y/2. + noise
    # # Create dents (reconstructed signal effect)
    # dent_positions = np.array([23, 26, 28, 30, 40, 50, 55, 60, 80])-7  # Positions where dents occur
    # dent_size = 0.1  # Size of the dents
    # for pos in dent_positions:
    #     y_noisy[pos-2:pos+2] -= dent_size

    # # Plot the resulting signal
    # ax_a.plot(x, 0.18+y_noisy, 'k-', linewidth=1)



    # ---Labels and Text---
    ax_a.text(0.6, 4.3, 'Hidden\nInput', ha='center')
    ax_a.text(2.6, 4.5, 'Input Ensemble', ha='center')
    ax_a.text(5.5, 4.5, 'Spiking Network', ha='center')

    ax_a.text(5.5, -0., 'Spiking Responses', ha='center')
    ax_a.text(2.6, -0.2, 'Linear\nDecoder', ha='center')
    ax_a.text(.6, -0.2, 'Decoded\nInput', ha='center')

    # # Moved Text Labels
    # ax_a.text(2.7, 1.4, 'Linear\nDecoder', ha='center')
    # ax_a.text(0.5, 1.4, 'Decoded\nStimulus', ha='center')

    ax_a.text(-0.2, 4.7, 'A', fontweight='bold')
    ax_a.text(7.4, 4.7, 'B', fontweight='bold')
    ax_a.text(14.3, 4.7, 'D', fontweight='bold')
    ax_a.text(7.4, 0.7, 'C', fontweight='bold')
    ax_a.text(14.3, 0.7, 'E', fontweight='bold')


    # ---Panel B (Moved to Right)---
    # ax_b = fig.add_axes([0.55, 0.5, 0.4, 0.1])  # [left, bottom, width, height]
    # ax_b.axis('off')

    # Create 2x3 grid of subplots in Panel B



    # First subplot: Silent Neurons on the left y-axis
    color_teal = '#008080'  # Teal color
    errorbar = result_axes[1,0].errorbar(V_th_std_all, firing_means, yerr=firing_stds, fmt='o-',
                                         color=color_teal, capsize=5, capthick=2, label='Firing', alpha=0.8)
    result_axes[1,0].set_xlabel(r"$\theta_{\text{th}}^{\text{std}}$"+" (mV)")
    result_axes[1,0].set_ylabel('Mean Firing Rate (Hz)', color=color_teal)
    result_axes[1,0].tick_params(axis='y', labelcolor=color_teal)
    result_axes[1,0].grid(True, alpha=0.3)
    result_axes[1,0].set_ylim(80,120)
    result_axes[1,0].set_yticks([90,100,110])


    # Create a twin axes sharing the same x-axis
    ax2 = result_axes[1,0].twinx()

    # Second subplot: Dimensionality on the right y-axis
    color_magenta = '#FF00FF'  # Magenta color
    line2, = ax2.plot(V_th_std_all, dimensionality_values, "o-", color=color_magenta, label='Dimensionality', alpha=0.8)
    ax2.set_ylabel('Participation Ratio', color=color_magenta)
    ax2.tick_params(axis='y', labelcolor=color_magenta)

    # Create legend entries manually
    # Create custom error bar object
    error_bar = result_axes[1,0].errorbar([0], [0], yerr=[0], fmt='o-', color=color_teal, capsize=3, capthick=1., label='Firing Rate')

    legend_elements = [
        error_bar,  # This will include the error bars
        Line2D([0], [0], color=color_magenta, marker='o', linestyle='-', label='Dimensionality')
    ]

    # Add combined legend with custom handler
    result_axes[1,0].legend(handles=legend_elements,
                            handler_map={error_bar: HandlerErrorbar(xerr_size=0.5, yerr_size=0.6)},
                            fontsize='small',
                            markerscale=0.9,
                            handlelength=1.2,
                            bbox_to_anchor=(0.8, 0.22))
    result_axes[1,0].set_title("Network Characteristic")





    result_axes[0,0].errorbar(V_th_std_all, stim_rmse_means, yerr=stim_rmse_stds,
                     fmt='o-', color="r", capsize=5, label="Stim. Subset", alpha=0.6)
    result_axes[0,0].errorbar(V_th_std_all, network_rmse_means, yerr=network_rmse_stds,
                     fmt='s-', color="b", capsize=5, label="Whole Net.", alpha=0.6)
    result_axes[0,0].set_xlabel(r"$\theta_{\text{th}}^{\text{std}}$"+" (mV)")
    result_axes[0,0].set_ylabel("RMSE (mV)")
    result_axes[0,0].set_title("Decoding Performance")
    result_axes[0,0].grid(True, alpha=0.3)
    result_axes[0,0].legend(loc='upper center', bbox_to_anchor=(0.24, 1.015), ncol=1, fontsize='small', markerscale=0.7, handlelength=1)

    ylim = result_axes[0,0].get_ylim()
    result_axes[0,0].set_ylim([ylim[0],2.1])
    result_axes[0,0].set_yticks([0.5, 1, 1.5, 2.])

    # Inset for first subplot (Hidden Input)
    axins1 = result_axes[0,0].inset_axes([0.52, 0.49, 0.45, 0.4])  # Top right position

    axins1.plot(np.arange(2000), decoded_test_signal_stim, 'r-', label='Decoded Input', alpha=0.6)
    axins1.plot(np.arange(2000), decoded_test_signal_network, 'b-', label='Decoded Input', alpha=0.6)
    axins1.plot(np.arange(2000), data_to_plot["sine"], 'k-', label='Hidden Input')
    axins1.set_title(r'$\theta_{\text{th}}^{\text{std}}$ = 2' + " mV", fontsize='medium')
    axins1.set_xlabel('Time (ms)', fontsize='small', labelpad=-2.5)
    axins1.set_ylabel('Amplitude', fontsize='small', labelpad=2.5)
    axins1.tick_params(axis='both', which='major', labelsize='small')

    axins1.set_xticks([0,2000],labels=[0,200])
    axins1.set_yticks([0, 2])

    # Adjust the position of the axes
    axins1.xaxis.set_tick_params(pad=1.8)
    axins1.yaxis.set_tick_params(pad=1.8)









    counts, xedges, yedges, im = result_axes[1,1].hist2d(V_th_repeated, weights_flattened, bins=50, cmap='viridis', norm=None)
    # Extract maximum non-zero weights for each V_th bin
    max_weights = []
    v_th_centers = []

    for i in range(len(xedges) - 1):
        bin_weights = weights_flattened[(V_th_repeated >= xedges[i]) & (V_th_repeated < xedges[i+1])]
        if len(bin_weights) > 0:
            non_zero_weights = bin_weights[bin_weights != 0]
            if len(non_zero_weights) > 0:
                max_weights.append(np.max(non_zero_weights))
                v_th_centers.append((xedges[i] + xedges[i+1]) / 2)

    max_weights = np.array(max_weights)
    v_th_centers = np.array(v_th_centers)

    # Define sigmoid function
    def sigmoid(x, L, x0, k, b):
        return L / (1 + np.exp(-k*(x-x0))) + b

    # Fit sigmoid function
    p0 = [max(max_weights), np.median(v_th_centers), 1, min(max_weights)]  # Initial guess
    popt, _ = curve_fit(sigmoid, v_th_centers, max_weights, p0=p0, maxfev=10000)

    # Generate points for the fitted curve
    v_th_fit = np.linspace(xedges[0], xedges[-1], 100)
    weights_fit = sigmoid(v_th_fit, *popt)

    # Add colorbar
    cbar = fig.colorbar(im, ax=result_axes[1,1])
    cbar.ax.set_title('Count', pad=1, fontsize=8)
    # Set custom ticks
    cbar.set_ticks([0, 90, 180])  # Replace with your desired tick values
    cbar.ax.tick_params(labelsize="small")

    # Plot sigmoid fit and scatter points

    result_axes[1,1].scatter(v_th_centers, max_weights, color='white', s=10, alpha=0.6, label='Max Weights')
    result_axes[1,1].plot(v_th_fit, weights_fit, 'r-', linewidth=2, label='Fit')

    # Set labels and title
    result_axes[1,1].set_xlabel(r'$\theta_{\text{th}}$'+" (mV)")
    result_axes[1,1].set_ylabel('Decoding Weight')
    result_axes[1,1].set_title('Decoding Efficacy')
    result_axes[1,1].legend(fontsize='small',
                            handlelength=1,
                            handletextpad=0.7,  # Remove padding between marker and text
                            borderpad=0.05,
                            borderaxespad=0.25,
                            handler_map={plt.Line2D: LeftAlignedLineHandler_panelD()},
                            loc='upper left',  # Set location to upper left
                            bbox_to_anchor=(0.002, 0.17))

    # Set y-axis to scientific notation
    result_axes[1,1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    result_axes[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Adjust the position of the exponent
    result_axes[1,1].yaxis.offsetText.set_fontsize(7)
    result_axes[1,1].yaxis.offsetText.set_position((-0.12, 1))



    # Second subplot
    result_axes[0,1].errorbar(V_th_std_all, stim_rmse_means_gen, yerr=stim_rmse_stds,
             fmt='o-', color="r", capsize=5, label="Stimulated Subset", alpha=0.6)
    result_axes[0,1].errorbar(V_th_std_all, network_rmse_means_gen, yerr=network_rmse_stds,
                 fmt='s-', color="b", capsize=5, label="Whole Network", alpha=0.6)

    result_axes[0,1].set_xlabel(r'$\theta_{\text{th}}^{\text{std}}$'+" (mV)")
    result_axes[0,1].set_ylabel("RMSE (mV)")
    result_axes[0,1].set_title("Generalization Performance")
    result_axes[0,1].grid(True, alpha=0.3)
    # result_axes[0,1].legend(loc='upper center', bbox_to_anchor=(0.3, 1.), ncol=1)
    result_axes[0,1].set_yticks([0.5, 1, 1.5, 2.])

    # Inset for second subplot (Generalization Input)
    axins2 = result_axes[0,1].inset_axes([0.41, 0.49, 0.56, 0.4])  # Top right position

    axins2.plot(decoded_test_signal_stim_gen, 'r-', alpha=0.6)
    axins2.plot(decoded_test_signal_network_gen, 'b-', alpha=0.6)
    axins2.plot(data_to_plot["bump"], 'k-', label='Novel Input')
    axins2.legend(fontsize='small', bbox_to_anchor=(0.05, 1.31), handlelength=0.8)
    axins2.set_title(r'$\theta_{\text{th}}^{\text{std}}$ = 2'+ " mV", fontsize='medium')
    axins2.set_xlabel('Time (ms)', fontsize='small', labelpad=-2.5)
    axins2.set_ylabel('Amplitude', fontsize='small', labelpad=2.5)
    axins2.tick_params(axis='both', which='major', labelsize='small')

    axins2.set_xticks([0,3000], labels=[0,300])
    axins2.set_yticks([0, 3])

    # Adjust the position of the axes
    axins2.xaxis.set_tick_params(pad=1.8)
    axins2.yaxis.set_tick_params(pad=1.8)



    ylim = result_axes[0,0].get_ylim()

    result_axes[0,0].set_ylim(ylim)
    result_axes[0,1].set_ylim(ylim)

    # Align y-labels
    fig.align_ylabels([result_axes[0,1],result_axes[1,1]])
    fig.align_ylabels([result_axes[0,0],result_axes[1,0]])

    fig.savefig(f'FR_FZ_CNS2025.{fig_format}', dpi=100, bbox_inches='tight')

    print(f"Image saved as FR_FZ_CNS2025.{fig_format}")
    plt.close()



for fig_format in ["pdf", "png"]:
    create_figure(fig_format)
