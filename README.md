# higher-threshold-neurons-info-encoding

This repository contains simulation scripts, data processing routines, and analysis tools for the CNS 2025 abstract: *Higher-Threshold Neurons Boost Information Encoding in Spiking Neural Networks*. The study explores how spike threshold heterogeneity enhances information encoding in spiking neural networks, focusing on the role of higher-threshold neurons.

## Overview

The brain exhibits remarkable neural heterogeneity, which has been shown to improve sequential tasks, efficient coding, and working memory. This project investigates how heterogeneity in spike thresholds enhances information encoding by reducing trial-to-trial variability in network responses. Using a recurrent spiking network of leaky integrate-and-fire (LIF) neurons, we demonstrate that increasing spike threshold heterogeneity improves firing rate variability, network dimensionality, and decoding accuracy. Notably, higher-threshold neurons play a critical role in these improvements.

For more details, refer to the figure summarizing the results:  
[**FR_FZ_CNS2025.pdf**](FR_FZ_CNS2025.pdf)

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Repository Structure

```
higher-threshold-neurons-info-encoding/
├── Decoding.ipynb # Jupyter notebook for developing the decoding script
├── Decoding.py # Python script for decoding stimuli from neural responses
├── JOBS.sh # Bash script to run all simulations and analyses
├── plot_data.ipynb # Jupyter notebook for developing data visualization routines
├── plot_data.py # Python script for plotting results
├── process_data.ipynb # Jupyter notebook for developing data processing routines
├── process_data.py # Python script for processing simulation data
├── README.md # Documentation file (this file)
├── LICENSE.txt  # The license file for the project.
└── FR_FZ_CNS2025.pdf # Figure summarizing key results from the study

```


### Script Descriptions

- **Decoding.ipynb**: This Jupyter notebook is used for developing and testing the decoding algorithm. It includes exploratory analysis of neural responses, implementation of the linear decoder, and evaluation of decoding performance across different levels of neural heterogeneity.

- **Decoding.py**: This script implements the finalized decoding algorithm. It takes processed neural activity data as input, applies the linear decoder to reconstruct the original input stimuli, and outputs the decoding performance metrics such as RMSE for different network configurations.

- **JOBS.sh**: This Bash script orchestrates the entire simulation and analysis pipeline. It sequentially runs process_data.py, Decoding.py, and plot_data.py, ensuring all analyses are performed in the correct order and with the necessary inputs.

- **plot_data.ipynb**: This notebook is used for developing and refining data visualization routines. It includes code for creating various plots such as firing rate distributions, participation ratios, and decoding performance comparisons.

- **plot_data.py**: This script generates all the figures for the study. It takes the outputs from process_data.py and Decoding.py to create visualizations of network characteristics, decoding performance, and the role of higher-threshold neurons in information encoding.

- **process_data.ipynb**: This notebook is used for developing data processing routines. It includes code for loading raw simulation data, calculating firing rates, computing participation ratios, and preparing data for decoding and visualization.

- **process_data.py**: This script implements the data processing pipeline. It takes raw simulation data as input, performs necessary calculations and transformations, and outputs processed data ready for decoding and plotting.

These scripts work together to simulate neural networks with varying levels of threshold heterogeneity, analyze their information encoding capabilities, and visualize the results, supporting the study's conclusion that higher-threshold neurons boost information encoding in spiking neural networks.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Prerequisites

This project requires two custom Python modules to be installed:

- **balanced_spiking_network**: A package for simulating balanced spiking neural networks.
- **linear_decoder**: A package for decoding stimuli from neural responses.

Install these modules using pip:

```
!pip install git+ssh://git@github.com/fraziphy/linear-decoder.git
!pip install git+ssh://git@github.com/fraziphy/balanced-spiking-network.git
```

Ensure you have Python 3.7 or higher installed.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Setup

Before running the simulations:

1. Create a directory named `data` to store input data:

```
mkdir data
```


2. Create a directory named `outputs` to store output files:

```
mkdir outputs
```

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Running Simulations

To run all scripts and simulations, execute the following command in your terminal:

```
bash JOBS.sh
```

This will sequentially run all the scripts defined in `JOBS.sh`.

### Checking Job Status

After running the simulations, you can check if all jobs were successful by running the following command:

```
cat_log outputs/output_*
```

Here is the Bash function `cat_log` that checks job statuses:

```
cat_log() {
local all_zero=true
local non_zero_files=()
for file in "$@"; do
    if [ -f "$file" ]; then
        exit_status=$(awk '/^Exit:/ {print $2}' "$file")
        if [ -z "$exit_status" ] || [ "$exit_status" != "0" ]; then
            all_zero=false
            non_zero_files+=("$file")
        fi
    fi
done

if $all_zero; then
    echo " "
    echo "----------------------------------------"
    echo "Great Job!"
    echo "----------------------------------------"
    echo " "
else
    echo " "
    echo "----------------------------------------"
    echo "Files with non-zero exit status or missing exit status:"
    echo "----------------------------------------"
    printf '%s\n' "${non_zero_files[@]}"
    echo "----------------------------------------"
    echo " "
fi
}
```

### Example Output

1. If all jobs are successful:

```

----------------------------------------
Great Job!
----------------------------------------


```

2. If some jobs failed:

```

----------------------------------------
Files with non-zero exit status or missing exit status:
----------------------------------------
outputs/output_decoding.log
outputs/output_plot_data.log
----------------------------------------


```

In case of failures, review the listed output files to identify issues. After addressing the problems, modify `JOBS.sh` to re-run only the failed scripts.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Development Notes

- The `.ipynb` files are Jupyter notebooks used for developing and testing their corresponding `.py` scripts.
- Once development is complete, ensure that changes in the notebooks are reflected in the Python scripts.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Methods Summary

The study uses a recurrent spiking network of LIF neurons with varying spike thresholds to simulate neural heterogeneity. Key steps include:

1. **Simulation**: Generate spiking activity using `balanced_spiking_network`.
2. **Decoding**: Train a linear decoder using `linear_decoder` to decode input stimuli from network responses.
3. **Analysis**: Evaluate information encoding by comparing RMSE between decoded and original inputs.

For more details on methods and results, refer to the CNS 2025 abstract.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Acknowledgments

This work was supported by the Dutch Research Council (NWO Vidi grant VI.Vidi.213.137).

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Contributing

Thank you for considering contributing to our project! We welcome contributions from the community to help improve our project and make it even better. To ensure a smooth contribution process, please follow these guidelines:

1. **Fork the Repository**: Fork our repository to your GitHub account and clone it to your local machine.

2. **Branching Strategy**: Create a new branch for your contribution. Use a descriptive branch name that reflects the purpose of your changes.

3. **Code Style**: Follow our coding standards and style guidelines. Make sure your code adheres to the existing conventions to maintain consistency across the project.

4. **Pull Request Process**:
    Before starting work, check the issue tracker to see if your contribution aligns with any existing issues or feature requests.
    Create a new branch for your contribution and make your changes.
    Commit your changes with clear and descriptive messages explaining the purpose of each commit.
    Once you are ready to submit your changes, push your branch to your forked repository.
    Submit a pull request to the main repository's develop branch. Provide a detailed description of your changes and reference any relevant issues or pull requests.

5. **Code Review**: Expect feedback and review from our maintainers or contributors. Address any comments or suggestions provided during the review process.

6. **Testing**: Ensure that your contribution is properly tested. Write unit tests or integration tests as necessary to validate your changes. Make sure all tests pass before submitting your pull request.

7. **Documentation**: Update the project's documentation to reflect your changes. Include any necessary documentation updates, such as code comments, README modifications, or user guides.

8. **License Agreement**: By contributing to our project, you agree to license your contributions under the terms of the project's license (GNU General Public License v3.0).

9. **Be Respectful**: Respect the opinions and efforts of other contributors. Maintain a positive and collaborative attitude throughout the contribution process.

We appreciate your contributions and look forward to working with you to improve our project! If you have any questions or need further assistance, please don't hesitate to reach out to us.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Credits

- **Author:** [Farhad Razi](https://github.com/fraziphy)

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE)

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Contact

- **Contact information:** [email](farhad.razi.1988@gmail.com)
# higher-threshold-neurons-info-encoding
