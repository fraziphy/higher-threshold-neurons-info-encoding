#!/bin/bash

# Total number of CPUs
total_cpus=$(sysctl -n hw.ncpu)   # On MacOS
# total_cpus=$(nproc)   # On Linux

# Counter for jobs
counter=0

# Temporary file to capture the script output
TMP_OUTPUT="/tmp/script_output.log"

# Function to calculate the current number of used CPUs
get_used_cpus() {
    ps -A -o %cpu | awk '{s+=$1} END {print int(s/100)}'   # On MacOS
}

# function get_used_cpus() {
#     ps -eo %cpu --no-headers | awk '{s+=$1} END {printf "%.0f", s/100}'    # On Linux
# }

# Start logging
exec > >(tee -a $TMP_OUTPUT) 2>&1


V_th_std_all="0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0"

# Main script logic
{
    echo "Script for stimulated networks started at $(date)"
    for V_th_std in $V_th_std_all; do
        echo " "
        for trial in {0..19}; do
            # Calculate currently used CPUs
            used_cpus=$(get_used_cpus)
            free_cpus=$((total_cpus - used_cpus))

            # Ensure there are at least 7 CPUs free for this job
            while [ $free_cpus -lt 7 ]; do
                # Wait for any job to finish
                sleep 5
                # Recalculate used and free CPUs
                used_cpus=$(get_used_cpus)
                free_cpus=$((total_cpus - used_cpus))
            done

            # Start bsn job
            echo "Starting bsn for V_th_std=${V_th_std}, trial=${trial}"

            output_file="data/spikes_sine_V_th_std_${V_th_std}_trial_${trial}.pkl"

#             nohup bash -c '/usr/bin/time -p bsn --duration 200 --burn_in 800 --mu_1 sine --mu_zero 31 --trial "$1" --V_th_std "$2" --output "$3"; echo Exit: $?' -- "${trial}" "${V_th_std}" "${output_file}" &> "outputs/output_bsn_V_th_std_${V_th_std}_trial_${trial}.log" &

            # Increment counter to track jobs
#             ((counter += 1))
        done
    done
    echo " "
    echo " "
    echo " "
    echo " "
    echo "Script for dimensionality started at $(date)"
    for V_th_std in $V_th_std_all; do
        echo " "
        for trial in {0..0}; do
            # Calculate currently used CPUs
            used_cpus=$(get_used_cpus)
            free_cpus=$((total_cpus - used_cpus))

            # Ensure there are at least 7 CPUs free for this job
            while [ $free_cpus -lt 7 ]; do
                # Wait for any job to finish
                sleep 5
                # Recalculate used and free CPUs
                used_cpus=$(get_used_cpus)
                free_cpus=$((total_cpus - used_cpus))
            done

            # Start bsn job
            echo "Starting bsn for V_th_std=${V_th_std}, trial=${trial}"

            output_file="data/spikes_V_th_std_${V_th_std}_trial_${trial}.pkl"

#             nohup bash -c '/usr/bin/time -p bsn --duration 2000 --burn_in 800 --mu_1 none --mu_zero 31 --trial "$1" --V_th_std "$2" --output "$3"; echo Exit: $?' -- "${trial}" "${V_th_std}" "${output_file}" &> "outputs/output_bsn_dimensionality_V_th_std_${V_th_std}_trial_${trial}.log" &

            # Increment counter to track jobs
#             ((counter += 1))
        done
    done
    echo " "
    echo " "
    echo " "
    echo " "
    echo "Script for generalization started at $(date)"
    for V_th_std in $V_th_std_all; do
        echo " "
        for trial in {0..0}; do
            # Calculate currently used CPUs
            used_cpus=$(get_used_cpus)
            free_cpus=$((total_cpus - used_cpus))

            # Ensure there are at least 7 CPUs free for this job
            while [ $free_cpus -lt 7 ]; do
                # Wait for any job to finish
                sleep 5
                # Recalculate used and free CPUs
                used_cpus=$(get_used_cpus)
                free_cpus=$((total_cpus - used_cpus))
            done

            # Start bsn job
            echo "Starting bsn for V_th_std=${V_th_std}, trial=${trial}"

            output_file="data/spikes_generalization_V_th_std_${V_th_std}_trial_${trial}.pkl"

            nohup bash -c '/usr/bin/time -p bsn --duration 300 --burn_in 800 --mu_1 bumps --mu_zero 31 --trial "$1" --V_th_std "$2" --output "$3"; echo Exit: $?' -- "${trial}" "${V_th_std}" "${output_file}" &> "outputs/output_bsn_generalization_V_th_std_${V_th_std}_trial_${trial}.log" &

            # Increment counter to track jobs
            ((counter += 1))
        done
    done

    # Wait for all remaining jobs to finish
    echo "Waiting for all jobs to finish..."
    wait
    echo "All jobs completed at $(date)"
    echo " "
    echo " "
    echo " "
    echo " "
    echo "Script for decoding started at $(date)"
    for V_th_std in $V_th_std_all; do
        echo " "
        # Calculate currently used CPUs
        used_cpus=$(get_used_cpus)
        free_cpus=$((total_cpus - used_cpus))

        # Ensure there are at least 7 CPUs free for this job
        while [ $free_cpus -lt 7 ]; do
            # Wait for any job to finish
            sleep 5
            # Recalculate used and free CPUs
            used_cpus=$(get_used_cpus)
            free_cpus=$((total_cpus - used_cpus))
        done

        # Start decoding job
        echo "Starting Decoding.py script for V_th_std=${V_th_std}"

        nohup bash -c '/usr/bin/time -p /opt/anaconda3/bin/python /Users/frazi/scripts/HLIF_7_mac/Decoding.py "$1"; echo Exit: $?' -- "${V_th_std}" &> /Users/frazi/scripts/HLIF_7_mac/outputs/output_Decoding_V_th_std_"${V_th_std}".log &

        # Increment counter to track jobs
        ((counter += 1))
    done
    echo " "
    echo " "
    # Wait for all remaining jobs to finish
    echo "Waiting for all decoding jobs to finish..."
    wait
    echo "All jobs completed at $(date)"
    echo " "
    echo " "
    echo " "
    echo " "
    echo "Script for processing data started at $(date)"

    # Start process_data job
    echo "Starting process_data.py script for V_th_std=${V_th_std}"

    nohup bash -c '/usr/bin/time -p /opt/anaconda3/bin/python /Users/frazi/scripts/HLIF_7_mac/process_data.py; echo Exit: $?' &> /Users/frazi/scripts/HLIF_7_mac/outputs/output_process_data.log &

    echo " "
    echo " "
    # Wait for all remaining jobs to finish
    echo "Waiting for the processing data job to finish..."
    wait
    echo "The processing data job completed at $(date)"
    echo " "
    echo " "
    echo " "
    echo " "
    echo "Script for plotting data started at $(date)"

    # Start plot_data job
    echo "Starting plot_data.py script for V_th_std=${V_th_std}"

    nohup bash -c '/usr/bin/time -p /opt/anaconda3/bin/python /Users/frazi/scripts/HLIF_7_mac/plot_data.py; echo Exit: $?' &> /Users/frazi/scripts/HLIF_7_mac/outputs/output_plot_data.log &

    echo " "
    echo " "
    # Wait for all remaining jobs to finish
    echo "Waiting for the plot data job to finish..."
    wait
    echo "The plot_data job completed at $(date)"
} | tee -a $TMP_OUTPUT

# Clean up
rm "$TMP_OUTPUT"
