"""
Analysis Script: MSE vs. Shots

This script runs a Monte Carlo simulation to analyze how the
Mean Squared Error (MSE) of the quantum reconstruction
improves with an increasing number of shots.

It generates and saves 'output_mse_vs_shots_simulation.png'.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_aer import Aer
import time

# Import all functions from our new library
from quantum_convolution_lib import *

if __name__ == '__main__':
    
    # ================================================
    # %% 1. Define Problem
    # ================================================
    print("--- 1. Defining Problem ---")
    custom_image = np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1]
    ])
    
    # Problem Parameters
    H, W = custom_image.shape
    C = 1  # Channels
    M, R, S = 2, 4, 4  # 2 Filters, each 4x4
    
    # --- Monte Carlo Simulation Parameters ---
    monte_carlo_runs = 10  # Number of runs to average for each shot count
    # List of shots, e.g., [32, 64, 128, ..., 524288]
    shots_list = [32 * (2**i) for i in range(14)] 
    
    all_mse_results = np.zeros((len(shots_list), monte_carlo_runs))

    # ================================================
    # %% 2. Classical & Quantum Setup (Done Once)
    # ================================================
    print("--- 2. Setting up Classical and Quantum components ---")
    simulator = Aer.get_backend('qasm_simulator')
    
    K, (E, F), Y_classical, patch_norms = create_classical_problem_and_patches(
        custom_image, R, S, M
    )

    num_data_qubits = int(np.ceil(np.log2(R * S * C)))
    num_spatial_qubits = int(np.ceil(np.log2(E * F)))
    num_filter_qubits = int(np.ceil(np.log2(M)))

    print("Building Oracles and Hadamard Test circuit...")
    uk_oracle = get_uk_oracle_sliding_window(K, M, num_filter_qubits, num_data_qubits)
    ux_oracle = get_ux_oracle_sliding_window(
        custom_image, R, S, E, F, num_spatial_qubits, num_data_qubits
    )
    hadamard_test_qc = create_sliding_window_hadamard_test(
        uk_oracle, ux_oracle, M, E, F
    )
    
    print("Transpiling circuit for simulator...")
    transpiled_circuit = transpile(hadamard_test_qc, simulator)
    print("Circuit setup complete.")

    # ================================================
    # %% 3. Run Monte Carlo Simulation
    # ================================================
    print(f"\n--- Starting Monte Carlo simulation ({monte_carlo_runs} runs per shot count) ---")
    total_start_time = time.time()

    for i, shots in enumerate(shots_list):
        print(f"\nProcessing shot count: {shots} ({i+1}/{len(shots_list)})")
        
        for j in range(monte_carlo_runs):
            run_start_time = time.time()
            
            # Run the simulation
            result = simulator.run(transpiled_circuit, shots=shots, memory=False).result()
            counts = result.get_counts()
            
            # Reconstruct the output
            Y_reconstructed = reconstruct_from_counts(
                counts, K, patch_norms, M, E, F
            )
            
            # Calculate Mean Squared Error
            mse = np.mean((Y_classical - Y_reconstructed)**2)
            all_mse_results[i, j] = mse
            
            run_end_time = time.time()
            print(f"  Run {j+1}/{monte_carlo_runs} -> MSE: {mse:.6f} (took {run_end_time - run_start_time:.2f}s)")

    print(f"\n--- Simulation finished in {(time.time() - total_start_time)/60:.2f} minutes ---")

    # ================================================
    # %% 4. Process and Plot Results
    # ================================================
    print("\n--- 4. Plotting averaged MSE vs. Shots ---")
    
    # Calculate mean and standard deviation across the Monte Carlo runs
    mean_mses = np.mean(all_mse_results, axis=1)
    std_dev_mses = np.std(all_mse_results, axis=1)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot mean MSE with error bars (std dev)
    ax1.errorbar(shots_list, mean_mses, yerr=std_dev_mses, fmt='o-', capsize=5, label='Mean MSE with Std Dev', color='b')

    # Plot the ideal 1/N scaling line for reference
    # We scale it to the first data point
    inverse_scaling = mean_mses[0] * (shots_list[0] / np.array(shots_list))
    ax1.plot(shots_list, inverse_scaling, 'r--', label=r'Reference $1/N_{shots}$ scaling')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Shots ($N_{shots}$)', fontsize=14)
    ax1.set_ylabel('Mean Squared Error (MSE)', fontsize=14)
    ax1.set_title(f'Quantum Convolution MSE vs. Shots (Averaged over {monte_carlo_runs} runs)', fontsize=16)
    ax1.grid(True, which="both", ls="--")
    ax1.legend(fontsize=12)
    plt.tight_layout()

    output_filename_mse = "output_mse_vs_shots_simulation.png"
    plt.savefig(output_filename_mse, dpi=300, bbox_inches='tight')
    print(f"MSE plot saved as '{output_filename_mse}'")
    plt.show()