"""
Analysis Script: Conceptual Qubit Scaling

This script generates and saves a conceptual plot comparing the
data qubit scaling of the proposed patch-based method (O(log RSC))
versus previous dense loading methods (O(log HWC)).

It generates and saves 'output_qubit_scaling_comparison.png'.
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("\n--- Generating conceptual qubit scaling plot... ---")

    # Define a range of image sizes (in total pixels)
    # From 1,000 (e.g., 32x32) to 1,000,000,000 (e.g., 32Kx32K) pixels
    image_sizes = np.logspace(3, 9, 100) 

    # --- Qubits for dense loading: O(log N) = O(log HWC) ---
    qubits_dense = np.log2(image_sizes)

    # --- Qubits for your patch-based method: O(log K) = O(log RSC) ---
    # This is constant and *independent* of the total image size N.
    # Using the parameters from your simulation: R=4, S=4, C=1
    kernel_size = 4 * 4 * 1
    qubits_patch = np.full_like(image_sizes, np.log2(kernel_size))

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(image_sizes, qubits_dense, 
            label=r'Previous Methods (Dense Loading): $O(\log HWC)$', 
            color='r', linewidth=2.5)
            
    ax.plot(image_sizes, qubits_patch, 
            label=r'Proposed Method (Patch-Based): $O(\log RSC)$', 
            color='b', linestyle='--', linewidth=2.5)

    ax.set_xscale('log')
    ax.set_xlabel('Total Input Image Size ($N = H \cdot W \cdot C$)', fontsize=14)
    ax.set_ylabel('Number of Data Qubits', fontsize=14)
    ax.set_title('Conceptual Advantage in Qubit Scaling', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="--")
    plt.tight_layout()

    output_filename_scaling = "output_qubit_scaling_comparison.png"
    plt.savefig(output_filename_scaling, dpi=300, bbox_inches='tight')
    print(f"Conceptual scaling plot saved as '{output_filename_scaling}'")
    plt.show()