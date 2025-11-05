"""
Demo Script: Single Run of Quantum Convolution

This script demonstrates the core functionality of the quantum convolution
algorithm. It performs the following steps:
1.  Defines a small 5x5 custom image.
2.  Sets up the classical and quantum problem parameters.
3.  Builds the oracles and the Hadamard test circuit.
4.  Executes the circuit on a Qiskit simulator.
5.  Reconstructs the quantum output.
6.  Visualizes and compares the classical and quantum results.
7.  Saves a diagram of the quantum circuit.
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
    print("--- Defining Custom Input Image Manually ---")
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
    
    # Simulation Parameters
    shots = 8192 # Use a high shot count for better accuracy
    simulator = Aer.get_backend('qasm_simulator')

    # ================================================
    # %% 2. Classical & Quantum Setup
    # ================================================
    start_time = time.time()
    
    print("--- 1. Setting up Classical Problem ---")
    K, (E, F), Y_classical, patch_norms = create_classical_problem_and_patches(
        custom_image, R, S, M
    )

    # Define qubit requirements
    num_data_qubits = int(np.ceil(np.log2(R * S * C)))
    num_spatial_qubits = int(np.ceil(np.log2(E * F)))
    num_filter_qubits = int(np.ceil(np.log2(M)))

    print("\n--- 2. Building Quantum Oracles ---")
    print("Building the U_K oracle for kernel states...")
    uk_oracle = get_uk_oracle_sliding_window(K, M, num_filter_qubits, num_data_qubits)

    print("Building the U_X oracle for image patch states...")
    ux_oracle = get_ux_oracle_sliding_window(
        custom_image, R, S, E, F, num_spatial_qubits, num_data_qubits
    )

    print("\n--- 3. Building Hadamard Test Circuit ---")
    hadamard_test_qc = create_sliding_window_hadamard_test(
        uk_oracle, ux_oracle, M, E, F
    )
    
    setup_end_time = time.time()
    print(f"Circuit setup complete in {setup_end_time - start_time:.2f}s")

    # ================================================
    # %% 3. Simulation
    # ================================================
    print("\n--- 4. Transpiling and Running Simulation ---")
    transpiled_circuit = transpile(hadamard_test_qc, simulator)
    
    sim_start_time = time.time()
    result = simulator.run(transpiled_circuit, shots=shots, memory=False).result()
    counts = result.get_counts()
    sim_end_time = time.time()
    
    print(f"Simulation finished in {sim_end_time - sim_start_time:.2f}s")
    print(f"Total unique outcomes measured: {len(counts)}")

    # ================================================
    # %% 4. Post-processing and Visualization
    # ================================================
    print("\n--- 5. Reconstructing Output ---")
    Y_reconstructed = reconstruct_from_counts(
        counts, K, patch_norms, M, E, F
    )

    print("\n--- Classical Convolution Result (Filter 0) ---")
    print(np.round(Y_classical[0], 2))
    print("\n--- Quantum Reconstructed Result (Filter 0) ---")
    print(np.round(Y_reconstructed[0], 2))

    print("\n--- 6. Visualizing and Saving Matrices ---")

    # --- Define plot appearance ---
    input_image_size = (5, 5)   # Size for the 5x5 input image
    input_font_size = 18        # Font size for the 5x5 input image
    output_image_size = (2, 2)  # Size for the 2x2 output images
    output_font_size = 18       # Font size for the 2x2 output images

    visualize_and_save_matrix(
        custom_image,
        "Input Image",
        "output_input_image.png",
        fig_width=input_image_size[0],
        fig_height=input_image_size[1],
        font_size=input_font_size
    )

    for m in range(M):
        visualize_and_save_matrix(
            Y_classical[m],
            f"Classical Output - Filter {m}",
            f"output_classical_filter_{m}.png",
            fig_width=output_image_size[0],
            fig_height=output_image_size[1],
            font_size=output_font_size
        )
        visualize_and_save_matrix(
            Y_reconstructed[m],
            f"Quantum Reconstructed Output - Filter {m}",
            f"output_quantum_filter_{m}.png",
            fig_width=output_image_size[0],
            fig_height=output_image_size[1],
            font_size=output_font_size
        )

    # --- 7. Save Circuit Diagram ---
    plot_circuit_diagram(hadamard_test_qc, "output_hadamard_circuit.png")

    print("\n--- Demo script finished. ---")