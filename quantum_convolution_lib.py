"""
Core library for the Quantum Convolution Sliding Window project.

This module provides all necessary functions for:
1.  Classical convolution preprocessing.
2.  Building the U_K (kernel) and U_X (patch) quantum oracles.
3.  Constructing the full Hadamard test circuit.
4.  Utility functions for visualizing and saving results.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import Aer
from typing import Tuple, Dict, Any

# ================================================
# %% Part 1: Classical Preprocessing
# ================================================

def create_classical_problem_and_patches(
    custom_image: np.ndarray,
    R: int,
    S: int,
    M: int
) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray, np.ndarray]:
    """
    Sets up the classical problem: defines kernels, performs classical
    convolution, and pre-calculates image patch norms.

    Args:
        custom_image: The 2D input image.
        R: Kernel height.
        S: Kernel width.
        M: Number of kernels (filters).

    Returns:
        A tuple containing:
        - K (np.ndarray): The kernel stack (R, S, C, M).
        - (E, F) (Tuple[int, int]): Output dimensions.
        - Y_classical (np.ndarray): The result of classical convolution.
        - patch_norms (np.ndarray): A flat array of norms for each image patch.
    """
    H, W = custom_image.shape
    C = 1 # Assuming single channel for this problem

    # E, F are the output feature map dimensions
    E, F = H - R + 1, W - S + 1

    # --- Define Hardcoded 4x4 Kernels ---
    # Kernel 1: Horizontal edge detector
    edge_horizontal_4x4 = np.array([
        [ 1,  1,  1,  1], [ 1,  1,  1,  1], [-1, -1, -1, -1], [-1, -1, -1, -1]
    ]).reshape(R, S, C, 1)
    
    # Kernel 2: Vertical edge detector
    edge_vertical_4x4 = np.array([
        [1, 1, -1, -1], [1, 1, -1, -1], [1, 1, -1, -1], [1, 1, -1, -1]
    ]).reshape(R, S, C, 1)

    # Stack kernels along the 4th dimension (M)
    K = np.concatenate([edge_horizontal_4x4, edge_vertical_4x4], axis=3)
    
    if K.shape[3] != M:
        raise ValueError(f"Kernel definition M={K.shape[3]} does not match parameter M={M}")

    # --- Full Classical Convolution (for comparison) ---
    Y_classical = np.zeros((M, E, F))
    for m in range(M):
        for i_prime in range(E):
            for j_prime in range(F):
                # Extract the (R, S) patch from the image
                image_patch = custom_image[i_prime : i_prime + R, j_prime : j_prime + S]
                # Get the m-th kernel
                kernel = K[:, :, 0, m]
                # Perform element-wise multiplication and sum
                Y_classical[m, i_prime, j_prime] = np.sum(image_patch * kernel)

    # --- Pre-calculate norms of all E*F image patches ---
    patch_norms = np.zeros(E * F)
    for p_spatial in range(E * F):
        i_prime, j_prime = p_spatial // F, p_spatial % F
        patch = custom_image[i_prime : i_prime + R, j_prime : j_prime + S]
        patch_norms[p_spatial] = np.linalg.norm(patch.flatten())

    print("Classical problem and patch norms defined.")
    return K, (E, F), Y_classical, patch_norms

# ================================================
# %% Part 2: Sliding Window Quantum Oracles
# ================================================

def get_uk_oracle_sliding_window(
    K: np.ndarray,
    M: int,
    num_filter_qubits: int,
    num_data_qubits: int
) -> QuantumCircuit:
    """
    Creates the U_K oracle (Quantum Multiplexer) to load kernel states.

    This oracle, controlled by the 'filter' register, prepares the
    corresponding normalized kernel vector in the 'data' register.
    
    |m>|0> -> |m>|k_m>
    """
    filter_ancilla = QuantumRegister(num_filter_qubits, name='filter')
    data_reg = QuantumRegister(num_data_qubits, name='data')
    oracle_qc = QuantumCircuit(filter_ancilla, data_reg, name='U_K_sliding')

    for m in range(M):
        # Flatten the m-th kernel
        kernel_m_flat = K[:, :, :, m].flatten()
        norm_k = np.linalg.norm(kernel_m_flat)

        if norm_k < 1e-9:
            continue # Skip zero-norm kernels

        # Create the state preparation gate for this normalized kernel
        prep_gate = StatePreparation(kernel_m_flat / norm_k, label=f'k_prep_{m}')
        
        # Get the binary control string for 'm'
        ctrl_state_m = format(m, f'0{num_filter_qubits}b')
        
        # Append the controlled gate
        oracle_qc.append(
            prep_gate.control(num_filter_qubits, ctrl_state=ctrl_state_m),
            filter_ancilla[:] + data_reg[:]
        )
    return oracle_qc.to_gate()


def get_ux_oracle_sliding_window(
    image: np.ndarray,
    R: int,
    S: int,
    E: int,
    F: int,
    num_spatial_qubits: int,
    num_data_qubits: int
) -> QuantumCircuit:
    """
    Creates the U_X oracle (Quantum Multiplexer) to load image patch states.

    This oracle, controlled by the 'spatial' register, prepares the
    corresponding normalized image patch vector in the 'data' register.

    |p>|0> -> |p>|x_p>
    """
    spatial_ancilla = QuantumRegister(num_spatial_qubits, name='spatial')
    data_reg = QuantumRegister(num_data_qubits, name='data')
    oracle_qc = QuantumCircuit(spatial_ancilla, data_reg, name='U_X_sliding')

    # Iterate over all E*F spatial patch positions
    for p_spatial in range(E * F):
        # Get (row, col) from the flat spatial index 'p'
        i_prime, j_prime = p_spatial // F, p_spatial % F
        
        # Extract and flatten the patch
        patch = image[i_prime : i_prime + R, j_prime : j_prime + S].flatten()
        norm_patch = np.linalg.norm(patch)

        if norm_patch < 1e-9:
            continue # Skip zero-norm patches

        # Create the state preparation gate for this normalized patch
        prep_gate = StatePreparation(patch / norm_patch, label=f'x_prep_{p_spatial}')
        
        # Get the binary control string for 'p'
        ctrl_state_p = format(p_spatial, f'0{num_spatial_qubits}b')
        
        # Append the controlled gate
        oracle_qc.append(
            prep_gate.control(num_spatial_qubits, ctrl_state=ctrl_state_p),
            spatial_ancilla[:] + data_reg[:]
        )
    return oracle_qc.to_gate()


def create_sliding_window_hadamard_test(
    uk_gate: QuantumCircuit,
    ux_gate: QuantumCircuit,
    M: int,
    E: int,
    F: int
) -> QuantumCircuit:
    """
    Builds the complete Hadamard test circuit.

    This circuit estimates the inner product <k_m | x_p> for all
    m and p simultaneously in superposition.
    """
    # Infer qubit counts from the gates
    num_filter_qubits = int(np.ceil(np.log2(M)))
    num_data_qubits = uk_gate.num_qubits - num_filter_qubits
    num_spatial_qubits = int(np.ceil(np.log2(E * F)))

    # --- Define all quantum and classical registers ---
    ancilla = QuantumRegister(1, name='anc')
    spatial_ancilla = QuantumRegister(num_spatial_qubits, name='spatial')
    filter_ancilla = QuantumRegister(num_filter_qubits, name='filter')
    data_reg = QuantumRegister(num_data_qubits, name='data')

    cr_anc = ClassicalRegister(1, name='c_anc')
    cr_spatial = ClassicalRegister(num_spatial_qubits, name='c_spatial')
    cr_filter = ClassicalRegister(num_filter_qubits, name='c_filter')

    qc = QuantumCircuit(
        ancilla, spatial_ancilla, filter_ancilla, data_reg,
        cr_anc, cr_spatial, cr_filter
    )

    # 1. Put ancilla and control registers in uniform superposition
    qc.h(ancilla)
    qc.h(spatial_ancilla)
    qc.h(filter_ancilla)
    qc.barrier()

    # 2. Apply controlled oracles
    # Apply U_K when ancilla is |0>
    qc.append(
        uk_gate.control(1, ctrl_state='0'),
        [ancilla[0]] + filter_ancilla[:] + data_reg[:]
    )
    # Apply U_X when ancilla is |1>
    qc.append(
        ux_gate.control(1, ctrl_state='1'),
        [ancilla[0]] + spatial_ancilla[:] + data_reg[:]
    )
    qc.barrier()

    # 3. Apply Hadamard to ancilla
    qc.h(ancilla)
    
    # 4. Measure all registers
    qc.measure(ancilla, cr_anc)
    qc.measure(spatial_ancilla, cr_spatial)
    qc.measure(filter_ancilla, cr_filter)
    
    return qc

# ================================================
# %% Part 3: Post-processing and Visualization
# ================================================

def reconstruct_from_counts(
    counts: Dict[str, int],
    K: np.ndarray,
    patch_norms: np.ndarray,
    M: int,
    E: int,
    F: int
) -> np.ndarray:
    """
    Reconstructs the convolution output matrix from raw Qiskit counts.

    Args:
        counts: The dictionary of counts from the Qiskit result.
        K: The kernel stack (used to get kernel norms).
        patch_norms: The pre-calculated flat array of patch norms.
        M, E, F: Dimensions.

    Returns:
        Y_reconstructed (np.ndarray): The (M, E, F) reconstructed output.
    """
    num_spatial_qubits = int(np.ceil(np.log2(E * F)))
    num_filter_qubits = int(np.ceil(np.log2(M)))
    
    Y_reconstructed = np.zeros((M, E, F))
    
    # Pre-calculate kernel norms
    kernel_norms = [np.linalg.norm(K[:, :, :, m].flatten()) for m in range(M)]

    # Iterate over every possible (filter, patch) combination
    for p_spatial in range(E * F):
        for m in range(M):
            norm_k = kernel_norms[m]
            norm_x = patch_norms[p_spatial]

            # Format the binary strings to look up in the counts dict
            # Qiskit counts are formatted as: 'c_filter c_spatial c_anc'
            p_spatial_binary = format(p_spatial, f'0{num_spatial_qubits}b')
            m_binary = format(m, f'0{num_filter_qubits}b')
            
            key_0 = f'{m_binary} {p_spatial_binary} 0'
            key_1 = f'{m_binary} {p_spatial_binary} 1'

            counts_0 = counts.get(key_0, 0)
            counts_1 = counts.get(key_1, 0)
            total_counts = counts_0 + counts_1

            reconstructed_val = 0.0
            if total_counts > 0 and norm_k > 1e-9 and norm_x > 1e-9:
                # P(0) = (counts_0 / total)
                # P(1) = (counts_1 / total)
                # <k_m|x_p> = P(0) - P(1)
                estimated_inner_product = (counts_0 - counts_1) / total_counts
                
                # Y[m, p] = <k_m|x_p> * ||k_m|| * ||x_p||
                reconstructed_val = estimated_inner_product * norm_k * norm_x

            # Map flat spatial index 'p' back to (i, j)
            i_prime, j_prime = p_spatial // F, p_spatial % F
            Y_reconstructed[m, i_prime, j_prime] = reconstructed_val
            
    return Y_reconstructed


def visualize_and_save_matrix(
    matrix: np.ndarray,
    title: str,
    filename: str,
    fig_width: float,
    fig_height: float,
    font_size: int
):
    """
    Displays a matrix as an image and saves it to a file.
    """
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(matrix, cmap='gray', interpolation='nearest')

    # Add the value of each cell as text
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            background_value = matrix[i, j]
            # Normalize for text color decision
            min_val, max_val = np.min(matrix), np.max(matrix)
            if max_val == min_val:
                normalized_value = 0.5
            else:
                normalized_value = (background_value - min_val) / (max_val - min_val)

            text_color = "white" if normalized_value < 0.5 else "black"
            
            ax.text(j, i, f'{matrix[i, j]:.2f}',
                    ha="center", va="center", color=text_color, fontsize=font_size)

    ax.set_title(title, fontsize=font_size * 1.2)
    ax.axis('off')

    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    print(f"Saved image: {filename}")


def plot_circuit_diagram(qc: QuantumCircuit, filename: str):
    """
    Saves a high-quality, black-and-white diagram of a quantum circuit.
    """
    print("\n--- Visualizing Quantum Circuit (Grayscale Palette) ---")
    # Use 'mpl' (matplotlib) for a high-quality image output
    # and 'bw' style for a black and white/grayscale look.
    circuit_diagram = qc.draw(output='mpl', style='bw', fold=-1)

    # Display the diagram
    print("Displaying circuit diagram...")
    circuit_diagram.show()

    # Save the diagram to a file
    circuit_diagram.savefig(filename, dpi=300)
    print(f"Saved circuit diagram: {filename}")