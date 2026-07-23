"""
Statistical Emulation Script: Large-Scale Quantum Convolution

This script validates the patch-based quantum convolution algorithm on a
standard benchmark image (the classic 'Cameraman'). Because simulating deep
quantum circuits for large images is computationally intractable, this script
leverages the theoretical noise model of the algorithm (the binomial
measurement statistics of the Hadamard test) to emulate the quantum
hardware's output.

The emulation is faithful in the following sense: in the Hadamard test the
ancilla is measured as |0> with probability P(0) = (1 + <k_m|x_p>) / 2, so
drawing 'counts_0' from a Binomial(N_shots, P(0)) distribution reproduces the
same sampling statistics that 'analysis_mse_vs_shots.py' obtains from the full
Qiskit simulation, at a fraction of the cost.

It generates comparative feature maps for the horizontal and vertical edge
filters across several shot counts (N), validating the convergence trend.

It generates and saves 'output_benchmark_emulation.pdf' (a high-quality PDF,
saved without opening an interactive window).
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
from typing import Dict, List, Tuple
import time

# ================================================
# %% Part 1: Benchmark Data and Filters
# ================================================

def load_standard_benchmark_image(size: int = 128) -> np.ndarray:
    """
    Loads the classic 'Cameraman' image, widely used in image processing
    and edge detection literature.

    Args:
        size: The side length (in pixels) of the square output image.

    Returns:
        img_resized (np.ndarray): The (size, size) image, normalized to [0, 1].
    """
    print(f"Loading standard benchmark image (Cameraman) and resizing to {size}x{size}...")

    # Load the standard 512x512 cameraman image
    img = data.camera()

    # Normalize pixel values to [0, 1]
    img = img.astype(float) / 255.0

    # Resize to speed up the emulation while maintaining visual recognizability
    img_resized = resize(img, (size, size), anti_aliasing=True)

    print("Benchmark image loaded.")
    return img_resized


def get_4x4_edge_filters() -> List[np.ndarray]:
    """
    Returns the horizontal and vertical 4x4 edge detection filters.

    These are the same kernels used in 'quantum_convolution_lib.py', kept
    here in plain (R, S) form because this script does not build any circuits.

    Returns:
        filters (List[np.ndarray]): [horizontal_filter, vertical_filter].
    """
    # Filter 0: Horizontal edge detector
    h_filter = np.array([
        [ 1,  1,  1,  1], [ 1,  1,  1,  1], [-1, -1, -1, -1], [-1, -1, -1, -1]
    ])

    # Filter 1: Vertical edge detector
    v_filter = np.array([
        [1, 1, -1, -1], [1, 1, -1, -1], [1, 1, -1, -1], [1, 1, -1, -1]
    ])

    return [h_filter, v_filter]

# ================================================
# %% Part 2: Statistical Emulation of the Hadamard Test
# ================================================

def run_hybrid_convolution(
    image: np.ndarray,
    filters: List[np.ndarray],
    shots_list: List[int]
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Performs both classical convolution and statistically emulated
    quantum convolution for varying shot counts.

    Args:
        image: The 2D input image.
        filters: A list of M kernels, each of shape (R, S).
        shots_list: The shot counts to emulate.

    Returns:
        A tuple containing:
        - Y_class (np.ndarray): The (M, E, F) exact classical convolution.
        - Y_quant (Dict[int, np.ndarray]): Maps each shot count to its
          (M, E, F) emulated quantum output.
    """
    H, W = image.shape
    M = len(filters)
    R, S = filters[0].shape

    # E, F are the output feature map dimensions
    E, F = H - R + 1, W - S + 1

    # Initialize output tensors
    Y_class = np.zeros((M, E, F))
    Y_quant = {shots: np.zeros((M, E, F)) for shots in shots_list}

    # Pre-calculate filter norms
    filter_norms = [np.linalg.norm(f) for f in filters]

    print(f"Processing {H}x{W} image into {E}x{F} output map...")
    print(f"Emulating {M * E * F} inner products for {len(shots_list)} shot counts...")

    progress_step = max(1, E // 10)

    # Slide the window (mimicking the patch-based quantum extraction)
    for i in range(E):
        if i % progress_step == 0:
            print(f"  Row {i}/{E} ({100 * i / E:.0f}%)")

        for j in range(F):
            patch = image[i : i + R, j : j + S]
            norm_x = np.linalg.norm(patch)

            for m in range(M):
                f = filters[m]
                norm_f = filter_norms[m]

                # Handle empty/dark patches to avoid division by zero
                if norm_x < 1e-9 or norm_f < 1e-9:
                    Y_class[m, i, j] = 0.0
                    for shots in shots_list:
                        Y_quant[shots][m, i, j] = 0.0
                    continue

                # --- Classical Computation ---
                true_inner_product = np.sum(patch * f)
                Y_class[m, i, j] = true_inner_product

                # --- Quantum Statistical Emulation ---
                # 1. Normalize the inner product to the range [-1, 1]
                v_norm = true_inner_product / (norm_x * norm_f)
                v_norm = np.clip(v_norm, -1.0, 1.0)

                # 2. Map to Hadamard test measurement probability P(|0>)
                p0 = (1.0 + v_norm) / 2.0

                # 3. Emulate quantum shots using the Binomial distribution
                for shots in shots_list:
                    # Simulate measuring the ancilla qubit 'shots' times
                    counts_0 = np.random.binomial(shots, p0)
                    counts_1 = shots - counts_0

                    # 4. Reconstruct the normalized value from counts
                    #    <k_m|x_p> = P(0) - P(1)
                    v_estimated = (counts_0 - counts_1) / shots

                    # 5. Classically rescale to recover the unnormalized output
                    #    Y[m, p] = <k_m|x_p> * ||k_m|| * ||x_p||
                    Y_quant[shots][m, i, j] = v_estimated * norm_x * norm_f

    print("Emulation complete.")
    return Y_class, Y_quant

# ================================================
# %% Part 3: Post-processing and Visualization
# ================================================

def plot_results(
    image: np.ndarray,
    Y_class: np.ndarray,
    Y_quant: Dict[int, np.ndarray],
    shots_list: List[int],
    filter_names: List[str],
    filename: str
):
    """
    Visualizes the classical and quantum reconstructed feature maps and
    saves the comparison grid as a high-quality PDF.

    Args:
        image: The 2D input image.
        Y_class: The (M, E, F) exact classical convolution.
        Y_quant: Maps each shot count to its (M, E, F) emulated output.
        shots_list: The shot counts that were emulated.
        filter_names: Human-readable label for each filter.
        filename: Destination path for the saved figure.
    """
    M = len(filter_names)
    num_cols = len(shots_list) + 1

    # Set up the figure grid
    fig, axes = plt.subplots(M + 1, num_cols, figsize=(4 * num_cols, 4 * (M + 1)))

    # --- Plot the original benchmark image ---
    ax_img = axes[0, 0]
    ax_img.imshow(image, cmap='gray')
    ax_img.set_title(f"Standard Input Image\n({image.shape[0]}x{image.shape[1]})", fontsize=16)
    ax_img.axis('off')

    # Hide the unused subplots in the first row
    for col in range(1, num_cols):
        axes[0, col].axis('off')

    # --- Plot the filter outputs ---
    for m in range(M):
        # Classical result
        ax_c = axes[m + 1, 0]
        ax_c.imshow(Y_class[m], cmap='gray')
        ax_c.set_title(f"Classical Output\n({filter_names[m]})", fontsize=16)
        ax_c.axis('off')

        # Quantum emulated results
        for idx, shots in enumerate(shots_list):
            ax_q = axes[m + 1, idx + 1]
            y_q = Y_quant[shots][m]
            ax_q.imshow(y_q, cmap='gray')

            # Calculate the local MSE for this specific feature map
            mse = np.mean((Y_class[m] - y_q) ** 2)

            ax_q.set_title(f"Quantum: {shots} Shots\nMSE: {mse:.4f}", fontsize=16)
            ax_q.axis('off')

    plt.tight_layout()

    # Save as a high-quality PDF
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved high-quality plot to '{filename}'")

    # Close the plot to free memory, without showing it interactively
    plt.close(fig)


if __name__ == '__main__':

    # ================================================
    # %% 1. Define Problem
    # ================================================
    print("--- 1. Defining Benchmark Problem ---")

    # Problem Parameters
    image_size = 128
    filter_labels = ["Horizontal Edge", "Vertical Edge"]

    # --- Emulation Parameters ---
    # Three distinct magnitudes of shots, to show the convergence progression
    shots_list = [64, 1024, 16384]

    test_image = load_standard_benchmark_image(size=image_size)
    edge_filters = get_4x4_edge_filters()

    # ================================================
    # %% 2. Run Statistical Emulation
    # ================================================
    print("\n--- 2. Running Statistical Emulation ---")
    total_start_time = time.time()

    Y_classical, Y_quantum = run_hybrid_convolution(
        test_image, edge_filters, shots_list
    )

    print(f"\n--- Emulation finished in {(time.time() - total_start_time) / 60:.2f} minutes ---")

    # ================================================
    # %% 3. Report Reconstruction Error
    # ================================================
    print("\n--- 3. Mean Squared Error per Feature Map ---")
    for m, label in enumerate(filter_labels):
        for shots in shots_list:
            mse = np.mean((Y_classical[m] - Y_quantum[shots][m]) ** 2)
            print(f"  {label:<18} | {shots:>6} shots -> MSE: {mse:.6f}")

    # ================================================
    # %% 4. Plot and Save Results
    # ================================================
    print("\n--- 4. Plotting Classical vs. Emulated Feature Maps ---")

    output_filename_benchmark = "output_benchmark_emulation.pdf"
    plot_results(
        test_image, Y_classical, Y_quantum, shots_list,
        filter_labels, output_filename_benchmark
    )

    print("\n--- Emulation script finished. ---")
