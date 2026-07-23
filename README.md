# PatchBased-Quantum-Convolution

This repository contains the official source code for the paper:

> **A Qubit-Efficient Quantum Algorithm for Convolutional Feature Extraction**
>
> Mohammad Rasoul Roshanshah, Payman Kazemikhah, Hossein Aghababa, Masoud Barati
>
> *arXiv (2025)*: **[Link to arXiv paper]**

## 💡 Citation

If you use this code in your academic work, please cite our paper. This is the most direct way to support this research.

We provide a pre-formatted BibTeX entry for your convenience:

```bibtex
@article{Roshanshah2025_QubitEfficient,
  title   = {A Qubit-Efficient Quantum Algorithm for Convolutional Feature Extraction},
  author  = {Roshanshah, Mohammad Rasoul and Kazemikhah, Payman and Aghababa, Hossein and Barati, Masoud},
  journal = {arXiv preprint arXiv:[arXiv_ID]},
  year    = {2025},
  url     = {[Link to arXiv paper]}
}
````

## Overview

This project implements a quantum algorithm for calculating convolutional feature maps. The core of the algorithm relies on a patch-based "sliding window" multiplexer design (`U_X`) and a kernel multiplexer (`U_K`) within a Hadamard test circuit.

This approach demonstrates a significant conceptual advantage in data qubit scaling. Unlike dense-loading methods that require $O(\log HWC)$ qubits for an image of size $H \times W \times C$, our method requires only $O(\log RSC)$ data qubits, where $R \times S \times C$ is the size of the kernel.

The repository covers the full validation pipeline: an end-to-end circuit-level demonstration on a small image, a Monte Carlo study of the measurement error, the conceptual qubit-scaling comparison, and a statistical emulation that extends the validation to a standard benchmark image.

## Features

  * **`quantum_convolution_lib.py`**: A core library with all necessary functions for building oracles and circuits.
  * **`demo_single_run.py`**: A demonstration script that compares a classical convolution with the quantum-reconstructed output.
  * **`analysis_mse_vs_shots.py`**: A Monte Carlo simulation script that analyzes the Mean Squared Error (MSE) vs. simulator shots.
  * **`analysis_qubit_scaling.py`**: A script to generate the conceptual plot comparing $O(\log HWC)$ vs. $O(\log RSC)$ qubit scaling.
  * **`large_scale_emulation.py`**: A statistical emulation script that validates the algorithm on a standard benchmark image (the classic "Cameraman") at a scale where full circuit simulation is intractable.

## Installation

To run this project, clone the repository and install the required Python packages.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Rasoul-Roshan/PatchBased-Quantum-Convolution.git
    cd PatchBased-Quantum-Convolution
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the different experiments using the main scripts.

### 1\. Single Run Demo

This script runs the full quantum-classical comparison and generates images of the input, outputs, and the circuit diagram.

```bash
python demo_single_run.py
```

### 2\. MSE vs. Shots Analysis

This script runs the Monte Carlo simulation to analyze error convergence.

```bash
python analysis_mse_vs_shots.py
```

### 3\. Qubit Scaling Analysis

This script generates the conceptual scaling graph.

```bash
python analysis_qubit_scaling.py
```

### 4\. Large-Scale Benchmark Emulation

This script validates the algorithm on a 128x128 benchmark image and produces the classical vs. quantum feature maps for several shot counts.

```bash
python large_scale_emulation.py
```

## Generated Outputs

| Script | Output files |
| --- | --- |
| `demo_single_run.py` | `output_input_image.png`, `output_classical_filter_{m}.png`, `output_quantum_filter_{m}.png`, `output_hadamard_circuit.png` |
| `analysis_mse_vs_shots.py` | `output_mse_vs_shots_simulation.png` |
| `analysis_qubit_scaling.py` | `output_qubit_scaling_comparison.png` |
| `large_scale_emulation.py` | `output_benchmark_emulation.pdf` |

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
