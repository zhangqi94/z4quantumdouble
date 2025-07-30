# Topological Multi-Critical Point Between $\mathbb{Z}_2$ Topologically Ordered Phases

This repository provides the implementation of infinite projected entangled pair states (iPEPS) used in the study of exotic topological phase transitions presented in the paper: "Topological Multi-Critical Point Between $\mathbb{Z}_2$ Topologically Ordered Phases".
The project explores various phase transition between $\mathbb{Z}_2$ toric code and double semion phases, driven by perturbations of a $\mathbb{Z}_4$ quantum double model.

## Getting Started

### Requirements
For optimal performance, we recommend using a machine with GPU acceleration. Running solely on CPU may result in significantly slower computation times.

Please ensure the following packages are installed:
- Python >= 3.12
- [PyTorch](https://pytorch.org/get-started/locally/) (with GPU support)
- [ncon](https://github.com/mhauru/ncon): For tensor network contractions

### Training

**Basic example for optimizing the ipeps:**
To start a basic optimization run with random initial tensors and specified model parameters (`h_x`, `h_z`, `h_w`), use:
```
python3 main.py --h_x 0.01 --h_z 0.01 --h_w 0.01 \
    -D 4 --chi 100 --instate "random" --output_path "data/ipeps_output"
```
- `D`: iPEPS bond dimension
- `chi`: Environment bond dimension for CTMRG
- `instate`: Initial state (“random” or path to pre-optimized tensor)
- `output_path`: Directory for storing output tensors and logs

### Code Structure

- `run/`: Scripts for job submission and hyperparameter configuration.
- `generate_square_lattice_sys_basis.py`: Generates the C4v-symmetric iPEPS tensor basis (only needs to be generated once for each bond dimension `D`).
- `main_optim_rum.py`: Core script for optimizing the iPEPS given a set of model parameters.

### Data
The relevant dataset is available at https://github.com/zhangqi94/z4quantumdouble_data.

## Citation

The corresponding paper can be cited as:
```
@article{zhang2025topological,
}
```

