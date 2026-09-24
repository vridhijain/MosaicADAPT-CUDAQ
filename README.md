# MosaicADAPT-CUDAQ

A CUDA-Q implementation of MosaicADAPT-QAOA with analytic adaptive gradients, KaMIS-based operator selection, and BFGS optimization.

This repository reproduces the main algorithmic workflow of the reference Julia implementation:

https://github.com/pratimugale/MosaicADAPT-QAOA

## Features

- CUDA-Q implementation of arbitrary-depth MosaicADAPT-QAOA
- MaxCut and Max-E3-SAT cost Hamiltonians
- Julia-compatible `qaoa_double_pool`
- analytic TetrisQAOA candidate scoring
- analytic full-ansatz variational gradients
- operator incompatibility-graph construction
- KaMIS maximum-weight independent-set selection
- BFGS optimization with an explicit analytic Jacobian
- reference-validation tests
- first-adaptation and end-to-end scaling benchmarks

## Requirements

The repository has been validated with:

```text
Python 3.12
CUDA-Q 0.15.1
NumPy 2.5.2
SciPy 1.18.1
```

Python 3.12 is recommended.

KaMIS is included as a Git submodule and must be built separately before running KaMIS-based selection or the benchmark scripts.

## Installation

Clone the repository with submodules:

```bash
git clone --recursive https://github.com/vridhijain/MosaicADAPT-CUDAQ.git
cd MosaicADAPT-CUDAQ
```

If the repository was cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

Create a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements-dev.txt
```

Build KaMIS:

```bash
./scripts/build_kamis.sh
```

The build script creates the MMWIS executable at:

```text
external/KaMIS/mmwis/deploy/mmwis
```

On macOS, the build script uses Homebrew GNU GCC. On Linux, it expects `cmake`, `gcc`, `g++`, and `make`.

## Quick Start

Run the validation suite:

```bash
pytest -q
```

Expected result:

```text
22 passed
```

Run a small first-adaptation benchmark:

```bash
python experiments/scaling_first_adaptation.py --sizes 4
```

Run a small end-to-end benchmark:

```bash
python experiments/scaling_full_mosaic.py --sizes 6
```

## Usage

The main entry point is `run_mosaic_adapt_qaoa`.

```python
from hamiltonian import (
    build_maxcut_hamiltonian,
    extract_hamiltonian_terms,
)
from mosaic_adapt import run_mosaic_adapt_qaoa
from operator_pool import build_qaoa_double_pool


n_qubits = 6

edges = [
    (0, 1),
    (0, 4),
    (1, 2),
    (2, 5),
    (3, 5),
]

cost_hamiltonian = build_maxcut_hamiltonian(
    n_qubits,
    edges,
)

cost_coefficients, cost_pauli_words = extract_hamiltonian_terms(
    cost_hamiltonian,
    n_qubits,
)

operator_pool = build_qaoa_double_pool(
    n_qubits
)

result = run_mosaic_adapt_qaoa(
    operator_pool=operator_pool,
    cost_hamiltonian=cost_hamiltonian,
    n_qubits=n_qubits,
    cost_coefficients=cost_coefficients,
    cost_pauli_words=cost_pauli_words,
    gamma0=0.001,
    gradient_threshold=1e-3,
    bfgs_gtol=1e-6,
    selection_backend="kamis",
    kamis_seed=42,
    verbose=True,
)

print("Final energy:", result["final_energy"])
print("Accepted layers:", result["number_of_layers"])
```

## Algorithm

Each MosaicADAPT-QAOA adaptation performs:

```text
current optimized ansatz
        |
        v
provisional cost evolution
        |
        v
analytic candidate gradients
        |
        v
gradient threshold
        |
        v
incompatibility graph
        |
        v
KaMIS MMWIS selection
        |
        v
append selected mixer layer
        |
        v
BFGS re-optimization
        |
        v
repeat
```

Candidate mixers are weighted by the magnitude of their adaptive gradient.

Two candidates are incompatible when their qubit supports overlap. Mosaic selection therefore reduces to a maximum-weight independent-set problem on the operator incompatibility graph.

Candidate eligibility follows:

```text
abs_gradient >= gradient_threshold
```

Newly selected mixer angles are initialized to zero, while the new cost angle is initialized to `gamma0`.

## Analytic Gradients

Production candidate scoring and variational optimization use analytic gradients.

For candidate selection, the current optimized ansatz is followed by a provisional cost layer with angle `gamma0`. Every candidate mixer is temporarily assigned:

```text
beta_j = 0
```

The full variational gradient is then evaluated once. The candidate score is:

```text
|dE / d beta_j| at beta_j = 0
```

Finite differences are not used in the production scoring path. They are retained only in the test suite as an independent numerical validation of the analytic gradient.

## Parameter Ordering

The implementation follows the TetrisQAOA parameter convention:

```text
[
    gamma_1,
    gamma_2,
    ...,
    beta_1,
    beta_2,
    ...
]
```

This ordering is used consistently by parameter packing, analytic-gradient evaluation, and BFGS optimization.

## Operator Pool

The MaxCut reference path implements the Julia `qaoa_double_pool` ordering:

```text
X_0, X_1, ..., X_(n-1)
SUM_X
XX, YY, YZ, ZY for each pair i < j
```

Canonical pool order is restored after thresholding and before construction of the KaMIS graph.

Preserving this ordering is important because graph-node order can affect which independent set is returned when multiple solutions have equal or numerically equivalent weight.

## Optimization

After a new Mosaic layer is selected, the complete accumulated ansatz is reoptimized using:

```python
scipy.optimize.minimize(
    method="BFGS",
    jac=analytic_gradient,
)
```

The default gradient tolerance is:

```text
gtol = 1e-6
```

The reference Julia implementation uses Optim.jl BFGS with an explicitly supplied gradient.

SciPy BFGS and Optim.jl BFGS are different numerical implementations, so exact optimizer trajectories are not expected to match.

Small numerical differences can become significant when a candidate gradient lies extremely close to the hard adaptive threshold. The implementation is therefore validated component by component rather than requiring every later adaptive decision to be identical across the two optimization libraries.

## Validation

The test suite checks:

- MaxCut Hamiltonian behavior
- operator-pool construction and ordering
- CUDA-Q state evolution against an independent NumPy implementation
- analytic full-ansatz variational gradients
- finite-difference validation of the analytic gradient
- reference first-adaptation candidate scores
- gradient-threshold behavior
- KaMIS integer-weight conversion
- KaMIS selection against an exact MWIS oracle
- BFGS optimization of the reference first layer

Run:

```bash
pytest -q
```

## KaMIS

KaMIS is used as the scalable maximum-weight independent-set backend.

Floating-point candidate scores are converted to integer KaMIS weights using the reference scaling procedure:

```text
scaled_weight = score * 1e14
```

If the total scaled weight exceeds:

```text
2e9
```

the weights are proportionally rescaled before integer rounding.

Positive scores that would otherwise round to zero receive integer weight `1`.

The weighted incompatibility graph is then serialized in METIS format and passed to the KaMIS `mmwis` executable.

## Benchmarks

Two scaling benchmarks are included.

### First-Adaptation Scaling

Run:

```bash
python experiments/scaling_first_adaptation.py
```

or specify problem sizes explicitly:

```bash
python experiments/scaling_first_adaptation.py \
    --sizes 4,6,8,10,12,14
```

This experiment isolates one adaptive-selection step:

```text
Hamiltonian construction
operator-pool construction
analytic candidate scoring
gradient threshold
incompatibility graph
KaMIS selection
```

BFGS optimization is intentionally excluded.

Recorded local results:

| Qubits | Pool size | Analytic scoring (s) | KaMIS (s) | Total (s) |
|------:|----------:|---------------------:|----------:|----------:|
| 4  | 29  | 0.203 | 0.357 | 0.563 |
| 6  | 67  | 0.097 | 0.106 | 0.204 |
| 8  | 121 | 0.224 | 0.127 | 0.352 |
| 10 | 191 | 1.089 | 0.157 | 1.249 |
| 12 | 277 | 6.381 | 0.183 | 6.570 |
| 14 | 379 | 37.044 | 0.237 | 37.291 |
| 16 | 497 | 205.763 | 0.519 | 206.301 |
| 18 | 631 | 1133.922 | 0.993 | 1134.948 |

The measurements show that statevector-based analytic candidate scoring becomes the dominant cost as the problem size increases.

At 18 qubits, analytic scoring accounts for approximately 99.9% of the measured first-adaptation runtime.

### End-to-End Scaling

Run:

```bash
python experiments/scaling_full_mosaic.py
```

Recorded local results:

| Qubits | Accepted layers | Attempts | BFGS iterations | Runtime (s) |
|------:|----------------:|---------:|----------------:|------------:|
| 6  | 2 | 3 | 30  | 7.771 |
| 8  | 5 | 6 | 210 | 57.910 |
| 10 | 6 | 7 | 241 | 199.041 |
| 12 | 7 | 8 | 293 | 1408.872 |

The full algorithm compounds the statevector cost through repeated candidate scoring, adaptive iterations, and BFGS evaluations.

The benchmark data are stored in:

```text
results/scaling_first_adaptation.csv
results/scaling_full_mosaic.csv
```

### Hardware Note

The checked-in benchmark results were obtained using the local macOS CPU simulation path.

CUDA-Q does not provide NVIDIA GPU acceleration on macOS, so these measurements should not be interpreted as CUDA-Q GPU scalability limits.

GPU scaling should be evaluated separately on a supported Linux/NVIDIA system.

## Project Structure

```text
.
├── ansatz_executor.py
├── bfgs_optimizer.py
├── hamiltonian.py
├── kamis_mwis.py
├── mosaic_adapt.py
├── mosaic_graph.py
├── operator_pool.py
├── optimization.py
├── tetris_qaoa_scoring.py
├── variational_gradients.py
├── experiments/
│   ├── scaling_first_adaptation.py
│   └── scaling_full_mosaic.py
├── results/
│   ├── scaling_first_adaptation.csv
│   └── scaling_full_mosaic.csv
├── scripts/
│   └── build_kamis.sh
├── tests/
├── external/
│   └── KaMIS/
├── requirements.txt
├── requirements-dev.txt
└── pytest.ini
```

## Reproducibility

A minimal reproduction workflow is:

```bash
git clone --recursive <repository-url>
cd MosaicADAPT-CUDAQ

python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements-dev.txt

./scripts/build_kamis.sh

pytest -q

python experiments/scaling_first_adaptation.py --sizes 4
python experiments/scaling_full_mosaic.py --sizes 6
```

## References

### MosaicADAPT-QAOA

Reference Julia implementation:

https://github.com/pratimugale/MosaicADAPT-QAOA

### KaMIS

KaMIS maximum independent-set solver:

https://github.com/KarlsruheMIS/KaMIS
