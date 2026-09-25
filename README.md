# Higher-order Persistence Diagrams

This repository contains the implementation and experiments for the paper **Higher-order Persistence Diagrams**.

Higher-order persistence diagrams recursively pair comparable persistence intervals to represent relationships among intervals that occur within the same observation. Harmonic aggregation (HA) computes coordinates of the resulting higher-order aggregate without explicitly constructing that aggregate. The experiments compare HA with persistence images (PI) and persistence landscapes (PL) on the random network model benchmark and the real-world data benchmark.

## Repository layout

The C++ implementation generates the raw experimental outputs, including persistence diagrams and harmonic aggregation features. The Python implementation performs the statistical analyses and generates processed results and figures.

```text
.
|-- LICENSE.txt
|-- README.md
`-- src/
    |-- run_all.ps1
    |-- cpp/
    |   |-- CMakeLists.txt
    |   |-- core/
    |   |   |-- graph/
    |   |   `-- random_network_models/
    |   |-- methods/
    |   |   `-- TDA/
    |   |-- experiments/
    |   |   |-- random_graph_classification/
    |   |   `-- real_world_data/
    |   `-- results/raw/
    `-- python/
        |-- pyproject.toml
        |-- core/
        |-- methods/
        |   `-- TDA/
        |-- experiments/
        |   |-- random_graph_classification/
        |   `-- real_world_data/
        `-- results/
            |-- figures/
            |-- processed/
            `-- tables/
```

## Requirements

The full experimental pipeline requires PowerShell, CMake with a compatible C++ compiler, Python, and the Python dependencies specified in `src/python/pyproject.toml`.

## Data

The repository does not distribute the datasets used by the real-world data benchmark. Obtain the required datasets independently from the TU dataset collection and place them in the following directories:

```text
src/cpp/experiments/real_world_data/data/
|-- DD/
|-- ENZYMES/
|-- IMDB-BINARY/
|-- IMDB-MULTI/
|-- MUTAG/
|-- NCI1/
|-- PROTEINS/
|-- PTC_MR/
`-- REDDIT-BINARY/
```

Each directory must contain the corresponding TU dataset files.

## Reproducing the experiments

From the repository root, run

```powershell
powershell -ExecutionPolicy Bypass -File .\src\run_all.ps1
```

The script builds both C++ executables, runs the C++ pipelines for the random network model and real-world data experiments, and then runs the corresponding Python analyses.

## Outputs

The C++ experiments write raw results to

```text
src/cpp/results/raw/
```

The Python analyses write processed results and figures to

```text
src/python/results/processed/
src/python/results/figures/
```

## Method

When observations use the same ordered filtration parameter space, their persistence intervals lie in a common space, so relationships among those intervals have consistent meanings across observations. Higher-order persistence diagrams recursively pair persistence intervals to represent these relationships. Aggregation preserves information about which intervals occur together within an observation.

Explicit aggregation constructs the higher-order aggregate. Harmonic aggregation evaluates harmonic aggregation coordinates through preorder-restricted weighted sums and therefore computes these coordinates without explicitly constructing the aggregate.

## Experiments

We compare harmonic aggregation (HA), persistence images (PI), and persistence landscapes (PL) using the same \(H_1\) persistence information within each benchmark.

### Random network model benchmark

We generate 100 batches from each of ten random network models, for 1,000 batches and 4,000 graphs in total. Each batch contains four independent graphs from the same model, and each graph has 100 vertices and 200 edges. We compute an \(H_1\) persistence diagram for each graph.

We evaluate HA, PI, and PL with the same nested cross-validation protocol, which uses 10 outer folds, 5 inner folds, and an SVM with an RBF kernel. The paper reports the following pooled out-of-fold accuracy and macro-\(F_1\), with \(95\%\) bootstrap confidence intervals.

| Representation |            Accuracy |       Macro-\(F_1\) |
| -------------- | ------------------: | ------------------: |
| PL             | \(0.786 \pm 0.025\) | \(0.785 \pm 0.026\) |
| PI             | \(0.940 \pm 0.014\) | \(0.939 \pm 0.015\) |
| HA             | \(0.941 \pm 0.014\) | \(0.941 \pm 0.014\) |

### Additional experiments

* **Ablations.** We evaluate the cross-observation, no-preorder, and linear ablations. The cross-observation ablation pairs persistence diagrams from different graphs in the same batch. The linear ablation uses only the first-order coordinates.
* **Batch-size sensitivity.** We evaluate \(b\in\{1,\ldots,7\}\), with \(b=4\) as the benchmark setting.
* **Character-pair sensitivity.** We evaluate \(K\in\{64,128,256,512,1024\}\) across ten replicates.
* **Robustness.** We evaluate edge deletion, edge insertion, degree-preserving rewiring, and filtration noise.
* **Scalability.** We evaluate the computational scaling of harmonic aggregation.

### Real-world data benchmark

The pipeline processes DD, MUTAG, NCI1, PROTEINS, PTC_MR, ENZYMES, IMDB-BINARY, IMDB-MULTI, and REDDIT-BINARY. The main paper reports DD, MUTAG, NCI1, PROTEINS, and REDDIT-BINARY.

We evaluate bag sizes \(b\in\{1,\ldots,7\}\). At each fixed \(b\), all methods use the same bags and the same \(H_1\) persistence information. PI and PL use the coordinatewise mean of their graph-level representations within each bag. HA averages the first-order coordinates within each bag and evaluates the order-2 coordinates on the mean aggregate. We evaluate all three representations with five outer and five inner cross-validation folds.

## Reproducibility

The implementation fixes the random seeds used by the reported experiments. We ran the reported experiments on Windows 11 Pro with an Intel Core i7-14700KF processor, 32 GB of RAM, Visual Studio 2022 Build Tools, CMake 4.2.1, and Python 3.13.2. We ran all reported experiments on the CPU.

## Paper

This repository accompanies **Higher-order Persistence Diagrams**.

## License

The source code is available under the terms in `LICENSE.txt`.