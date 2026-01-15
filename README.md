# GNN Molecular Graph Classification Challenge

> **Predict BACE-1 enzyme inhibition using Graph Neural Networks**

[![Leaderboard](https://img.shields.io/badge/Leaderboard-View-blue)](leaderboard.md)
[![Dataset](https://img.shields.io/badge/Dataset-OGB_MolBACE-green)](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Evaluation](https://img.shields.io/badge/Evaluation-Automated-orange)](.github/workflows/evaluate.yml)

---

## Overview

Welcome to the **GNN Molecular Graph Classification Challenge** — a Kaggle-style competition designed to benchmark Graph Neural Network architectures on molecular property prediction.

### The Task

Given a molecular graph $G = (V, E)$ where:
- **Nodes** $V$ represent atoms with features $\mathbf{x}_v \in \mathbb{R}^d$ encoding atomic properties
- **Edges** $E$ represent chemical bonds with features encoding bond types

Your goal is to learn a graph-level representation and predict a **binary label** $y \in \{0, 1\}$ indicating whether the molecule is an active inhibitor of BACE-1 (Beta-secretase 1), an enzyme associated with Alzheimer's disease.

### Prize

> **Top performers** will be invited to join a high-level research project aiming for publication at **NeurIPS 2026**.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Evaluation Metrics](#evaluation-metrics)
3. [Getting Started](#getting-started)
4. [Baseline Architectures](#baseline-gnn-architectures)
5. [Advanced Architectures](#advanced-gnn-architectures)
6. [Submission Process](#submission-process)
7. [Evaluation Dimensions](#evaluation-dimensions)
8. [Repository Structure](#repository-structure)
9. [Rules](#rules)
10. [References](#references-and-citations)

---

## Dataset

We use the **OGB MolBACE** dataset from the [Open Graph Benchmark](https://ogb.stanford.edu/):

| Split | Molecules | Description |
|-------|-----------|-------------|
| Train | 1,210 | For training your model |
| Valid | 151 | For local validation and hyperparameter tuning |
| Test | 152 | For final evaluation (**labels hidden**) |

### Molecular Features

Each molecule is represented as a graph with:
- **Node features**: 9-dimensional vectors $\mathbf{x}_v \in \mathbb{R}^9$ encoding:
  - Atomic number (type of atom)
  - Chirality tag
  - Degree, formal charge, number of H atoms
  - Hybridization, aromaticity, and ring membership
- **Edge features**: 3-dimensional vectors encoding bond type, stereochemistry, and conjugation

### Scaffold Split

The dataset uses a **scaffold split** based on molecular substructures, ensuring that:
- Test molecules are **structurally different** from training molecules
- This simulates real-world drug discovery scenarios
- Prevents data leakage from similar molecular scaffolds

### Class Imbalance

The dataset is **imbalanced** with approximately 30% positive class (active inhibitors). This makes the task non-trivial — a naive classifier predicting all zeros would achieve ~70% accuracy but poor F1.

---

## Evaluation Metric

Submissions are evaluated using **Macro F1 Score**, which equally weights performance on both classes:

$$F1_{\text{macro}} = \frac{1}{2}\left(F1_{\text{class}_0} + F1_{\text{class}_1}\right)$$

where for each class $c$:

$$F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

with:

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

**Why Macro F1?**
- Treats both classes equally regardless of sample size
- Penalizes poor performance on the minority class
- More challenging than accuracy for imbalanced datasets
- Standard metric in molecular property prediction benchmarks

### Secondary Metric: Efficiency Score

We also track computational efficiency to encourage practical solutions:

$$\text{Efficiency} = \frac{F_1^2}{\log_{10}(\text{time}_{ms}) \times \log_{10}(\text{params})}$$

where:
- $\text{time}_{ms}$ = average inference time per batch (milliseconds)
- $\text{params}$ = total number of trainable parameters

**Interpretation:**
- Logarithmic scaling ensures 10x speedup always gives the same benefit
- Squaring F1 heavily rewards prediction quality
- Balances accuracy with practical deployment considerations

The leaderboard shows both Macro F1 (primary ranking) and Efficiency (secondary metric).

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/muuki2/gnn-ddi.git
cd gnn-ddi
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r starter_code/requirements.txt
```

### 3. Run the Baseline Models

```bash
cd starter_code

# Run GraphSAGE baseline (default)
python baseline.py

# Run specific model
python baseline.py --model graphsage
python baseline.py --model gcn
python baseline.py --model gin

# Run all baselines for comparison
python baseline.py --all
```

This will:
- Download the OGB MolBACE dataset automatically
- Train the selected GNN model for 50 epochs
- Generate `{model}_submission.csv` in the `submissions/` folder
- Report validation F1 score

### Baseline Performance

| Model | Validation Macro F1 |
|-------|---------------------|
| GCN | 0.6153 |
| GIN | 0.6103 |
| GraphSAGE | 0.5835 |

### 4. Explore the Data

```python
from ogb.graphproppred import PygGraphPropPredDataset

dataset = PygGraphPropPredDataset(name='ogbg-molbace')
split_idx = dataset.get_idx_split()

# Get a sample graph
graph = dataset[0]
print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
print(f"Node features shape: {graph.x.shape}")
print(f"Label: {graph.y.item()}")
```

---

## Submission Process

### Step 1: Generate Predictions

Create a CSV file with predictions for all test molecules:

```csv
id,target
0,1
1,0
6,1
...
```

- `id`: Molecule index from `data/test.csv`
- `target`: Your binary prediction (0 or 1)

### Step 2: Submit via Pull Request

1. **Fork** this repository
2. Add your submission file to `submissions/` folder
   - Name it `your_github_username.csv`
3. Create a **Pull Request** to the main repository

### Automated Evaluation

When you open a Pull Request, the system automatically:

1. **Validates** your submission format
2. **Evaluates** against hidden test labels (stored in a private repository)
3. **Comments** on your PR with your Macro F1 score
4. **Updates** the [leaderboard](leaderboard.md) with your result

The test labels are **never exposed** to participants — they are fetched from a private repository during GitHub Actions execution, ensuring fair evaluation.

### Optional: Efficiency Metadata

To appear on the leaderboard with efficiency metrics, include a metadata file:

**Format:** Create `submissions/your_username_metadata.yaml`:

```yaml
team_name: your_team
model_name: MyGNN
model_architecture:
  type: GCN
  num_layers: 3
  hidden_dim: 64
efficiency_metrics:
  inference_time_ms: 5.2
  total_params: 45000
```

Use `evaluation/speed_benchmark.py` to measure these values:

```python
from evaluation.speed_benchmark import ModelProfiler

profiler = ModelProfiler(model)
metrics = profiler.profile_model(loader, device)
print(f"Inference time: {metrics.mean_inference_time_ms} ms")
print(f"Parameters: {metrics.total_params}")
```

See `schema/submission_metadata.json` for the full schema.

### Submission Format

```
submissions/
├── sample_submission.csv     # Example format (152 predictions)
├── your_username.csv         # Your submission
└── your_username_metadata.yaml  # Optional efficiency metadata
```

---

## Current Leaderboard

| Rank | Participant | Macro-F1 | Efficiency | Params |
|------|-------------|----------|------------|--------|
| 🥇 1 | *Baseline-GCN* | 0.6153 | - | 45K |
| 🥈 2 | *Baseline-GIN* | 0.6103 | - | 52K |
| 🥉 3 | *Baseline-GraphSAGE* | 0.5835 | - | 48K |

[View Full Leaderboard](leaderboard.md)

---

## Baseline GNN Architectures

The competition provides three baseline GNN architectures. Below are their message-passing formulations.

### Graph Convolutional Network (GCN)

GCN (Kipf & Welling, 2017) performs spectral graph convolutions using a first-order approximation:

$$\mathbf{h}_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\hat{d}_v \hat{d}_u}} \mathbf{W}^{(l)} \mathbf{h}_u^{(l)}\right)$$

where $\hat{d}_v = 1 + |\mathcal{N}(v)|$ is the augmented degree and $\mathbf{W}^{(l)}$ is a learnable weight matrix.

### GraphSAGE

GraphSAGE (Hamilton et al., 2017) learns to aggregate neighborhood features:

$$\mathbf{h}_v^{(l+1)} = \sigma\left(\mathbf{W}^{(l)} \cdot \text{CONCAT}\left(\mathbf{h}_v^{(l)}, \text{AGG}\left(\{\mathbf{h}_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)\right)$$

where AGG can be mean, max-pool, or LSTM aggregation. Our baseline uses mean aggregation.

### Graph Isomorphism Network (GIN)

GIN (Xu et al., 2019) achieves maximal expressive power among message-passing GNNs:

$$\mathbf{h}_v^{(l+1)} = \text{MLP}^{(l)}\left((1 + \epsilon^{(l)}) \cdot \mathbf{h}_v^{(l)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l)}\right)$$

where $\epsilon$ is a learnable scalar. GIN is as powerful as the Weisfeiler-Lehman graph isomorphism test.

### Graph-Level Readout

All models use global mean pooling for graph-level prediction:

$$\mathbf{h}_G = \frac{1}{|V|} \sum_{v \in V} \mathbf{h}_v^{(L)}$$

followed by a linear classifier: $\hat{y} = \sigma(\mathbf{w}^\top \mathbf{h}_G + b)$

---

## Advanced GNN Architectures

Beyond the baselines, we provide two advanced architectures with stronger mathematical foundations.

### Directed Message Passing Neural Network (D-MPNN)

D-MPNN (Yang et al., 2019) is an edge-centric GNN designed for molecular graphs that prevents "message backflow" — a key limitation of standard MPNNs.

**Message Passing:**

$$\mathbf{m}_{uv}^{(l+1)} = \sum_{w \in \mathcal{N}(u) \setminus \{v\}} f\left(\mathbf{h}_u^{(l)}, \mathbf{m}_{wu}^{(l)}, \mathbf{e}_{uv}\right)$$

$$\mathbf{h}_v^{(l+1)} = g\left(\mathbf{h}_v^{(l)}, \sum_{u \in \mathcal{N}(v)} \mathbf{m}_{uv}^{(l+1)}\right)$$

**Key Features:**
- Messages flow along directed edges
- Prevents information from immediately flowing back to source
- Edge features are first-class citizens
- Particularly effective for molecular property prediction

**Implementation:** `advanced_baselines/dmpnn.py`

### Spectral GNN with Laplacian Regularization

Our Spectral GNN operates in the graph frequency domain using Chebyshev polynomial approximations.

**Chebyshev Convolution:**

$$\mathbf{x} * g_\theta \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{\mathbf{L}}) \mathbf{x}$$

where:
- $\tilde{\mathbf{L}} = \frac{2}{\lambda_{max}} \mathbf{L} - \mathbf{I}$ is the scaled Laplacian
- $T_k$ are Chebyshev polynomials: $T_0 = 1, T_1 = x, T_k = 2xT_{k-1} - T_{k-2}$
- $\theta_k$ are learnable spectral coefficients

**Laplacian Regularization Loss:**

We minimize the Dirichlet energy to encourage smoothness:

$$\mathcal{L}_{smooth} = \frac{1}{|V|} \mathbf{h}^\top \mathbf{L} \mathbf{h} = \frac{1}{|V|} \sum_{(i,j) \in E} \|\mathbf{h}_i - \mathbf{h}_j\|^2$$

**Laplacian Positional Encodings:**

Optional positional features from Laplacian eigenvectors:

$$\mathbf{L} \mathbf{u}_k = \lambda_k \mathbf{u}_k$$

The first $k$ eigenvectors provide structural position information.

**Implementation:** `advanced_baselines/spectral_gnn.py`

---

## Evaluation Dimensions

We evaluate submissions along multiple dimensions beyond raw accuracy.

### 1. Prediction Quality (Primary)

**Macro F1 Score** is the primary ranking metric (see [Evaluation Metrics](#evaluation-metric)).

### 2. Computational Efficiency

Tracked via the efficiency formula above. We record:
- Inference time (ms per batch)
- Parameter count
- Memory usage
- FLOPs estimate

Use the profiler in `evaluation/speed_benchmark.py` to measure your model.

### 3. Uncertainty Quantification

Good models should know when they don't know. We provide tools to evaluate:

**MC Dropout:** Epistemic uncertainty via multiple forward passes with dropout enabled:

$$\sigma^2_{\text{epistemic}} = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t^2 - \left(\frac{1}{T} \sum_{t=1}^{T} \hat{y}_t\right)^2$$

**Conformal Prediction:** Distribution-free prediction sets with coverage guarantees:

$$C(x) = \{y : s(x, y) \leq \hat{q}\}$$

where $\hat{q}$ is calibrated on a holdout set to achieve $(1-\alpha)$ coverage.

**Temperature Scaling:** Post-hoc calibration via:

$$p(y|x) = \text{softmax}(z/T)$$

with temperature $T$ optimized on validation data.

**Metrics:**
- Expected Calibration Error (ECE)
- Brier Score
- Empirical Coverage at 90%

**Implementation:** `evaluation/uncertainty.py`

### 4. Adversarial Robustness

We evaluate model robustness to graph perturbations:

**Attack Types:**
1. **Random Edge Perturbation:** Add/remove random edges
2. **Gradient-Based Attack:** Remove high-importance edges
3. **Feature Noise:** Gaussian noise on node features
4. **Feature Masking:** Zero out random features

**Metrics:**
- Robust Accuracy under attack
- Attack Success Rate (ASR)

$$\text{ASR} = \frac{|\{x : f(x + \delta) \neq y, f(x) = y\}|}{|\{x : f(x) = y\}|}$$

**Implementation:** `evaluation/adversarial.py`

### 5. Pareto Efficiency

We visualize the accuracy-efficiency trade-off:

A model is **Pareto optimal** if no other model is:
- Better in accuracy AND equally efficient, OR
- Equally accurate AND more efficient, OR
- Better in both

**Hypervolume Indicator:**

$$\text{HV}(S) = \text{volume dominated by Pareto front } S$$

Higher hypervolume indicates better overall performance.

**Visualization:** `visualization/pareto_plot.py`

---

## Tips and Ideas

### Additional GNN Architectures
- **GAT** (Graph Attention Network) — attention-weighted message passing
- **MPNN** (Message Passing Neural Network) — edge-conditioned convolutions
- **AttentiveFP** — designed specifically for molecular property prediction
- **D-MPNN** — see our implementation in `advanced_baselines/dmpnn.py`
- **Spectral GNN** — see our implementation in `advanced_baselines/spectral_gnn.py`
- **Ensemble methods** — combine multiple architectures

### Techniques to Consider
- **Class weighting** — address class imbalance via weighted cross-entropy
- **Focal loss** — down-weight easy examples, focus on hard ones
- **Laplacian regularization** — encourage smooth representations (see Spectral GNN)
- **Data augmentation** — random edge dropping, node feature masking
- **Different pooling** — sum pooling, attention-based pooling, Set2Set
- **Virtual nodes** — add a global node connected to all atoms
- **Positional encodings** — Laplacian eigenvectors, random walk features
- **Learning rate scheduling** — cosine annealing, warm restarts
- **Early stopping** — monitor validation F1 to prevent overfitting

### Evaluation Tools
- **Speed benchmark**: `evaluation/speed_benchmark.py` — profile inference time
- **Uncertainty**: `evaluation/uncertainty.py` — MC Dropout, Conformal Prediction
- **Adversarial**: `evaluation/adversarial.py` — robustness testing
- **Visualization**: `visualization/pareto_plot.py` — Pareto front analysis

### Resources
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [OGB Leaderboard for MolBACE](https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-molbace)
- [Graph Neural Networks: A Review](https://arxiv.org/abs/1901.00596)
- [GraphSAGE Paper](https://arxiv.org/abs/1706.02216)
- [GIN Paper](https://arxiv.org/abs/1810.00826)

---

## Repository Structure

```
gnn-ddi/
├── data/
│   ├── train.csv           # Training molecule indices
│   ├── valid.csv           # Validation molecule indices
│   ├── test.csv            # Test molecule indices (labels hidden)
│   └── ogb/                # OGB dataset (auto-downloaded)
├── submissions/
│   ├── sample_submission.csv
│   ├── gcn_submission.csv
│   ├── gin_submission.csv
│   └── graphsage_submission.csv
├── starter_code/
│   ├── baseline.py         # Baseline models (GraphSAGE, GCN, GIN)
│   └── requirements.txt    # Python dependencies
├── advanced_baselines/
│   ├── dmpnn.py            # Directed Message Passing NN
│   └── spectral_gnn.py     # Spectral GNN with Laplacian regularization
├── evaluation/
│   ├── speed_benchmark.py  # Performance profiling
│   ├── uncertainty.py      # Uncertainty quantification
│   └── adversarial.py      # Adversarial robustness tests
├── visualization/
│   └── pareto_plot.py      # Pareto front visualization
├── schema/
│   └── submission_metadata.json  # Metadata JSON schema
├── scripts/
│   └── generate_labels.py  # Label generation utility
├── docs/
│   └── PRIVATE_REPO_SETUP.md  # Private repo setup guide
├── .github/
│   └── workflows/
│       └── evaluate.yml    # Automated scoring workflow
├── scoring_script.py       # Evaluation script (Macro F1 + Efficiency)
├── update_leaderboard.py   # Leaderboard update utility
├── leaderboard.md          # Current standings
└── README.md
```

### Hidden Infrastructure

Test and validation labels are stored in a **private repository** (`gnn-ddi-private`) and are only accessed during GitHub Actions evaluation. This ensures:
- Participants cannot access ground truth labels
- Fair and tamper-proof evaluation
- Transparent scoring via automated comments

---

## Rules

1. **No external data**: Use only the provided OGB MolBACE dataset
2. **No pre-trained models**: Train from scratch; pre-trained molecular embeddings are not allowed
3. **One submission per PR**: Each pull request should contain exactly one submission file
4. **Best score kept**: Multiple submissions allowed; the leaderboard shows your best score
5. **Code sharing encouraged**: You may share code and ideas, but submit individually
6. **Fair play**: Do not attempt to access test labels or exploit the evaluation system

---

## FAQ

**Q: Can I use libraries other than PyTorch Geometric?**
> Yes. You can use DGL, Spektral, JAX, or any other framework. Ensure your final predictions follow the CSV format.

**Q: How do I test locally before submitting?**
> Use the validation set to evaluate your model locally. Training labels are available via OGB; only test labels are hidden.

**Q: Can I submit multiple times?**
> Yes. The leaderboard keeps your best score. Each submission triggers a fresh evaluation.

**Q: How does the automated scoring work?**
> When you open a PR, GitHub Actions fetches the hidden test labels from a private repository, runs the scoring script, and comments on your PR with the result.

**Q: When does the competition end?**
> This is an ongoing challenge. Top performers will be contacted for the research opportunity.

---

## Acknowledgments

- **Dataset**: [Open Graph Benchmark](https://ogb.stanford.edu/)
- **Original BACE data**: [MoleculeNet](https://moleculenet.org/)

---

## References and Citations

If you use this challenge or the methods implemented here, please cite the following:

### Dataset

**Open Graph Benchmark (OGB)**
```bibtex
@article{hu2020ogb,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={22118--22133},
  year={2020}
}
```

**MoleculeNet**
```bibtex
@article{wu2018moleculenet,
  title={MoleculeNet: A Benchmark for Molecular Machine Learning},
  author={Wu, Zhenqin and Ramsundar, Bharath and Feinberg, Evan N and Gomes, Joseph and Geniesse, Caleb and Pappu, Aneesh S and Leswing, Karl and Pande, Vijay},
  journal={Chemical Science},
  volume={9},
  number={2},
  pages={513--530},
  year={2018},
  publisher={Royal Society of Chemistry}
}
```

### GNN Architectures

**GraphSAGE**
```bibtex
@inproceedings{hamilton2017inductive,
  title={Inductive Representation Learning on Large Graphs},
  author={Hamilton, William L and Ying, Rex and Leskovec, Jure},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}
```

**Graph Convolutional Networks (GCN)**
```bibtex
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  booktitle={International Conference on Learning Representations},
  year={2017}
}
```

**Graph Isomorphism Network (GIN)**
```bibtex
@inproceedings{xu2019powerful,
  title={How Powerful are Graph Neural Networks?},
  author={Xu, Keyulu and Hu, Weihua and Leskovec, Jure and Jegelka, Stefanie},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

**Directed Message Passing Neural Network (D-MPNN)**
```bibtex
@article{yang2019analyzing,
  title={Analyzing Learned Molecular Representations for Property Prediction},
  author={Yang, Kevin and Swanson, Kyle and Jin, Wengong and Coley, Connor and 
          Eiden, Philipp and Gao, Hua and Guzman-Perez, Angel and Hopper, Timothy and 
          Kelley, Brian and Mathea, Miriam and others},
  journal={Journal of Chemical Information and Modeling},
  volume={59},
  number={8},
  pages={3370--3388},
  year={2019},
  publisher={ACS Publications}
}
```

**Spectral Graph Theory**
```bibtex
@book{chung1997spectral,
  title={Spectral Graph Theory},
  author={Chung, Fan RK},
  year={1997},
  publisher={American Mathematical Society}
}
```

**Chebyshev Spectral Convolutions**
```bibtex
@inproceedings{defferrard2016convolutional,
  title={Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering},
  author={Defferrard, Micha{\"e}l and Bresson, Xavier and Vandergheynst, Pierre},
  booktitle={Advances in Neural Information Processing Systems},
  volume={29},
  year={2016}
}
```

**Conformal Prediction**
```bibtex
@article{romano2020classification,
  title={Classification with Valid and Adaptive Coverage},
  author={Romano, Yaniv and Sesia, Matteo and Candes, Emmanuel},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={3581--3591},
  year={2020}
}
```

### Libraries

**PyTorch Geometric**
```bibtex
@inproceedings{fey2019fast,
  title={Fast Graph Representation Learning with PyTorch Geometric},
  author={Fey, Matthias and Lenssen, Jan Eric},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019}
}
```

---

## Credits

### Dataset Creators
- **Jure Leskovec** (Stanford University) — Open Graph Benchmark, GraphSAGE
- **Weihua Hu** (Stanford University) — Open Graph Benchmark
- **Zhenqin Wu** and **Vijay Pande** (Stanford University) — MoleculeNet

### GNN Architecture Authors
- **William L. Hamilton**, **Rex Ying**, **Jure Leskovec** — GraphSAGE
- **Thomas N. Kipf**, **Max Welling** — Graph Convolutional Networks
- **Keyulu Xu**, **Weihua Hu**, **Jure Leskovec**, **Stefanie Jegelka** — Graph Isomorphism Network

### Library Developers
- **Matthias Fey**, **Jan Eric Lenssen** — PyTorch Geometric
- **Deep Graph Library (DGL) Team** — DGL Framework

### Special Thanks
- **[BASIRA Lab](https://basira-lab.com/)** — Research collaboration and support
- **Prof. Islem Rekik** (Imperial College London) — Mentorship and guidance

### Competition Organizer
- **Murat Kolic** — Sarajevo, Bosnia and Herzegovina

---

## Contact

For questions or issues, please open a [GitHub Issue](../../issues).

**Organizer:** Murat Kolic ([@muuki2](https://github.com/muuki2))  
**Location:** Sarajevo, Bosnia and Herzegovina

---

*Good luck. May the best GNN win.*