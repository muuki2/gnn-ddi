# 🧬 GNN Molecular Graph Classification Challenge

> **Predict BACE-1 enzyme inhibition using Graph Neural Networks**

[![Leaderboard](https://img.shields.io/badge/📊-Leaderboard-blue)](leaderboard.md)
[![Dataset](https://img.shields.io/badge/📦-OGB_MolBACE-green)](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

Welcome to the **GNN Molecular Graph Classification Challenge**! This is a mini-Kaggle-style competition where you'll build Graph Neural Networks to predict whether molecules inhibit the BACE-1 enzyme.

### The Task

Given a molecular graph where:
- **Nodes** represent atoms
- **Edges** represent chemical bonds

Your goal is to predict a **binary label** indicating whether the molecule is an active inhibitor of BACE-1 (Beta-secretase 1), an enzyme associated with Alzheimer's disease.

### 🏅 Prize

> **Important:** Top performers will be invited to join a high-level research project aiming to submit to **NeurIPS 2026**!

---

## 📊 Dataset

We use the **OGB MolBACE** dataset from the [Open Graph Benchmark](https://ogb.stanford.edu/):

| Split | Molecules | Description |
|-------|-----------|-------------|
| Train | ~1,210 | For training your model |
| Valid | ~151 | For validation/hyperparameter tuning |
| Test | ~152 | For final evaluation (labels hidden) |

**Features:**
- Node features: 9-dimensional vectors encoding atom properties (atomic number, chirality, etc.)
- Edge features: Bond type information
- Split method: Scaffold split (ensures test molecules are structurally different)

**Class Distribution:** The dataset is **imbalanced** (~30% positive class), making this a challenging task.

---

## 📐 Evaluation Metric

Submissions are evaluated using **Macro F1 Score**:

$$F1_{macro} = \frac{1}{2}(F1_{class0} + F1_{class1})$$

This metric:
- Treats both classes equally
- Penalizes poor performance on the minority class
- Is harder to optimize than accuracy for imbalanced data

---

## 🚀 Getting Started

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

### 3. Run the Baseline

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
- Download the OGB MolBACE dataset
- Train the selected GNN model for 50 epochs
- Generate `{model}_submission.csv` in the `submissions/` folder
- Print validation F1 score (~0.52-0.55 depending on model)

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

## 📝 How to Submit

### Step 1: Generate Predictions

Create a CSV file with your predictions for the test set:

```csv
id,target
0,1
1,0
6,1
...
```

- `id`: Molecule ID (from test.csv)
- `target`: Your prediction (0 or 1)

### Step 2: Submit via Pull Request

1. **Fork** this repository
2. Add your submission file to `submissions/` folder
   - Name it `your_github_username.csv`
3. Create a **Pull Request** to the main repository
4. Our automated system will:
   - Evaluate your submission
   - Comment with your score
   - Update the [leaderboard](leaderboard.md)

### Submission Format Example

```
submissions/
├── sample_submission.csv   # Example format
└── your_username.csv       # Your submission
```

---

## 🏆 Current Leaderboard

| Rank | Participant | Macro-F1 Score |
|------|-------------|----------------|
| 🥇 1 | *Baseline-GCN* | 0.6153 |
| 🥈 2 | *Baseline-GIN* | 0.6103 |
| 🥉 3 | *Baseline-GraphSAGE* | 0.5835 |

👉 [View Full Leaderboard](leaderboard.md)

---

## 💡 Tips & Ideas

### GNN Architectures to Try
- **GCN** (Graph Convolutional Network)
- **GAT** (Graph Attention Network)
- **GIN** (Graph Isomorphism Network) - often strong on molecular tasks
- **MPNN** (Message Passing Neural Network)
- **Ensemble methods**

### Techniques to Consider
- **Class weighting** for imbalanced data
- **Data augmentation** (e.g., random edge dropping)
- **Different pooling methods** (mean, sum, attention-based)
- **Learning rate scheduling**
- **Early stopping** on validation F1
- **Feature engineering** with RDKit descriptors

### Resources
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [OGB Leaderboard for MolBACE](https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-molbace)
- [Graph Neural Networks: A Review](https://arxiv.org/abs/1901.00596)
- [GraphSAGE Paper](https://arxiv.org/abs/1706.02216)

---

## 📁 Repository Structure

```
gnn-ddi/
├── data/
│   ├── train.csv           # Training molecule IDs
│   ├── valid.csv           # Validation molecule IDs  
│   └── test.csv            # Test molecule IDs (no labels)
├── submissions/
│   └── sample_submission.csv   # Example submission format
├── starter_code/
│   ├── baseline.py         # Baseline models (GraphSAGE, GCN, GIN)
│   └── requirements.txt    # Python dependencies
├── .github/
│   └── workflows/
│       └── evaluate.yml    # Automated scoring workflow
├── scoring_script.py       # Evaluation script
├── update_leaderboard.py   # Leaderboard update script
├── leaderboard.md          # Current standings
└── README.md               # This file
```

---

## 📜 Rules

1. **No external data**: Use only the provided dataset
2. **No pre-trained models**: Train from scratch (pre-trained GNN embeddings not allowed)
3. **One submission per PR**: Each pull request should contain exactly one submission file
4. **Best score kept**: Multiple submissions allowed; leaderboard shows your best score
5. **Code sharing**: You may share code/ideas, but each participant must submit individually
6. **Fair play**: Do not attempt to access test labels or exploit the evaluation system

---

## ❓ FAQ

**Q: Can I use libraries other than PyTorch Geometric?**
> Yes! You can use DGL, Spektral, or any other framework. Just make sure your final predictions are in the correct CSV format.

**Q: How do I check my score before submitting?**
> Use the validation set to tune your model. The validation labels are available for local testing.

**Q: Can I submit multiple times?**
> Yes! The leaderboard will keep your best score.

**Q: When does the competition end?**
> This is an ongoing challenge. Top performers will be contacted for the research opportunity.

---

## 🙏 Acknowledgments

- Dataset: [Open Graph Benchmark](https://ogb.stanford.edu/)
- Original BACE data: [MoleculeNet](https://moleculenet.org/)

---

## 📚 References & Citations

If you use this challenge or the methods implemented here, please consider citing the following papers:

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

## 👏 Credits & Acknowledgments

### Dataset Creators
- **Jure Leskovec** (Stanford University) - Open Graph Benchmark, GraphSAGE
- **Weihua Hu** (Stanford University) - Open Graph Benchmark
- **Zhenqin Wu** & **Vijay Pande** (Stanford University) - MoleculeNet

### GNN Architecture Authors
- **William L. Hamilton**, **Rex Ying**, **Jure Leskovec** - GraphSAGE
- **Thomas N. Kipf**, **Max Welling** - Graph Convolutional Networks
- **Keyulu Xu**, **Weihua Hu**, **Jure Leskovec**, **Stefanie Jegelka** - Graph Isomorphism Network

### Library Developers
- **Matthias Fey**, **Jan Eric Lenssen** - PyTorch Geometric
- **Deep Graph Library (DGL) Team** - DGL Framework

### Special Thanks
- **[BASIRA Lab](https://basira-lab.com/)** - For support and research collaboration
- **Prof. Islem Rekik** (Imperial College London) - For mentorship, guidance, and access to resources

### Competition Organizer
- **Murat Kolic** - Sarajevo, Bosnia and Herzegovina 🇧🇦

---

## 📧 Contact

For questions or issues, please open a [GitHub Issue](../../issues) or contact the organizers.

**Organizer:** Murat Kolic ([@muuki2](https://github.com/muuki2))  
**Location:** Sarajevo, Bosnia and Herzegovina

---

**Good luck! May the best GNN win! 🚀🧬**