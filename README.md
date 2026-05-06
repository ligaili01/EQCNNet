# EQCNNet: A Quantum-Inspired Equivariant Deep Learning Framework for Protein-Ligand Binding Affinity Prediction from 3D Spatial Conformations

## Overview

EQCNNet is an SE(3)-equivariant quantum-inspired convolutional neural network designed for protein-ligand binding affinity prediction. The model integrates:

- **SO(3)-equivariant representations** via Clebsch-Gordan (CG) products and spherical harmonics
- **Gaussian radial basis functions** for distance encoding
- **Quantum-inspired convolutional layers (QCNN)** for enhanced feature extraction
- **Dynamic graph construction** based on atomic coordinates

This repository contains the complete source code and scripts required to reproduce all experimental results reported in the manuscript, including Tables 3–5 and the ablation studies.

---

## Repository Structure

```
EQCNNet/
├── README.md               # This file
├── requirements            # Python dependencies
├── model.py                # Core model implementations (ENN_LBA, ENN_LBA_QCNN, QCNNLayer)
├── train.py                # Main training script
├── evaluate.py             # Evaluation script
├── data.py                 # Data loading and collation functions
├── datasets.py             # Dataset classes and sequence identity splitting
├── utils.py                # Argument parsing and file path utilities
├── setup.py                # Package setup
└── LICENSE                 # License file
```

---

## Requirements

### Hardware
- GPU with CUDA support recommended (tested on NVIDIA A100/V100)
- Minimum 16 GB RAM

### Software Dependencies

Install all dependencies using:

```bash
pip install -r requirements
```

Key dependencies include:

```
torch>=1.9.0
torch-geometric
numpy
scipy
pandas
atom3d
cormorant
scikit-learn
biopython
```

> **Note:** The `cormorant` package is required for CG product computations and spherical harmonics. Install it via:
> ```bash
> pip install git+https://github.com/risilab/cormorant.git
> ```

---

## Dataset Preparation

### 1. Download PDBbind 2019

Download the PDBbind v.2019 refined set from the official website:

```
http://www.pdbbind.org.cn/download.asp
```

Download the following files:
- `PDBbind_v2019_refined.tar.gz` (refined set)
- `PDBbind_v2019_plain_text_index.tar.gz` (index files)

Extract to a local directory, e.g. `./data/pdbbind2019/`.

### 2. Preprocess the Dataset

Run the preprocessing script to convert raw PDB files into the format required by the model:

```bash
python data.py --datadir ./data/pdbbind2019/ --outdir ./data/processed/


### 3. Sequence Identity Splitting

This repository implements two partitioning schemes as described in the manuscript:

**30% sequence identity split** (stringent out-of-distribution evaluation):
```bash
python datasets.py \
    --datadir ./data/pdbbind2019/ \
    --outdir ./data/split_30/ \
    --identity-threshold 0.30


**60% sequence identity split** (moderate homology evaluation):
```bash
python datasets.py \
    --datadir ./data/pdbbind2019/ \
    --outdir ./data/split_60/ \
    --identity-threshold 0.60


The resulting splits will be saved as `train.csv`, `val.csv`, and `test.csv` in the respective output directories.



## Training

### Train EQCNNet (Full Model)

To train the full EQCNNet model under the **60% sequence identity split**:

```bash
python train.py \
    --datadir ./data/split_60/ \
    --prefix EQCNNet_60 \
    --maxl 2 \
    --max-sh 2 \
    --num-cg-levels 4 \
    --num-channels 32 \
    --num-species 5 \
    --batch-size 16 \
    --num-epoch 100 \
    --lr 1e-3 \
    --cutoff-type hard \
    --hard-cut-rad 10.0


To train under the **30% sequence identity split**:

```bash
python train.py \
    --datadir ./data/split_30/ \
    --prefix EQCNNet_30 \
    --maxl 2 \
    --max-sh 2 \
    --num-cg-levels 4 \
    --num-channels 32 \
    --num-species 5 \
    --batch-size 16 \
    --num-epoch 100 \
    --lr 1e-3 \
    --cutoff-type hard \
    --hard-cut-rad 10.0


### Train Ablation Variants

**Model 1** (Conventional features + Gaussian RBF + CG + EQCNN, no SO(3)):
bash


**Model 2** (SO(3) + simplified distance features + CG + EQCNN):
bash
python train.py --datadir ./data/split_60/ --prefix Model2 --simplified-dist


**Model 3** (SO(3) + Gaussian RBF + MLP, no CG coefficients):
bash
python train.py --datadir ./data/split_60/ --prefix Model3 --no-cg


**Model 4** (SO(3) + Gaussian RBF + CG + FC layer, no EQCNN):
bash
python train.py --datadir ./data/split_60/ --prefix Model4 --use-fc


**Model 5** (SO(3) + Gaussian RBF + alternative coupling coefficients):
```bash
python train.py --datadir ./data/split_60/ --prefix Model5 --alt-cg


## Evaluation

To evaluate a trained model on the test set:

```bash
python evaluate.py \
    --datadir ./data/split_60/ \
    --checkpoint ./checkpoints/EQCNNet_60_best.pt \
    --outfile ./results/EQCNNet_60_predictions.txt


The output file will contain per-sample actual and predicted binding affinities (in -log(K) units), along with aggregate metrics:


## Reproducing Reported Results

### Reproduce Table Results (Tables 3–5 in the manuscript)

Run the following script to reproduce all comparison results:

bash
# 60% split (used in ablation study, Table in ablation section)
python train.py --datadir ./data/split_60/ --prefix EQCNNet_60
python evaluate.py --datadir ./data/split_60/ --checkpoint ./checkpoints/EQCNNet_60_best.pt

# 30% split
python train.py --datadir ./data/split_30/ --prefix EQCNNet_30
python evaluate.py --datadir ./data/split_30/ --checkpoint ./checkpoints/EQCNNet_30_best.pt


To reproduce the Wilcoxon signed-rank test p-values reported in the ablation table:

bash
python evaluate.py \
    --datadir ./data/split_60/ \
    --checkpoint ./checkpoints/EQCNNet_60_best.pt \
    --compare-ablations \
    --ablation-checkpoints \
        ./checkpoints/Model1_best.pt \
        ./checkpoints/Model2_best.pt \
        ./checkpoints/Model3_best.pt \
        ./checkpoints/Model4_best.pt \
        ./checkpoints/Model5_best.pt


This will output the two-sided Wilcoxon signed-rank test p-values comparing each ablation variant against EQCNNet.



## External Benchmark: CSAR-HiQ

To evaluate on the CSAR-HiQ external benchmark:

```bash
python evaluate.py \
    --datadir ./data/csar_hiq/ \
    --checkpoint ./checkpoints/EQCNNet_60_best.pt \
    --outfile ./results/EQCNNet_csarhiq.txt


Download CSAR-HiQ from: http://www.csardock.org/



## Model Architecture

The core model (`ENN_LBA_QCNN` in `model.py`) consists of:

1. **Input Layer** (`InputLinear`): Projects atom scalar features
2. **Spherical Harmonics** (`SphericalHarmonicsRel`): Computes SE(3)-equivariant angular features
3. **Radial Filters** (`RadialFilters`): Gaussian radial basis function distance encoding
4. **Equivariant CG Network** (`ENN`): Multi-level Clebsch-Gordan products for equivariant feature learning
5. **Quantum-inspired Convolutional Layer** (`QCNNLayer`): Classical simulation of quantum convolutional operations using parameterized rotation and entangling gate analogues
6. **Output MLP**: Final binding affinity prediction

