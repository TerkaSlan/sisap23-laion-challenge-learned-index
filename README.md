# SISAP 2023 LAION2B Challenge: Learned index
This repository contains our submission into the SISAP 2023 LAION2B Challenge, team **LMI**.

**Members:**
- [Terézia Slanináková](https://github.com/TerkaSlan), Masaryk University
- Jaroslav Oľha, Masaryk University
- Matej Antol, Masaryk University
- David Procházka, Masaryk University
- [Vlastislav Dohnal](https://github.com/dohnal), Masaryk University

The index uses K-Means partitioning from FAISS to create partitioning that is subsequently learned by a multi-layer perceptron. During the search, navigation is guided using probability distribution of the neural network.


## Results
**10M:**
- Recall: 91.421%
- Search runtime (for 10k queries): 663.86s
- Build time: 20828s
- Datasets used: pca96 for index building and navigation, clip768 for sequential search
- Hardware used:
    - CPU Intel Xeon Gold 6130
    - 42gb RAM
    - 1 CPU core
- Hyperparameters:
    - 120 leaf nodes
    - 200 epochs
    - 1 hidden layer with 512 neurons
    - 0.01 learning rate
    - 4 leaf nodes stop condition

## How to reproduce

### Installation
see also .github/workflows/ci.yml

```bash
conda create -n env python=3.8
conda activate env
conda install matplotlib pandas scikit-learn
pip install h5py flake8 setuptools tqdm faiss-cpu
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Running
```bash
pip install --editable .
python3 search/search.py
```

### Evaluation
```bash
python3 eval/eval.py
python3 eval/plot.py res.csv
```

## Hardware requirements
**10M:**
- 42gb RAM
- 1 CPU core
- ~6h of runtime (waries depending on the hardware)
