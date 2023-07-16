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
### 10M
- **Recall:** 91.039%
- **Search runtime (for 10k queries):** 526.359743s
- **Build time:** 32231.6s == ~9h
- **Datasets used:** pca96 for index building and navigation, clip768 for sequential search
- **Hardware used:**
    - CPU: AMD EPYC 7532
    - 36gb RAM
    - 1 CPU core
- **Time taken:** 9.5h
- **Hyperparameters:**
    - 122 leaf nodes
    - 210 epochs
    - 1 hidden layer with 256 neurons
    - 0.009 learning rate
    - 4 leaf nodes stop condition

### 300K
- **Recall:** 91.081%
- **Search runtime (for 10k queries):** 21.93s
- **Build time:** 786.8s
- **Datasets used:** pca96 for index building and navigation, clip768 for sequential search
- **Hyperparameters:**
    - same as for 10M, not optimized for this subset, 7 leaf nodes stop condition

See also github actions

## How to reproduce

### Installation
see also `.github/workflows/ci.yml`

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
python search/search.py # for 10M index
python search/search.py --size=300K -bp 6 # for 300K index
```

### Evaluation
```bash
python eval/eval.py
python eval/plot.py res.csv
cat res.csv
```
