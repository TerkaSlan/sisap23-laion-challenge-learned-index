# SISAP 2023 LAION2B Challenge: Learned index
This repository contains our submission into the SISAP 2023 LAION2B Challenge.
The index uses K-Means partitioning from FAISS and incorporates a multi-layer perceptron to guide the search of a given query.


## Submission
in results/

## How to reproduce

### Installation
also, see .github/workflows/ci.yml

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

**Hardware requirements:**
- 32gb RAM
- 1 CPU core
- 4h of runtime (waries depending on the hardware)

### Evaluation
```bash
python3 eval/eval.py
python3 eval/plot.py res.csv
```