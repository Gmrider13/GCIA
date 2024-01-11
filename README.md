# GCIA
Source Code for ICASSP'24 paper: GCIA: A BLACK-BOX GRAPH INJECTION ATTACK METHOD VIA GRAPH CONTRASTIVE LEARNING
1. Dataset:
* Cora, Citeseer and PubMed datasets can be found in torch_geometric.datasets.Planetoid  
* Reddit-12k dataset can be found in [G-NIA]([https://pytorch.org](https://github.com/TaoShuchang/G-NIA)https://github.com/TaoShuchang/G-NIA)

2. The required packages are as follows:
* Python 3.8+
* PyTorch 1.9+
* PyTorch-Geometric 1.7
* DGL 0.7+
* Scikit-learn 0.24+
* Numpy
* tqdm
* NetworkX

3. Running:
* First train gnns, use
''
python data_model_prepare.py
''
* Then perform attack with GCIA, use 
''
pthon GCIA.py  --dataset $dataset_name --model_name $target_model
''
