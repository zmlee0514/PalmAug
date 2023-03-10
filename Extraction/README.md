## Module
1. Training  
2. Recognition  
3. Hyper-paramter search  

## Directory
```
Extraction
│ hyper-parameter search.ipynb
│ hyper-parameter search.md
│ recognition.ipynb
│ recognition.md
│ requirements.txt
│ training.ipynb
│ training.md
│
├─model
│     augmentation-reorder-first-50.db
│     best_params-reorder-first-50.pt
│     best_params-reorder-third.pt
│     PolyU-5s-Reduced_ResNeSt50d_pretrained-128emb-LMCL-Focal+Huber-1l-0.001lr-0.9mm-0l2-20e.pt
│     ...
│
├─package
│     dataset.py
│     initModel.py
│     loss.py
│     optunaPlot.py
│     resnet20.py
│     __init__.py
│    
│
└─raw
      optuna.ipynb
      refactor.ipynb
```

## Install
### Drivers
NVIDIA-SMI 510.85.02  
Driver Version: 510.85.02  
CUDA Version: 11.6  

### Packages
jupyter==1.0.0  
jupyterlab==3.3.2  
torch==1.11.0+cu113  
torchvision==0.12.0  
matplotlib==3.5.1  
numpy==1.21.5  
opencv-python==4.5.4.60  
pandas==1.3.5  
Pillow==8.4.0  
scikit-learn==1.0.2  
timm @ git+https://github.com/rwightman/pytorch-image-models.git@9e12530433f38c536e9a5fdfe5c9f455638a3e8a  
optuna==2.10.1  
umap-learn==0.5.3  

## Model name
The name of a model or training records follow the rules descried below.  
Basic:  
`"{}-{}s-{}-{}emb-{}-{}-{}l-{}lr-{}mm-{}l2".format(dataset_type, shot, model_type, feature_dim, head_type, loss_func, lamb, lr, mm, l2)`

1. dataset_type  
   1. Dataset  
      1. PolyU  
      2. Tongji  
      3. MPD_h  
      4. MPD_m  
   1. Postfix  
      1. rotation: Oversampled dataset  
      2. optuna-third: Using the augmentaion searched by TPE+Threshold  
2. shot: Quantity of the registration samples  
3. model_type  
   1. Structure  
      1. Reduced_ResNeSt50  
      2. Reduced_ResNeSt26  
      3. ResNet18  
      4. ResNet20_basic  
   2. Postfix  
      1. pretrained  
4. feature_dim: Dimension of the output feature vector.  
5. head_type: Cosine margin loss.  
   1. LMCL  
   2. ArcFace  
   3. CurricularFace  
6. loss_func: A pair contains classification loss and L2 distance loss.  
   1. Classification loss  
      1. Cross-entropy  
      2. Focal loss  
   2. L2 distance loss  
      1. Center loss  
      2. Huber loss  
7. lamb: Coefficient for L2 distance loss.  
   1. Focal loss => 1  
   2. Center loss => 0.1  
8. lr: Learning rate.  
9. mm: Momentum of SGDM optimizer that update model.  
10. l2: Weight decay of Adam optimizer, which is to update center vector.  

Postfix:  
1. train: Indicating this file is a training record.  
2. baseline-aug: Using the augmentaion presented in reference paper.  
3. optuna-first: Using the augmentaion searched by TPE.  
4. optuna-third: Using the augmentaion searched by TPE+Threshold.  
5. pretrained, nopretrained: For Chap 5.2.1  
6. reduced(0,1,2): For Chap 5.2.2
7. comparison: For Chap 5.2.3
