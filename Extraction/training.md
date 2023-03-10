## Training 
1. Open training.ipynb  
2. Executing cells in sequence until "training curve" cell  

> Whether to execute "rotation-based oversampling" cell is up to user. It will expand the training data 4 times.

## Parameters
### Dataset parameter
Change `dataset = dataset_choices[0]` to switch training dataset between `["PolyU","Tongji","MPD_h","MPD_m"]`.  
`val_ratio` is useless, there is no validation code in training process.  
`test_ratio` is to evaluate performance in the source domain. It split the classes not samples, so the test data would need to further divide into registration set and probe set for recognition.  
`shot` is the sample number of registraction set.   
In the training, this three parameters are always the same. For the testing dataset, the `test_ratio` world be set to 0 indicates the registration set and probe set contains all classes.

### Model
There is a "model type" section in the model parameter cell, which contains a few choices like ResNet18, ResNet20, and ResNeSt50.  

### Loss
There is also a "loss functions" section in the model parameter cell contains head and losses. Heas refers to the classifier, whose default is LMCL. Losses compose of classification loss and L2 distance loss, Focal loss and Huber loss is the default.

### Others
`lamb` is the coefficent of L2 distance loss. `Focal loss => 1`, `Center loss => 0.1`  
`lr` is learning rate. While `Center loss = 0.1`, it must not set to larger than `0.001`.  
`mm` is the momentum of `SGDM` optimizer.  
`l2` is the weight decay of optimizers.  

