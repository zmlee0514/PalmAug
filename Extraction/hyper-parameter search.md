## Hyper-parameter search - Optuna
1. Open hyper-parameter search.ipynb 
2. Execute cells in sequence until "optimization" cell

## Training parameters
Defining at the `objective` function.  
1. bounds: Deciding the upper and lower bounds of each transformation.  
2. validation loss: The combination of classification and L2 distance loss, Ex: Focal+Huber or Softmax+Center.  
3. trial.report: Reporting the validation loss of each epoch in each trial, only report the value produced in the first training.  

## Optuna parameters
`storage` is the saved db path.  

`sampler`: TPE sampler
1.  n_startup_trials: Start to use TPE after this number of trials.  
2.  n_ei_candidates: Sample times to calculate EI.  
3.  gamma: Parameter of the TPE definition for choosing quantity of records to form probabiltiy distribution of L.  

`pruner`: MedianPruner
1. n_startup_trials: Quantity of trials not to prune.  
2. n_warmup_steps: Quantity of epoch not to prune.  

Reference: https://optuna.readthedocs.io/en/stable/reference/index.html