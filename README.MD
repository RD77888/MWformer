# MWformer

This is the code implementation of “MWformer: A Multi-scale Wavelet-enhanced Transformer for Runoff Forecasting”

## Repository Directory

MWformer/
│  eval.py
│  README.MD
│  run.py

│  LICENSE
├─data
│      data.MD
│
├─data_provider
│  │  data_factory.py
│  │  data_loader.py
│ 
├─exp
│  │  exp.py
│  │  exp_basic.py
│  │  __init__.py
│ 
├─layers
│      Attn.py
│      DualAttn.py
│      DWT.py
│      Patch.py
│      Pred.py
│      RMSnorm.py
│      ROPE.py
│
├─model
│      MWformer.py
│
├─scripts
│      run.sh
│
└─utils
    │  HydrologicalScaler.py
    │  loss.py
    │  metrics.py
    │  print_args.py
    │  timefeatures.py
    │  tools.py

### Repository  Notes

1. eval： Model evaluation, load the trained model weights, load the test data, and calculate the evaluation metrics.
2. run： The main running entry of the project is responsible for parsing command line parameters, initializing the experimental environment, and starting the training process.
3. data_provider：  Data loading enables data reading and provides a ready-to-use dataset for model training/evaluation.
4. exp： Define the general logic of the experiment flow.
5. layers:    This directory houses the core components of the MWformer model.
6. model:    Define the complete structure of the MWformer model.
7. scripts:    Train/test startup scripts.
8. utils:     Provide general auxiliary functions for the project.



## Usage

1. Install Python 3.11. For convenience, execute the following command.

   ```
   pip install -r requirements.txt
   ```

2. Prepare Data. place your data in the folder`./data`.

3. Train and evaluate model. The output is printed to the terminal and logged locally.

   ```
   bash ./scripts/run.sh
   ```

## Acknowledgments

Our code is developed based on [[Time-Series-Library](https://github.com/thuml/Time-Series-Library)].

