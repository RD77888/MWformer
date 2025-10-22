# MWformer

This is the code implementation of “MWformer: A Multi-scale Wavelet-enhanced Transformer for Runoff Forecasting”

## Repository Directory

```
MWformer/
├── LICENSE
├── eval.py
├── README.md
├── run.py
├── data/
│   ├── data.md
│   └── data_provider/
│       ├── data_factory.py
│       └── data_loader.py
├── exp/
│   ├── exp.py
│   ├── exp_basic.py
│   
├── layers/
│   ├── Attn.py
│   ├── DualAttn.py
│   ├── DWT.py
│   ├── Patch.py
│   ├── Pred.py
│   ├── RMSnorm.py
│   └── ROPE.py
├── model/
│   └── MWformer.py
├── scripts/
│   └── run.sh
└── utils/
    ├── HydrologicalScaler.py
    ├── loss.py
    ├── metrics.py
    ├── print_args.py
    ├── timefeatures.py
    └── tools.py
```



### Repository  Notes

1. eval： Model evaluation, load the trained model weights, load the test data, and calculate the evaluation metrics.
2. run： The main running entry of the project is responsible for parsing command line parameters, initializing the experimental environment, and starting the training process.
3. data_provider：  Data loading enables data reading and provides a ready-to-use dataset for model training/evaluation.
4. exp： Define the general logic of the experiment flow.
5. layers:    This directory houses the core components of the MWformer model.
6. model:    Define the complete structure of the MWformer model.
7. scripts:    Train and test startup scripts.
8. utils:     Provide general auxiliary functions for the project.



## Usage

1. Install Python 3.11. For convenience, execute the following command.

   ```
   pip install -r requirements.txt
   ```

2. Prepare Data. place your data in the folder`./data`.

3. Model hyperparameters are set according to the data format and prediction requirements.

   ```
   #train and eval scripts
   #!/bin/bash
   set -euo pipefail
   export CUDA_VISIBLE_DEVICES=0
   model_name="model name" # specify your model name here
   task_id="task name "
   data_name="data file" 
   target="forecasting target" 
   wavelet_name="db4" 
   seq_len=96 
   pred_len=24
   patch_num=24 
   num_rain=6
   num_runoff=6
   loss="MTL" 
   CHECKPOINT_PATH="./checkpoints/${model_name}_taskid${task_id}_sl${seq_len}_pl${pred_len}_itr0_${loss}/checkpoint.pth"
   ARGS_JSON_PATH="./args_setting/${model_name}_args.json"
   GPU_ID=0
   OUTPUT_DIR="./eval_results/${loss}"
   PREDICT_SEGMENTS="0:5 6:11 12:17 18:23"
   
   python -u run.py \
     --is_training 1 \
     --task_id $task_id \
     --root_path  data\
     --data_path $data_name\
     --model $model_name \
     --patience 50\
     --train_epochs 200\
     --data new \
     --target $target\
     --seq_len $seq_len \
     --pred_len $pred_len\
     --patch_num $patch_num\
     --wavelet_name $wavelet_name\
     --num_rain $num_rain \
     --num_runoff $num_runoff \
     --des 'Exp' \
     --loss MTL\
     --MTL_loss Smooth_L1 \
     --learning_rate 0.0001 \
     --delta 0.002 \
     --scheduler 'cosine_warm_restarts' \
     --dropout 0.2 \
     --itr 1
   
   python -u eval.py \
           --checkpoint "$CHECKPOINT_PATH" \
           --args_path "$ARGS_JSON_PATH" \
           --gpu "$GPU_ID" \
           --output_dir "$OUTPUT_DIR" \
           --predict_segments "$PREDICT_SEGMENTS" \
           2>&1 | tee "$OUTPUT_DIR/eval_log.txt"
   
   ```

   

4. Train and evaluate model. The output is printed to the terminal and logged locally.

   ```
   bash ./scripts/run.sh
   ```

## Acknowledgments

Our code is developed based on [[Time-Series-Library](https://github.com/thuml/Time-Series-Library)].

