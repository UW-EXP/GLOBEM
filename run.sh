#!/bin/bash

# environment setup
conda create -n globem python=3.7
conda activate globem
pip install -r requirements.txt

# data preparation
python data/data_prep.py

# training and evaluation of two models from two algorithms

python evaluation/model_train_eval.py \
  --config_name=ml_chikersal \
  --pred_target=dep_weekly \
  --eval_task=single_within_user

python evaluation/model_train_eval.py \
  --config_name=dl_reorder \
  --pred_target=dep_weekly \
  --eval_task=allbutone

# key results of the evaluation
python tmp/example_read_results.py