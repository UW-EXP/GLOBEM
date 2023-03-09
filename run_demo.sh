# python evaluation/model_train_eval.py --config_name=ml_chikersal --pred_target=dep_weekly --eval_task=single
# python evaluation/model_train_eval.py --config_name=ml_xu_interpretable --pred_target=dep_weekly --eval_task=single

python evaluation/model_train_eval.py --config_name=ml_chikersal --pred_target=dep_endterm --eval_task=single
python evaluation/model_train_eval.py --config_name=ml_wang --pred_target=dep_endterm --eval_task=single
python evaluation/model_train_eval.py --config_name=ml_xu_interpretable --pred_target=dep_endterm --eval_task=single
