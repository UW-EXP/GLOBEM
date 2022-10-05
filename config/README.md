# Introduction of Configuration Module

This is a short introduction of the configuration module. As mentioned in the main [README](../README.md), this module provides the flexibility of controlling different parameters in the other modules.

The platform leverages a [`global_config.yaml`](./global_config.yaml) to set a small number of parameters that can be widely applied to multiple models. In addition, each model has its unique config file to enable custom adjustment.

## Global config

The [`global_config.yaml`](./global_config.yaml) determines the following setups:
- `all` - applying to all models
  - `prediction_tasks`: a list that includes all supported model prediction tasks
  - `ds_keys`: a list of dataset keys involved in the evaluation
  - `flag_more_feat_types`: whether to use additional feature types. Currently only can be `True` when `ds_keys` only contains `INS-W`.
- `ml` - applying to traditional models
  - `save_and_reload`: a flag to indicate whether to save and re-use features repetitively (intermediate files will be saved in `tmp` folder). Default `False`. Be careful when turning this flag on, as it will not update the feature file once it is saved. Set it to `True` only when re-running the exact same algorithm.
- `dl` - applying to deep models
  - `best_epoch_strategy`: a flag to choose the best training epoch as the final prediction model: `direct` or `on_test`.
    When it is set as `direct`, it will use a standard strategy: picking the best training epoch on the validation/training set.
    When it is set as `on_test`, it will use another strategy that involves information leakage. It iterates through all training epochs, and performs the same `direct` strategy at each epoch. Then, the results on the testing set across all epochs are compared to identify the best epoch. The results only indicate whether a model is overfitted, and reflect the theoretical upper bound performance during the training.
  - `skip_training`: similar to `save_and_reload` in `ml`, this is a flag to accelerate the deep model evaluation process. A model's intermediate training epoch results will be saved in `tmp` folder. When this flag is turned on, the model can leverage the saved results to re-identify the best epoch. A typical usage case: (1) set `skip_training` as `False` and `best_epoch_strategy` as `direct` to go through the training. (2) set `skip_training` as `True` and `best_epoch_strategy` as `on_test` to find another epoch without the need to re-train the model.

It is worth noting that `global_config.yaml` will overwrite the individual config files on the same items. This can save the effort of changing individual parameters one by one.

## Model config

Each algorithm can lead to one or more models, and each model is accompanied by one config yaml file with a unique name.

Here is a list of the current supported models:
- Traditional Machine Learning Model
  - [Canzian *et al.*](../algorithm/ml_canzian.py) - [`ml_canzian.yaml`](./ml_canzian.yaml)
  - [Saeb *et al.*](../algorithm/ml_saeb.py) - [`ml_saeb.yaml`](./ml_saeb.yaml)
  - [Farhan *et al.*](../algorithm/ml_farhan.py) - [`ml_farhan.yaml`](./ml_farhan.yaml)
  - [Wahle *et al.*](../algorithm/ml_wahle.py) - [`ml_wahle.yaml`](./ml_wahle.yaml)
  - [Lu *et al.*](../algorithm/ml_lu.py) - [`ml_lu.yaml`](./ml_lu.yaml)
  - [Wang *et al.*](../algorithm/ml_wang.py) - [`ml_wang.yaml`](./ml_wang.yaml)
  - [Xu *et al.* - Interpretable](../algorithm/ml_xu_interpretable.py) - [`ml_xu_interpretable.yaml`](./ml_xu_interpretable.yaml)
  - [Xu *et al.* - Personalized](../algorithm/ml_xu_personalized.py) - [`ml_xu_personalized.yaml`](./ml_xu_personalized.yaml)
  - [Chikersal *et al.*](../algorithm/ml_chikersal.py) - [`ml_chikersal.yaml`](./ml_chikersal.yaml)
- Deep-learning Model
  - ERM
    - [ERM-1D-CNN](../algorithm/dl_erm.py) - [`dl_erm_1dCNN.yaml`](./dl_erm_1dCNN.yaml)
    - [ERM-2D-CNN](../algorithm/dl_erm.py) - [`dl_erm_2dCNN.yaml`](./dl_erm_2dCNN.yaml)
    - [ERM-LSTM](../algorithm/dl_erm.py) - [`dl_erm_LSTM.yaml`](./dl_erm_LSTM.yaml)
    - [ERM-Transformer](../algorithm/dl_erm.py) - [`dl_erm_Transformer.yaml`](./dl_erm_Transformer.yaml)
  - [Mixup](../algorithm/dl_erm.py) - [`dl_erm_mixup.yaml`](./dl_erm_mixup.yaml)
  - DANN
    - [DANN - Dataset as Domain](../algorithm/dl_dann.py) - [`dl_dann_ds_as_domain.yaml`](./dl_dann_ds_as_domain.yaml)
    - [DANN - Person as Domain](../algorithm/dl_dann.py) - [`dl_dann_person_as_domain.yaml`](./dl_dann_person_as_domain.yaml)
  - [IRM](../algorithm/dl_irm.py) - [`dl_irm.yaml`](./dl_irm.yaml)
  - CSD
    - [CSD - Dataset as Domain](../algorithm/dl_csd.py) - [`dl_csd_ds_as_domain.yaml`](./dl_csd_ds_as_domain.yaml)
    - [CSD - Person as Domain](../algorithm/dl_csd.py) - [`dl_csd_person_as_domain.yaml`](./dl_csd_person_as_domain.yaml)
  - MLDG
    - [MLDG - Dataset as Domain](../algorithm/dl_mldg.py) - [`dl_mldg_ds_as_domain.yaml`](./dl_mldg_ds_as_domain.yaml)
    - [MLDG - Person as Domain](../algorithm/dl_mldg.py) - [`dl_mldg_person_as_domain.yaml`](./dl_mldg_person_as_domain.yaml)
  - MASF
    - [MASF - Dataset as Domain](../algorithm/dl_masf.py) - [`dl_masf_ds_as_domain.yaml`](./dl_masf_ds_as_domain.yaml)
    - [MASF - Person as Domain](../algorithm/dl_masf.py) - [`dl_masf_person_as_domain.yaml`](./dl_masf_person_as_domain.yaml)
  - [Siamese](../algorithm/dl_siamese.py) - [`dl_siamese.yaml`](./dl_siamese.yaml)
  - [Clustering](../algorithm/dl_clustering.py) - [`dl_clustering.yaml`](./dl_clustering.yaml)
  - [Reorder](../algorithm/dl_reorder.py) - [`dl_reorder.yaml`](./dl_reorder.yaml)