import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from data_loader import data_loader_ml, data_loader_dl
from utils import path_definitions, train_eval_pipeline
from algorithm import algorithm_factory
from algorithm.ml_basic import DepressionDetectionAlgorithm_ML_basic
from algorithm.dl_erm import DepressionDetectionAlgorithm_DL_erm
import argparse

if __name__ == "__main__":

    flag_compute_data = False
    folder = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description='Description of the parameters for model evaluation')
    parser.add_argument('--config_name', default='ml_saeb', type=str,
        help = "The name of model configuration file. Default 'ml_saeb'.")
    parser.add_argument('--pred_target', default='dep_weekly', type=str,
        help = "prediction task. Currently it supports 'dep_endterm' and 'dep_weekly'. Default 'dep_weekly'")
    parser.add_argument('--eval_task', default='allbutone', type=str,
        help = "Evaluation of single or multiple datasets. Currently it supports: \n" + 
        "(1) 'single_within_user' - within user training (past) - testing (future) on a single dataset \n" + 
        "(2) 'allbutone' - leave-one-dataset-out setup \n" + 
        "(3) 'crosscovid' - pre/post COVID setup, only support certain datasets \n" + 
        "(4) 'two_overlap' - train/test on overlapping users between two datasets \n" + 
        "(5) 'single' - cross-validation on a single dataset \n" + 
        "(6) 'two' - all combinations of two-dataset pairs, one as training set and the other as testing set \n" + 
        "(7) 'crossgroup` - cross-institute or cross-year setup \n" + 
        "(8) 'all' - do all evaluation setup from (1) to (7).\n Default 'allbutone'.")
    parser.add_argument('--verbose', default=1, type = int,
        help = "Whether to print intermediate pipeline results. 0: minimal output, 1: normal output, 2: detail output")

    args = parser.parse_args()
    print(args)

    config_name = args.config_name
    pred_target = args.pred_target
    eval_task = args.eval_task
    verbose = int(args.verbose)
    eval_task = eval_task.split("--")

    assert pred_target in global_config["all"]["prediction_tasks"]
    assert set(eval_task).issubset(set(["all", "two", "allbutone", "crossgroup", "crosscovid", "two_overlap", "single", "single_within_user"]))

    ds_keys = global_config["all"]["ds_keys"]
    flag_more_feat_types = global_config["all"]["flag_more_feat_types"]
    ins_list = [i.split("_")[0] for i in ds_keys]
    phase_list = [i.split("_")[1] for i in ds_keys]

    pred_targets = [pred_target]

    if (config_name.startswith("dl_") or "dl_clustering" in config_name or "dl_reordering" in config_name):
        flag_dl = True
        # Do not need to load the whole dataset pickle file
        dataset_dict = data_loader_dl.data_loader_dl_placeholder(pred_targets, ds_keys, verbose=verbose>0)
    else:
        flag_dl = False
        dataset_dict = data_loader_ml.data_loader(
            ds_keys_dict={pt: ds_keys for pt in pred_targets}, verbose=verbose>0, flag_more_feat_types=flag_more_feat_types)

    algorithm = algorithm_factory.load_algorithm(config_name=config_name)

    try:
        dl_strategy = global_config["dl"]["training_params"]["best_epoch_strategy"]
    except:
        dl_strategy = ""


    # Single Dataset Evaluation
    if ("single" in eval_task or "all" in eval_task):
        if not flag_dl:
            ray.init(num_cpus=NJOB, ignore_reinit_error=True)
        print("|"*10, "start --", config_name, "single ds", "|"*10)
        evaluation_single_datasets_results = train_eval_pipeline.single_dataset_driver(
            dataset_dict, pred_targets, ds_keys, algorithm,
            multi_thread_flag=False, cv_evaluation_repeat_num=2, verbose=verbose)
        if not flag_dl:
            ray.shutdown()

        results_folder_single = os.path.join(folder, "..", "evaluation_output", "evaluation_single_dataset", pred_target)
        if issubclass(type(algorithm), DepressionDetectionAlgorithm_DL_erm) and dl_strategy != "direct":
            results_folder_single_file = results_folder_single + f"/{config_name}_{dl_strategy}.pkl"
        else:        
            results_folder_single_file = results_folder_single + f"/{config_name}.pkl"
        Path(os.path.split(results_folder_single_file)[0]).mkdir(exist_ok=True, parents=True)
        with open(results_folder_single_file, "wb") as f:
            pickle.dump(evaluation_single_datasets_results, f)
        print("|"*10, "done --", config_name, "single ds", "|"*10)

    # Single Dataset within User Evaluation
    if ("single_within_user" in eval_task or "all" in eval_task):
        assert len(pred_targets) == 1
        assert pred_targets[0] == "dep_weekly"
        if not flag_dl:
            ray.init(num_cpus=NJOB, ignore_reinit_error=True)
        print("|"*10, "start --", config_name, "single ds within user", "|"*10)
        evaluation_single_datasets_within_user_results = train_eval_pipeline.single_dataset_within_user_driver(
            dataset_dict, pred_targets, ds_keys, algorithm, verbose=verbose)
        if not flag_dl:
            ray.shutdown()

        results_folder_single = os.path.join(folder, "..", "evaluation_output", "evaluation_single_dataset_within_user", pred_target)
        if issubclass(type(algorithm), DepressionDetectionAlgorithm_DL_erm) and dl_strategy != "direct":
            results_folder_single_withinuser_file = results_folder_single + f"/{config_name}_{dl_strategy}.pkl"
        else:        
            results_folder_single_withinuser_file = results_folder_single + f"/{config_name}.pkl"
        Path(os.path.split(results_folder_single_withinuser_file)[0]).mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame(evaluation_single_datasets_within_user_results["results_repo"]["dep_weekly"]).T
        print(df[["test_balanced_acc", "test_roc_auc"]])
        with open(results_folder_single_withinuser_file, "wb") as f:
            pickle.dump(evaluation_single_datasets_within_user_results, f)
        print("|"*10, "done --", config_name, "single ds within user", "|"*10)

    # Train-one-test-one Evaluation
    if ("two" in eval_task or "all" in eval_task and len(ds_keys) > 1):
        if not flag_dl:
            ray.init(num_cpus=NJOB, ignore_reinit_error=True)
        print("|"*10, "start --", config_name, "two ds", "|"*10)
        evaluation_two_datasets_results = train_eval_pipeline.two_datasets_driver(
            dataset_dict, pred_targets, ds_keys, algorithm, verbose=verbose)
        if not flag_dl:
            ray.shutdown()

        results_folder_twods = os.path.join(folder, "..", "evaluation_output", "evaluation_two_datasets", pred_target)
        if issubclass(type(algorithm), DepressionDetectionAlgorithm_DL_erm) and dl_strategy != "direct":
            results_folder_twods_file = results_folder_twods + f"/{config_name}_{dl_strategy}.pkl"
        else:
            results_folder_twods_file = results_folder_twods + f"/{config_name}.pkl"
        Path(os.path.split(results_folder_twods_file)[0]).mkdir(exist_ok=True, parents=True)
        with open(results_folder_twods_file, "wb") as f:
            pickle.dump(evaluation_two_datasets_results, f)
        print("|"*10, "done --", config_name, "two ds", "|"*10)

    # Leave-one-dataset-out Evaluation
    if ("allbutone" in eval_task or "all" in eval_task and len(ds_keys) > 1):
        if not flag_dl:
            ray.init(num_cpus=NJOB, ignore_reinit_error=True)
        print("|"*10, "start --", config_name, "all but one ds", "|"*10)
        evaluation_allbutone_datasets_results = train_eval_pipeline.allbutone_datasets_driver(
            dataset_dict, pred_targets, ds_keys, algorithm, verbose=verbose)
        if not flag_dl:
            ray.shutdown()

        results_folder_allbutoneds = os.path.join(folder, "..", "evaluation_output", "evaluation_allbutone_datasets", pred_target)
        if issubclass(type(algorithm), DepressionDetectionAlgorithm_DL_erm) and dl_strategy != "direct":
            results_folder_allbutoneds_file = results_folder_allbutoneds + f"/{config_name}_{dl_strategy}.pkl"
        else:
            results_folder_allbutoneds_file = results_folder_allbutoneds + f"/{config_name}.pkl"
        Path(os.path.split(results_folder_allbutoneds_file)[0]).mkdir(exist_ok=True, parents=True)
        with open(results_folder_allbutoneds_file, "wb") as f:
            pickle.dump(evaluation_allbutone_datasets_results, f)
        print("|"*10, "done --", config_name, "all but one ds", "|"*10)

    # Cross-insitute or Cross-year Evaluation
    # Only applicable for multi-institute situation
    if ("crossgroup" in eval_task or "all" in eval_task and len(ds_keys) > 1):
        if (len(set(ins_list)) <= 1):
            print("Not applicable as only one institute is involved")
        else:
            if not flag_dl:
                ray.init(num_cpus=NJOB, ignore_reinit_error=True)
            print("|"*10, "start --", config_name, "cross group ds", "|"*10)
            evaluation_crossgroup_datasets_results = train_eval_pipeline.crossgroup_datasets_driver(
                dataset_dict, pred_targets, ds_keys, algorithm, verbose=verbose)
            if not flag_dl:
                ray.shutdown()

            results_folder_crossgroupds = os.path.join(folder, "..", "evaluation_output", "evaluation_crossgroup_datasets", pred_target)
            if issubclass(type(algorithm), DepressionDetectionAlgorithm_DL_erm) and dl_strategy != "direct":
                results_folder_crossgroupds_file = results_folder_crossgroupds + f"/{config_name}_{dl_strategy}.pkl"
            else:        
                results_folder_crossgroupds_file = results_folder_crossgroupds + f"/{config_name}.pkl"
            Path(os.path.split(results_folder_crossgroupds_file)[0]).mkdir(exist_ok=True, parents=True)
            with open(results_folder_crossgroupds_file, "wb") as f:
                pickle.dump(evaluation_crossgroup_datasets_results, f)
        print("|"*10, "done --", config_name, "cross group ds", "|"*10)

    # Pre/Post-COVID Evaluation
    # Only applicable for one institute INS-W
    if ("crosscovid" in eval_task or "all" in eval_task and len(ds_keys) > 1):
        if (len(set(ins_list)) == 1 and ins_list[0] in ["INS-W", "INS-W-sample"]):
            if not flag_dl:
                ray.init(num_cpus=NJOB, ignore_reinit_error=True)
            print("|"*10, "start --", config_name, "cross covid ds", "|"*10)
            evaluation_crossgroup_datasets_results = train_eval_pipeline.crosscovid_datasets_driver(
                dataset_dict, pred_targets, ds_keys, algorithm, verbose=verbose)
            if not flag_dl:
                ray.shutdown()

            results_folder_crossgroupds = os.path.join(folder, "..", "evaluation_output", "evaluation_crosscovid_datasets", pred_target)
            if issubclass(type(algorithm), DepressionDetectionAlgorithm_DL_erm) and dl_strategy != "direct":
                results_folder_crossgroupds_file = results_folder_crossgroupds + f"/{config_name}_{dl_strategy}.pkl"
            else:
                results_folder_crossgroupds_file = results_folder_crossgroupds + f"/{config_name}.pkl"
            Path(os.path.split(results_folder_crossgroupds_file)[0]).mkdir(exist_ok=True, parents=True)
            with open(results_folder_crossgroupds_file, "wb") as f:
                pickle.dump(evaluation_crossgroup_datasets_results, f)
        else:
            print("Not applicable. The current code only support one insitute")
        print("|"*10, "done --", config_name, "cross covid ds", "|"*10)


    # Overlapping Participants Evaluation
    if ("two_overlap" in eval_task or "all" in eval_task and len(ds_keys) > 1):
        if (issubclass(type(algorithm), DepressionDetectionAlgorithm_ML_basic)
            and global_config["ml"]["training_params"].get("save_and_reload", False)):
            raise ValueError("two_overlap cannot turn on save_and_reload for ML algorithms." +\
                "Reusing previous calculation could lead to error")

        if not flag_dl:
            ray.init(num_cpus=NJOB, ignore_reinit_error=True)
        print("|"*10, "start --", config_name, "two_overlap ds", "|"*10)
        evaluation_two_overlap_datasets_results = train_eval_pipeline.two_datasets_overlap_driver(
            dataset_dict, pred_targets, ds_keys, algorithm, verbose=verbose)
        if not flag_dl:
            ray.shutdown()

        results_folder_twodsoverlap = os.path.join(folder, "..", "evaluation_output", "evaluation_two_datasets_overlap", pred_target)
        if issubclass(type(algorithm), DepressionDetectionAlgorithm_DL_erm) and dl_strategy != "direct":
            results_folder_twodsoverlap_file = results_folder_twodsoverlap + f"/{config_name}_{dl_strategy}.pkl"
        else:
            results_folder_twodsoverlap_file = results_folder_twodsoverlap + f"/{config_name}.pkl"
        Path(os.path.split(results_folder_twodsoverlap_file)[0]).mkdir(exist_ok=True, parents=True)
        with open(results_folder_twodsoverlap_file, "wb") as f:
            pickle.dump(evaluation_two_overlap_datasets_results, f)
        print("|"*10, "done --", config_name, "two_overlap ds", "|"*10)
