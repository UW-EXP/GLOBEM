"""
Implementation of prior depression detection algorithm:

Xuhai Xu, Prerna Chikersal, Afsaneh Doryab, Daniella K. Villalba, Janine M. Dutcher, Michael J. Tumminia,
Tim Althoff, Sheldon Cohen, Kasey G. Creswell, J. David Creswell, Jennifer Mankoff, and Anind K. Dey. 2019.
Leveraging Routine Behavior and Contextually-Filtered Features for Depression Detection among College Students.
Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 3, 3 (Sept. 2019), 1–33.
https://doi.org/10.1145/3351274
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from algorithm.ml_basic import DepressionDetectionAlgorithm_ML_basic
from data_loader.data_loader_ml import DatasetDict, DataRepo
from algorithm.base import DepressionDetectionClassifierBase

from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
import psutil
if ("pandarallel" in sys.modules):
    sys.modules.pop("pandarallel")
if ("simplefilter" in sys.modules):
    sys.modules.pop("simplefilter")
warnings.filterwarnings("ignore")
import signal

class DepressionDetectionClassifier_ML_xu_interpretable(DepressionDetectionClassifierBase):
    """Classifier for Xu et al. interpretable work. Train a simple adaboost model with decision tree as the base model. """
    def __init__(self, model_params, selected_features):
        self.model_params = model_params
        self.selected_features = selected_features
        
        self.clf = utils_ml.get_clf("adaboost", self.model_params, direct_param_flag = False)
    def fit(self, X, y):
        assert set(self.selected_features).issubset(set(X.columns))
        set_random_seed(42)
        return self.clf.fit(X[self.selected_features], y)
    def predict(self, X, y=None):
        return self.clf.predict(X[self.selected_features])
    def predict_proba(self, X, y=None):
        return self.clf.predict_proba(X[self.selected_features])

class DepressionDetectionAlgorithm_ML_xu_interpretable(DepressionDetectionAlgorithm_ML_basic):
    """Algirithm for Xu et al. interpretable work, extending the basic traditional ml algorithm """

    def __init__(self, config_dict = None, config_name = "ml_xu_interpretable"):
        super().__init__(config_dict, config_name)

        self.feature_list = [
            f"{f}:{epoch}" for f in self.feature_list_base \
                for epoch in epochs_4
        ]

        self.feature_list_norm = [
                f"{f}_norm:{epoch}" for f in self.feature_list_base \
                    for epoch in epochs_4
            ]

        self.NAFILL = - 10<<10

        self.model_params = self.config["model_params"]

        self.flag_small_data_size = False
        self.thresholds_arm = self.set_arm_threshold(flag_th_memory_safe="normal")

        self.top_k = self.model_params["num_top_rule"]
        self.w1 = self.model_params["metric_w1"]
        self.w2 = self.model_params["metric_w2"]
        self.w3 = self.model_params["metric_w3"]

        self.SYS_MEM_MAX_GB = int(psutil.virtual_memory().total / (1024.0**3))

    def set_arm_threshold(self, flag_th_memory_safe:str = "normal"):
        if (flag_th_memory_safe == "normal"):
            key = "arm_thresholds"
        elif (flag_th_memory_safe == "safe"):
            key = "arm_thresholds_memory_safe"
        elif (flag_th_memory_safe == "safer"):
            key = "arm_thresholds_memory_safer"
        elif (flag_th_memory_safe == "safest"):
            key = "arm_thresholds_memory_safest"
        else:
            key = "arm_thresholds_memory_safer"
        if self.verbose > 0:
            print("set arm theshold: ", key)
        thresholds_arm = {
            "wkdy_morning": {"supp": self.model_params[key]["weekday"]["morning"]["supp"],
                            "conf": self.model_params[key]["weekday"]["morning"]["conf"]},
            "wkdy_afternoon": {"supp": self.model_params[key]["weekday"]["afternoon"]["supp"],
                            "conf": self.model_params[key]["weekday"]["afternoon"]["conf"]},
            "wkdy_evening": {"supp": self.model_params[key]["weekday"]["evening"]["supp"],
                            "conf": self.model_params[key]["weekday"]["evening"]["conf"]},
            "wkdy_night": {"supp": self.model_params[key]["weekday"]["night"]["supp"],
                            "conf": self.model_params[key]["weekday"]["night"]["conf"]},
            "wkend_morning": {"supp": self.model_params[key]["weekend"]["morning"]["supp"],
                            "conf": self.model_params[key]["weekend"]["morning"]["conf"]},
            "wkend_afternoon": {"supp": self.model_params[key]["weekend"]["afternoon"]["supp"],
                            "conf": self.model_params[key]["weekend"]["afternoon"]["conf"]},
            "wkend_evening": {"supp": self.model_params[key]["weekend"]["evening"]["supp"],
                            "conf": self.model_params[key]["weekend"]["evening"]["conf"]},
            "wkend_night": {"supp": self.model_params[key]["weekend"]["night"]["supp"],
                            "conf": self.model_params[key]["weekend"]["night"]["conf"]},
        }
        return thresholds_arm

    def pick_arm_threshold(self, dataset: DatasetDict, pids_all: list):
        # save memory for small data size as there will be too many rules with low thresholds
        if (hasattr(dataset,"eval_task") and dataset.eval_task == "two_overlap"):
            self.thresholds_arm = self.set_arm_threshold(flag_th_memory_safe="safe")
        if (len(pids_all) <= 30 and len(pids_all) > 15):
            self.thresholds_arm = self.set_arm_threshold(flag_th_memory_safe="safe")
        elif (len(pids_all) <= 15):
            self.flag_small_data_size = True
            if (len(dataset.datapoints) < 15):
                self.thresholds_arm = self.set_arm_threshold(flag_th_memory_safe="safest")
            else:
                self.thresholds_arm = self.set_arm_threshold(flag_th_memory_safe="safer")
        else:
            self.flag_small_data_size = False

    ### Step 1: Feature Selection ###
    def feature_selection(self, df_features: pd.DataFrame, df_labels:pd.Series or np.ndarray):
        NAFILL_placeholder = self.NAFILL
        @ray.remote
        def feature_select_mutual_info(df_full, y, feats, slice_key):
            signal.signal(signal.SIGTERM, lambda signalNumber, frame: False)
            set_random_seed(42)
            df = df_full[feats]
            top_features = feats
            num_repeat = 30
            for _ in range(num_repeat):
                ig = dict(zip(df.columns, mutual_info_classif(df.fillna(NAFILL_placeholder), y, discrete_features = False)))
                ig = [(k, ig[k]) for k in sorted(ig, key=ig.get, reverse = True) if ig[k] > 0]
                top_features_ig = [x[0] for x in ig[:int(len(feats) / 2)]]
                top_features_ = list(set(top_features).intersection(set(top_features_ig)))
                if (len(top_features_) >= 10):
                    top_features = top_features_
                else:
                    break
            return slice_key, top_features
        
        df_features_id = ray.put(df_features)
        df_labels_id = ray.put(df_labels)
        pool_results = ray.get([feature_select_mutual_info.remote(
            df_features_id, df_labels_id, feats, slice_key) for slice_key, feats in self.feature_dict.items()]
            )

        top_feature_dict_comp = {r[0]:r[1] for r in pool_results}

        top_feature_dict = {slice_key:[] for slice_key in top_feature_dict_comp}
        top_feature_dict_dis = {slice_key:[] for slice_key in top_feature_dict_comp}
        count_large_featurenum_with_small_datasize = 0
        for slice_key, featcomps in top_feature_dict_comp.items():
            for featcomp in featcomps:
                idx_split = featcomp.rfind(":")
                feat = featcomp[:idx_split] + featcomp[idx_split:].split("_")[0]
                feat_dis = featcomp[:idx_split] + "_dis" + featcomp[idx_split:].split("_")[0]
                top_feature_dict[slice_key].append(feat)
                top_feature_dict_dis[slice_key].append(feat_dis)
            top_feature_dict[slice_key] = list(np.sort(list(set(top_feature_dict[slice_key]))))
            top_feature_dict_dis[slice_key] = list(np.sort(list(set(top_feature_dict_dis[slice_key]))))
            if (self.flag_small_data_size and len(featcomps) > 27):
                count_large_featurenum_with_small_datasize += 1
        if (count_large_featurenum_with_small_datasize > len(top_feature_dict_comp) // 2):
            self.set_arm_threshold(flag_th_memory_safe="safest")
        
        self.top_feature_dict = deepcopy(top_feature_dict)
        self.top_feature_dict_dis = deepcopy(top_feature_dict_dis)
        self.assign_feat_int_dict(self.top_feature_dict, self.top_feature_dict_dis)

        if (self.verbose > 0):
            print("top features selected via mutual info", {k:len(v) for k, v in top_feature_dict.items()})

    def assign_feat_int_dict(self, top_feature_dict, top_feature_dict_dis):
        feat_to_int_dict = {}
        int_to_feat_dict = {}
        featdis_to_int_dict = {}
        int_to_featdis_dict = {}
        count = 1
        for feat, featdis in zip(np.sort(list(set([i for l in top_feature_dict.values() for i in l]))),
                        np.sort(list(set([i for l in top_feature_dict_dis.values() for i in l])))):
            feat_to_int_dict[feat] = {"l": count, "m": count + 1, "h": count + 2}
            feat_to_int_dict[feat + "#l"] = count
            feat_to_int_dict[feat + "#m"] = count + 1
            feat_to_int_dict[feat + "#h"] = count + 2
            int_to_feat_dict[count] = feat + "#l"
            int_to_feat_dict[count + 1] = feat + "#m"
            int_to_feat_dict[count + 2] = feat + "#h"
            
            featdis_to_int_dict[featdis] = {"l": count, "m": count + 1, "h": count + 2}
            featdis_to_int_dict[featdis + "#l"] = count
            featdis_to_int_dict[featdis + "#m"] = count + 1
            featdis_to_int_dict[featdis + "#h"] = count + 2
            int_to_featdis_dict[count] = featdis + "#l"
            int_to_featdis_dict[count + 1] = featdis + "#m"
            int_to_featdis_dict[count + 2] = featdis + "#h"
            count += 3
        self.feat_to_int_dict = deepcopy(feat_to_int_dict)
        self.int_to_feat_dict = deepcopy(int_to_feat_dict)
        self.featdis_to_int_dict = deepcopy(featdis_to_int_dict)
        self.int_to_featdis_dict = deepcopy(int_to_featdis_dict)


    ### Step 2: Assocaition Rule Mining ###
    def arm_behavior_rules(self, df_grp:pd.DataFrame, slice_key:str, min_supp:float=0.5, min_conf:float=0.7):
        def prep_arm(df, top_features):
            # filter very empty rows (i.e., days)
            df_tmp_ = df[["pid", "date"] + top_features]
            df_tmp = df_tmp_[df_tmp_.isna().sum(axis = 1) <= df_tmp_.shape[1]/2].copy()
            # obtain data points per row (i.e., per day per person)
            if (df_tmp.empty):
                if (self.verbose > 0):
                    print("empty")
                df_tmp["dis_value"] = pd.NA
            else:
                df_tmp["dis_value"] = df_tmp.apply(
                    lambda row : [self.featdis_to_int_dict[i][row[i]] for i in top_features if not pd.isna(row[i])],
                    axis = 1).values
            return df_tmp[["pid", "date", "dis_value"]]    

        # prep int list for arm
        data_arm_int = df_grp["X_raw"].apply(lambda x : prep_arm(x, self.top_feature_dict_dis[slice_key]))
        # drop duplicate person day
        data_arm = list(pd.concat(data_arm_int.values, axis = 0).drop_duplicates(["pid", "date"])["dis_value"].values)

        spark = SparkSession.builder.appName("FPGrowthExample")\
            .config("spark.executor.memory", f"{int(self.SYS_MEM_MAX_GB // 3)}G") \
            .config("spark.driver.memory", f"{int(self.SYS_MEM_MAX_GB // 3)}G") \
            .config('spark.driver.maxResultSize', f"{int(self.SYS_MEM_MAX_GB // 5)}G") \
            .getOrCreate()
        df_arm_spark = spark.createDataFrame(
            data=[(idx, arm_int_epoch) for idx, arm_int_epoch in enumerate(data_arm) if len(arm_int_epoch) > 1],
            schema=["id", "items"])

        fpGrowth = FPGrowth(itemsCol="items", minSupport=min_supp, minConfidence=min_conf)
        model_arm = fpGrowth.fit(df_arm_spark)
        df_arm_output = model_arm.associationRules.toPandas()
        df_arm_output["X"] = df_arm_output["antecedent"].apply(lambda x : ";".join(map(str,np.sort(x))))
        df_arm_output["Y"] = df_arm_output["consequent"].apply(lambda x : ";".join(map(str,np.sort(x))))
        df_arm_output["idx"] = np.arange(1, df_arm_output.shape[0]+1)
        spark.stop()
        return df_arm_output[["X","Y","idx","support","confidence","lift"]]
        
    def arm_grp_contrast_slice(self, df_twogrps: pd.DataFrame, slice_key: str):
        df_grp1 = df_twogrps[df_twogrps["y_raw"]]
        df_grp2 = df_twogrps[~df_twogrps["y_raw"]]

        df_arm_grp1 = self.arm_behavior_rules(df_grp1, slice_key,
            min_supp=self.thresholds_arm[slice_key]["supp"],
            min_conf=self.thresholds_arm[slice_key]["conf"])
        df_arm_grp2 = self.arm_behavior_rules(df_grp2, slice_key,
            min_supp=self.thresholds_arm[slice_key]["supp"],
            min_conf=self.thresholds_arm[slice_key]["conf"])

        df_arm_merge = df_arm_grp1.merge(df_arm_grp2,
                        left_on = ["X", "Y"],
                        right_on = ["X", "Y"],
                        how = "outer")
        df_arm_merge.columns = ["X", "Y"] + [f"{j}_{i}" for i in ["grp1", "grp2"] for j in ["idx", "supp", "conf", "lift"]]
        df_arm_merge = df_arm_merge.fillna(0)

        for coef in ["supp", "conf", "lift"]:
            df_arm_merge[f"{coef}_diff"] = df_arm_merge[f"{coef}_grp1"] - df_arm_merge[f"{coef}_grp2"]
        df_arm_merge["X_sym"] = df_arm_merge["X"].apply(lambda x : [self.int_to_feat_dict[int(i)] for i in x.split(";")])
        df_arm_merge["Y_sym"] = df_arm_merge["Y"].apply(lambda x : [self.int_to_feat_dict[int(i)] for i in x.split(";")])
        df_arm_merge["X_sym_dis"] = df_arm_merge["X"].apply(lambda x : [self.int_to_featdis_dict[int(i)] for i in x.split(";")])
        df_arm_merge["Y_sym_dis"] = df_arm_merge["Y"].apply(lambda x : [self.int_to_featdis_dict[int(i)] for i in x.split(";")])
        df_arm_merge["slice"] = slice_key
        
        self.df_arm_merge = deepcopy(df_arm_merge)
        return df_arm_merge
    
    def arm_grp_contrast(self, df_datapoints_wkdy: pd.DataFrame, df_datapoints_wkend: pd.DataFrame):
        df_rule_twogrps = {}
        for slice_key in self.feature_dict:
            if (self.verbose > 0):
                print(slice_key)
            df_rule_twogrps[slice_key] = self.arm_grp_contrast_slice(
                df_datapoints_wkdy if slice_key.startswith("wkdy") else df_datapoints_wkend,
                slice_key)
        if (self.verbose > 0):
            print("Rule mining: ", {k:len(v) for k, v in df_rule_twogrps.items()})
        self.df_rule_twogrps = deepcopy(df_rule_twogrps)
        return df_rule_twogrps

    ### Step 3: Rule Selection ###
    def behavior_rules_selection(self, df_rule_contrast: pd.DataFrame):
        dfs_filtered_asso = {}
        dfs_filtered_asso_straight = {}
        dfs_filtered_asso_nostraight = {}
        for slice_key, df_combined in df_rule_contrast.items():
            th_delta = self.straight_paired(df_combined)[["supp_diff","conf_diff"]].abs().quantile(0.5)
            df_filtered = self.rule_threshold_filter(df_combined,
                                    self.thresholds_arm[slice_key]["supp"] + th_delta["supp_diff"],
                                    self.thresholds_arm[slice_key]["conf"] + th_delta["conf_diff"]
                                    )
            dfs_filtered_asso[slice_key] = deepcopy(df_filtered)
            dfs_filtered_asso_straight[slice_key] = deepcopy(self.straight_paired(df_filtered))
            dfs_filtered_asso_nostraight[slice_key] = deepcopy(self.nostraight_paired(df_filtered))

        rulesets_asso_straight = {}
        for slice_key in dfs_filtered_asso_straight:
            df_filtered = deepcopy(dfs_filtered_asso_straight[slice_key])
            df_filtered["p_y1"] = df_filtered["conf_grp1"] / df_filtered["lift_grp1"]
            df_filtered["p_y2"] = df_filtered["conf_grp2"] / df_filtered["lift_grp2"]
            df_filtered["p_x1"] = df_filtered["supp_grp1"] / df_filtered["conf_grp1"]
            df_filtered["p_x2"] = df_filtered["supp_grp2"] / df_filtered["conf_grp2"]

            X_len = df_filtered["X"].apply(lambda x : x.count(";")+1)
            df_filtered["delta_p_x"] = df_filtered["p_x1"] - df_filtered["p_x2"]
            delta_supp_1 = 2 * df_filtered["supp_diff"] / (df_filtered["p_y1"] + df_filtered["p_y2"])
            delta_supp_2 = (df_filtered["p_y2"] - df_filtered["p_y1"]) * (df_filtered["supp_grp1"] + df_filtered["supp_grp2"]) / (df_filtered["p_y1"] + df_filtered["p_y2"])
            df_filtered["supp_delta"] = delta_supp_1 + delta_supp_2

            df_filtered["new_three_weight"] = np.sign(df_filtered["delta_p_x"]) * np.sign(df_filtered["conf_diff"]) *\
                                            np.power(X_len,self.w1) * \
                                            np.power(np.abs(df_filtered["delta_p_x"]),self.w2) * \
                                            np.power(np.abs(df_filtered["conf_diff"]),self.w3)
            ruleset1 = deepcopy(df_filtered.sort_values(by = ["new_three_weight"], ascending=False).head(self.top_k*5))

            ruleset_asso_straight = pd.concat([ruleset1])
            ruleset_asso_straight_ = self.remove_close_rules(ruleset_asso_straight)
            rulesets_asso_straight[slice_key] = deepcopy(ruleset_asso_straight_.head(self.top_k))
            

        rulesets_asso_nostraight = {}
        for slice_key in dfs_filtered_asso_straight:
            df_filtered = deepcopy(dfs_filtered_asso_nostraight[slice_key])
            df_filtered[["p_x1", "p_x2", "p_y1", "p_y2"]] = 0

            flag_grp1 = df_filtered["idx_grp1"] == 0
            df_filtered.loc[flag_grp1, "p_y1"] = 0
            df_filtered.loc[flag_grp1, "p_y2"] = df_filtered[flag_grp1]["conf_grp2"] / df_filtered[flag_grp1]["lift_grp2"]
            df_filtered.loc[flag_grp1, "p_x1"] = 0
            df_filtered.loc[flag_grp1, "p_x2"] = df_filtered[flag_grp1]["supp_grp2"] / df_filtered[flag_grp1]["conf_grp2"]

            flag_grp2 = df_filtered["idx_grp2"] == 0
            df_filtered.loc[flag_grp2, "p_y1"] = df_filtered[flag_grp2]["conf_grp1"] / df_filtered[flag_grp2]["lift_grp1"]
            df_filtered.loc[flag_grp2, "p_y2"] = 0
            df_filtered.loc[flag_grp2, "p_x1"] = df_filtered[flag_grp2]["supp_grp1"] / df_filtered[flag_grp2]["conf_grp1"]
            df_filtered.loc[flag_grp2, "p_x2"] = 0

            X_len = df_filtered["X"].apply(lambda x : x.count(";")+1)
            df_filtered["delta_p_x"] = df_filtered["p_x1"] - df_filtered["p_x2"]
            delta_supp_1 = 2 * df_filtered["supp_diff"] / (df_filtered["p_y1"] + df_filtered["p_y2"])
            delta_supp_2 = (df_filtered["p_y2"] - df_filtered["p_y1"]) * (df_filtered["supp_grp1"] + df_filtered["supp_grp2"]) / (df_filtered["p_y1"] + df_filtered["p_y2"])
            df_filtered["supp_delta"] = delta_supp_1 + delta_supp_2
            df_filtered["new_three_weight"] = np.sign(df_filtered["delta_p_x"]) * np.sign(df_filtered["conf_diff"]) *\
                                            np.power(X_len,self.w1) * \
                                            np.power(np.abs(df_filtered["delta_p_x"]),self.w2) * \
                                            np.power(np.abs(df_filtered["conf_diff"]),self.w3)
            ruleset1 = deepcopy(df_filtered.sort_values(by = ["new_three_weight"], ascending=False).head(self.top_k*5))
            ruleset_asso_nostraight = pd.concat([ruleset1])
            ruleset_asso_nostraight_ = self.remove_close_rules(ruleset_asso_nostraight)
            rulesets_asso_nostraight[slice_key] = deepcopy(ruleset_asso_nostraight_.head(self.top_k))

        rulesets_final = {}
        for slice_key in rulesets_asso_straight:
            ruleset = pd.concat([rulesets_asso_straight[slice_key], rulesets_asso_nostraight[slice_key]]).reset_index(drop = True)
            ruleset["X"] = ruleset["X"].apply(lambda x : [int(i) for i in x.split(";")])
            ruleset["Y"] = ruleset["Y"].apply(lambda x : [int(i) for i in x.split(";")])
            rulesets_final[slice_key] = ruleset
        self.rulesets_final = deepcopy(rulesets_final)
        if (self.verbose > 0):
            print("Straight rule selection results: ", {slice_key: rulesets.shape[0] for slice_key, rulesets in rulesets_asso_straight.items()})
            print("Non-straight rule selection results: ", {slice_key: rulesets.shape[0] for slice_key, rulesets in rulesets_asso_nostraight.items()})
            print("Final rule selection results: ", {slice_key: rulesets.shape[0] for slice_key, rulesets in rulesets_final.items()})
        return rulesets_final

    ### Step 4: Feature Extraction ###
    def feature_extraction(self, df_datapoints_wkdy: pd.DataFrame, df_datapoints_wkend: pd.DataFrame, rulesets: pd.DataFrame, flag_train:bool):
        @ray.remote
        def extract_feature_arm(df, ruleset, slice_key, int_to_feat_dict, int_to_featdis_dict, flag_train):
            signal.signal(signal.SIGTERM, lambda signalNumber, frame: False)
            def extract_feature_arm_slice(df_data, ruleset, slice_key, int_to_feat_dict, int_to_featdis_dict, flag_train):
                df_arm_features = pd.Series([],dtype='float64')
                for rule_idx, rule in ruleset.iterrows():
                    index_flag = True
                    for x in rule["X"]:
                        dis_label_col = int_to_featdis_dict[x][:-2] 
                        dis_label = int_to_featdis_dict[x][-1]
                        index_flag = index_flag & (df_data[dis_label_col] == dis_label)
                    if (sum(index_flag) == 0 and flag_train): # Testing data does not skip rules to ensure compatibility
                        continue
                    feature_columns = [int_to_feat_dict[y][:-2] for y in rule["Y"]]
                    feature_columns_name = [y + "#" + slice_key + "#rule" + str(rule_idx) for y in feature_columns]

                    df_multimodal = df_data[index_flag][feature_columns]
                    df_multimodal_mean = df_multimodal.mean()
                    df_multimodal_mean.index = ["mean_" + y for y in feature_columns_name]

                    df_multimodal_std = df_multimodal.std()
                    df_multimodal_std.index = ["std_" + y for y in feature_columns_name]

                    df_multimodal_mean_std = pd.concat([df_multimodal_mean,df_multimodal_std])
                    df_arm_features = pd.concat([df_arm_features, df_multimodal_mean_std])
                return df_arm_features
            return df["X_raw"].apply(lambda x : extract_feature_arm_slice(x, ruleset, slice_key,
                int_to_feat_dict, int_to_featdis_dict, flag_train))
        
        df_datapoints_wkdy_id = ray.put(df_datapoints_wkdy)
        df_datapoints_wkend_id = ray.put(df_datapoints_wkend)
        int_to_feat_dict_id = ray.put(deepcopy(self.int_to_feat_dict))
        int_to_featdis_dict_id = ray.put(deepcopy(self.int_to_featdis_dict))

        results_pool = ray.get([
            extract_feature_arm.remote(
                df_datapoints_wkdy_id if slice_key.startswith("wkdy_") else df_datapoints_wkend_id,
                rulesets[slice_key], slice_key, int_to_feat_dict_id, int_to_featdis_dict_id, flag_train)\
            for slice_key in rulesets])
        return results_pool

    def prep_data_repo(self, dataset:DatasetDict, flag_train:bool = True) -> DataRepo:
        set_random_seed(42)
        df_datapoints = deepcopy(dataset.datapoints)

        pids_all = df_datapoints["pid"].unique()
        pids_arm = np.random.choice(pids_all, int(0.35 * len(pids_all)), replace = False)
        pids_traintest = [i for i in pids_all if i not in pids_arm]

        self.pick_arm_threshold(dataset, pids_all)

        idx_flag = df_datapoints["pid"].apply(lambda x :x in pids_arm)
        df_datapoints_arm = df_datapoints[idx_flag]
        df_datapoints_traintest = df_datapoints[~idx_flag]

        df_datapoints_arm_wkdy = deepcopy(df_datapoints_arm)
        df_datapoints_arm_wkdy["X_raw"] = df_datapoints_arm["X_raw"].apply(lambda x : self.get_wks(x, "wkdy"))
        df_datapoints_arm_wkend = deepcopy(df_datapoints_arm)
        df_datapoints_arm_wkend["X_raw"] = df_datapoints_arm["X_raw"].apply(lambda x : self.get_wks(x, "wkend"))

        df_datapoints_traintest_wkdy = deepcopy(df_datapoints_traintest)
        df_datapoints_traintest_wkdy["X_raw"] = df_datapoints_traintest["X_raw"].apply(lambda x : self.get_wks(x, "wkdy"))
        df_datapoints_traintest_wkend = deepcopy(df_datapoints_traintest)
        df_datapoints_traintest_wkend["X_raw"] = df_datapoints_traintest["X_raw"].apply(lambda x : self.get_wks(x, "wkend"))

        df_epoch_features_wkdy = df_datapoints_arm_wkdy["X_raw"].apply(lambda x : self.get_epoch_features(x, "wkdy"))
        df_epoch_features_wkend = df_datapoints_arm_wkend["X_raw"].apply(lambda x : self.get_epoch_features(x, "wkend"))
        df_epoch_features = pd.concat([df_epoch_features_wkdy, df_epoch_features_wkend], axis = 1)

        shape1 = df_epoch_features.shape
        if (flag_train): # Testing dataset does not delete features to ensure compatibility
            df_epoch_features = df_epoch_features[df_epoch_features.columns[(df_epoch_features.isna().sum(axis = 0) < df_epoch_features.shape[0]/1.1)]] # filter very empty features
        shape2 = df_epoch_features.shape
        del_cols = shape1[1] - shape2[1]
        if (self.verbose > 0):
            print("Delete rows for initial feature extraction:", del_cols)

        self.feature_dict = {}
        for feat in df_epoch_features.columns:
            slice_str = feat.split(":")[-1]
            epoch, wk, comp = slice_str.split("_")
            slice_key = f"{wk}_{epoch}"
            if (slice_key) in self.feature_dict:
                self.feature_dict[slice_key] += [feat]
            else:
                self.feature_dict[slice_key] = [feat]

        if (flag_train):
            
            Path(self.results_save_folder).mkdir(parents=True, exist_ok=True)
            self.save_file_path = os.path.join(self.results_save_folder, dataset.key + "--" + dataset.prediction_target + ".pkl")

            if (self.config["training_params"]["save_and_reload"] and os.path.exists(self.save_file_path)):
                with open(self.save_file_path, "rb") as f:
                    data_repo = pickle.load(f)
                results_pool = self.results_pool = data_repo["results_pool"]
                rulesets_final = self.rulesets_final = data_repo["rulesets_final"]
                self.top_feature_dict = data_repo["top_feature_dict"]
                self.top_feature_dict_dis = data_repo["top_feature_dict_dis"]
                self.assign_feat_int_dict(self.top_feature_dict, self.top_feature_dict_dis)
            else:
                ### Step 1: Feature Selection ###
                self.feature_selection(df_features=df_epoch_features, df_labels=df_datapoints_arm["y_raw"])

                ### Step 2: Assocaition Rule Mining ###
                df_rule_twogrps = self.arm_grp_contrast(df_datapoints_wkdy=df_datapoints_arm_wkdy, df_datapoints_wkend=df_datapoints_arm_wkend)

                ### Step 3: Rule Selection ###
                rulesets_final = self.behavior_rules_selection(df_rule_twogrps)

                ### Step 4: Feature Extraction ###
                results_pool = self.feature_extraction(df_datapoints_traintest_wkdy, df_datapoints_traintest_wkend, rulesets_final, flag_train)

                if (self.config["training_params"]["save_and_reload"]):
                    data_repo = {
                        "results_pool": results_pool,
                        "rulesets_final": rulesets_final,
                        "top_feature_dict": self.top_feature_dict,
                        "top_feature_dict_dis": self.top_feature_dict_dis,
                    }
                    with open(self.save_file_path, "wb") as f:
                        pickle.dump(data_repo, f)
        else:
            assert hasattr(self, "rulesets_final")
            rulesets_final = self.rulesets_final
            results_pool = self.feature_extraction(df_datapoints_traintest_wkdy, df_datapoints_traintest_wkend, rulesets_final, flag_train)

        X_tmp = pd.concat(results_pool, axis = 1, ignore_index = False)
        shape1 = X_tmp.shape
        if (flag_train): # Testing dataset does not delete features to ensure compatibility
            X_tmp = X_tmp[X_tmp.columns[(X_tmp.isna().sum(axis = 0) < \
                (X_tmp.shape[0] * self.config["feature_definition"]["empty_feature_filtering_th"]))]] # filter very empty features
        shape2 = X_tmp.shape
        del_cols = shape1[1] - shape2[1]
        X_tmp = X_tmp[X_tmp.isna().sum(axis = 1) < X_tmp.shape[1] / 1.1] # filter very empty person-days
        shape3 = X_tmp.shape
        del_rows = shape2[0] - shape3[0]

        X = deepcopy(X_tmp)
        scl = RobustScaler(quantile_range = (5,95), unit_variance = True).fit(X)
        X[X.columns] = scl.transform(X)

        if (self.verbose > 0):
            print(f"Final rule-based feature: filter {del_cols} cols and {del_rows} rows")
            print(f"NA rate: {100* X.isna().sum().sum() / X.shape[0] / X.shape[1]}%" )

        X = X.fillna(self.NAFILL)

        y = df_datapoints_traintest["y_raw"][X.index]
        pids = df_datapoints_traintest["pid"][X.index]

        self.data_repo = DataRepo(X=X, y=y, pids=pids)
        return self.data_repo

    def get_wks(self, df: pd.DataFrame, wk: str):
        if (wk == "wkdy"):
            return df[df["date"].dt.dayofweek < 5]
        elif (wk == 'wkend'):
            return df[df["date"].dt.dayofweek >= 5]
        else:
            raise ValueError("Invalid wk argument")

    def get_epoch_features(self, df: pd.DataFrame, wk: str):
        if (self.flag_use_norm_features):
            df_mean = df[self.feature_list_norm].mean(axis = 0)
        else:    
            df_mean = df[self.feature_list].mean(axis = 0)
        df_mean.index = [f + f"_{wk}_mean" for f in self.feature_list]
        if (self.flag_use_norm_features):
            df_std = df[self.feature_list_norm].std(axis = 0)
        else:
            df_std = df[self.feature_list].std(axis = 0)
        df_std.index = [f + f"_{wk}_std" for f in self.feature_list]
        return pd.concat([df_mean, df_std])
        
    def rule_threshold_filter(self, df: pd.DataFrame, minsupp: float, minconf: float):
        df_filtered = deepcopy(df)
        df_filtered = df_filtered[
            (
                (
                (df_filtered["supp" + '_grp1'] >= minsupp) &
                (df_filtered["conf" + '_grp1'] >= minconf)
                ) 
                |
                (
                (df_filtered["supp" + '_grp2'] >= minsupp) &
                (df_filtered["conf" + '_grp2'] >= minconf)
                )
            )]
        return df_filtered

    def remove_close_rules(self, ruleset):
        Xs = ruleset["X"].apply(lambda x : x.split(";"))
        Ys = ruleset["Y"].apply(lambda x : x.split(";"))
        idx_to_remove = []
        idx_parent = []
        for idx, X, Y in zip(range(ruleset.shape[0]), Xs, Ys):
            for idxx, X_, Y_ in zip(range(ruleset.shape[0]), Xs, Ys):
                if ((set(X) < set(X_) and set(Y) <= set(Y_))
                or 
                (set(X) <= set(X_) and set(Y) < set(Y_))):
                    idx_to_remove.append(idx)
                    idx_parent.append(idxx)
                    break
        return ruleset.drop(ruleset.index[idx_to_remove])

    def nostraight_pair_rulenum(self, df: pd.DataFrame):
        return sum((df["idx_grp1"] == 0) | (df["idx_grp2"] == 0))
    def nostraight_paired(self, df: pd.DataFrame):
        return df[(df["idx_grp1"] == 0) | (df["idx_grp2"] == 0)]
    def straight_paired(self, df: pd.DataFrame):
        return df[(df["idx_grp1"] != 0) & (df["idx_grp2"] != 0)]

    def prep_model(self, data_train: DataRepo, criteria: str = "balanced_acc") -> sklearn.base.ClassifierMixin:
        super().prep_model()
        set_random_seed(42)

        data_train = deepcopy(data_train)
        
        @ray.remote
        def train_small_cv(data_repo: DataRepo, model_parameters: dict):
            signal.signal(signal.SIGTERM, lambda signalNumber, frame: False)
            X = deepcopy(data_repo.X)
            y = data_repo.y
            pids = data_repo.pids
            pidnum_min = get_min_count_class(labels=y, groups=pids)

            clf = utils_ml.get_clf("adaboost", model_parameters, direct_param_flag = False)

            # select stable features
            selected_features_list = []
            cv_tmp = StratifiedGroupKFold(n_splits=min(20,pidnum_min), shuffle=True, random_state=4200)
            for train_idx, test_idx in cv_tmp.split(X=X, y=y, groups=pids):
                X_tmp = deepcopy(X.iloc[train_idx])
                y_tmp = deepcopy(y.iloc[train_idx])
                corr = {}
                for col in X_tmp:
                    corr[col] =  X_tmp[col].corr(y_tmp)
                df_corr = pd.Series(corr)
                selected_features = df_corr[df_corr.abs() > 0.05].index
                for _ in range(3):
                    clf = utils_ml.get_clf("rf", model_parameters, direct_param_flag = False)
                    clf.fit(X_tmp[selected_features], y_tmp)
                    selected_features = selected_features[np.where(clf.feature_importances_)[0]]
                selected_features_list.append(deepcopy(selected_features))
            
            selected_features_final = [k for k, v in collections.Counter([j for i in selected_features_list for j in i]).items() if v > 10]
            clf = utils_ml.get_clf("adaboost", model_parameters, direct_param_flag = False)
            if (len(selected_features_final) == 0):
                selected_features_final = X.columns # if 0, disregard selection
            cv = StratifiedGroupKFold(n_splits=min(20,pidnum_min), shuffle=True, random_state=42)
            r = cross_validate(clf, X=X[selected_features_final], y=y, groups= pids, cv = cv,
                    scoring = utils_ml.results_report_sklearn, return_train_score=False)
            r = {k:np.mean(v) for k,v in r.items()}
            r.update({"parameters":model_parameters, "selected_features": selected_features_final})
            return r

        n_estimators_list = [5,7,10,13,15,20,25,30,50]
        learning_rate_list = [10**i for i in range(-2,2)]
        max_leaf_nodes_list = [i for i in range(4,30)]
        max_depth_list = [2,3,4,5]
        
        parameters_list = []
        for n_estimators, max_leaf_nodes, learning_rate in itertools.product(n_estimators_list, max_leaf_nodes_list, learning_rate_list):
            parameters_tmp = {"n_estimators":n_estimators, "max_leaf_nodes": max_leaf_nodes,"learning_rate":learning_rate, "random_state":42}
            parameters_list.append(parameters_tmp)
            
        for n_estimators, max_depth, learning_rate in itertools.product(n_estimators_list, max_depth_list, learning_rate_list):
            parameters_tmp = {"n_estimators":n_estimators, "max_depth": max_depth,"learning_rate":learning_rate, "random_state":42}
            parameters_list.append(parameters_tmp)
        
        data_train_id = ray.put(data_train)
        results_list = ray.get([train_small_cv.remote(data_train_id,i) for i in parameters_list])
        results_list = pd.DataFrame(results_list)

        best_row = results_list.iloc[results_list[f"test_{criteria}"].argmax()]
        best_params = best_row['parameters']
        best_features = best_row['selected_features']
        if (self.verbose > 0):
            print(best_row)
            print(best_params)
        
        return DepressionDetectionClassifier_ML_xu_interpretable(model_params=best_params, selected_features=best_features)
