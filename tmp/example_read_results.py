import pickle
import pandas, numpy

print("\nIn these two examples, they will be saved at `evaluation_output/evaluation_single_dataset_within_user/dep_weekly/ml_chikersal.pkl\n",
    "and `evaluation_output/evaluation_allbutone_datasets/dep_weekly/dl_reoreder.pkl`, respectively.\n\n")

# with open("evaluation_output/evaluation_single_dataset/dep_endterm/ml_chikersal.pkl", "rb") as f:
#     evaluation_results = pickle.load(f)
#     df = pandas.DataFrame(evaluation_results["results_repo"]["dep_endterm"]).T
#     print(pandas.DataFrame([df.apply(lambda row: [numpy.mean(r["test_balanced_acc"]) for r in row]).mean(axis=1),
#         df.apply(lambda row: [numpy.mean(r["test_roc_auc"]) for r in row]).mean(axis=1)],
#         index = ["test_balanced_acc", "test_roc_auc"]).T)

with open("evaluation_output/evaluation_single_dataset_within_user/dep_weekly/ml_chikersal.pkl", "rb") as f:
    evaluation_results = pickle.load(f)
    df = pandas.DataFrame(evaluation_results["results_repo"]["dep_weekly"]).T
    print(df[["test_balanced_acc", "test_roc_auc"]])

with open("evaluation_output/evaluation_allbutone_datasets/dep_weekly/dl_reorder.pkl", "rb") as f:
    evaluation_results = pickle.load(f)
    df = pandas.DataFrame(evaluation_results["results_repo"]["dep_weekly"]).T
    print(df[["test_balanced_acc", "test_roc_auc"]])