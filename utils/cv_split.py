import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import check_array
from data_loader.data_loader_ml import DataRepo

class GroupKFoldRandom(_BaseKFold):
    """Modifed from the original sklearn GroupKFold to add random state
        https://github.com/scikit-learn/scikit-learn/blob/baf828ca1/sklearn/model_selection/_split.py#L453
    """
    def __init__(self, n_splits=5, random_state = None):
        super().__init__(n_splits, shuffle=True, random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)

        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError(
                "Cannot have number of splits n_splits=%d greater"
                " than the number of groups: %d." % (self.n_splits, n_groups)
            )

        # Weight groups by their number of occurrences
        n_samples_per_group = np.bincount(groups)

        # Distribute the most frequent groups first
        indices = np.argsort(n_samples_per_group)[::-1]
        n_samples_per_group = n_samples_per_group[indices]

        # Total weight of each fold
        n_samples_per_fold = np.zeros(self.n_splits)

        # Mapping from group index to fold index
        group_to_fold = np.zeros(len(unique_groups))

        # Distribute samples by adding the largest weight to the lightest fold
        for group_index, weight in enumerate(n_samples_per_group):
            lightest_fold = np.argmin(n_samples_per_fold)
            n_samples_per_fold[lightest_fold] += weight
            group_to_fold[indices[group_index]] = lightest_fold

        indices = group_to_fold[groups]

        for f in range(self.n_splits):
            yield np.where(indices == f)[0]

    def split(self, X, y=None, groups=None):
        return super().split(X, y, groups)

def judge_corner_cvsplit(cv:_BaseKFold, data_repo: DataRepo) -> bool:
    """Avoid splitting cases with single label or device_type"""
    same_label_flag = False
    for train_idx, test_idx in cv.split(X = data_repo.X, y = data_repo.y, groups=data_repo.pids):
        if (len(set(data_repo.y.iloc[train_idx])) == 1):
            same_label_flag = True
            break
        elif ("device_type" in data_repo.X):
            devices_train = data_repo.X.values[train_idx,-1]
            train_idx_ios = np.where(devices_train == 1)[0]
            train_idx_android = np.where(devices_train != 1)[0]
            y_tmp = data_repo.y.iloc[train_idx].values
            if (len(set(y_tmp[train_idx_ios])) == 1 or len(set(y_tmp[train_idx_android])) == 1):
                same_label_flag = True
                break
    return same_label_flag

