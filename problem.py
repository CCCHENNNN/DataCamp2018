import os
import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


problem_title = 'TV Channel Commercial Detection'
_target_column_name = 'Label'
_prediction_label_names = [-1, 1]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()


from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from sklearn.metrics import cohen_kappa_score
class Kappa(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = -1.0
    maximum = 1.0

    def __init__(self, name='Kappa', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        score = cohen_kappa_score(y_true_label_index, y_pred_label_index)
        return score

from sklearn.metrics import matthews_corrcoef
class Matthews_corrcoef(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = -1.0
    maximum = 1.0

    def __init__(self, name='Matthews', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        score = matthews_corrcoef(y_true_label_index, y_pred_label_index)
        return score

score_types = [
    Kappa(name='kappa', precision=3),
    Matthews_corrcoef(name='matthews', precision=3),
    rw.score_types.ROCAUC(name='roc_auc', precision=3),
    rw.score_types.Accuracy(name='acc', precision=3),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.5, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    test = os.getenv('RAMP_TEST_MODE', 0)
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    data = data.fillna(0)
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


