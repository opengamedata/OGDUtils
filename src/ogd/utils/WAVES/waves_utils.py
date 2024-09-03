# standard library imports
import sys
import copy
import pickle
import os
from collections import Counter
from io import BytesIO
from zipfile import ZipFile
import copy
import pickle
from math import ceil
import importlib
import urllib.request

# math imports
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
sns.set()

# google imports

# Jupyter Imports
from IPython.display import display
import ipywidgets as widgets
# from google.colab import files

# ML imports
# models
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeCV, SGDRegressor
from sklearn.svm import LinearSVR

# preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split

# sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, RepeatedEditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek

# metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_precision_recall_curve, plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, accuracy_score

# other
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# custom imports
import general.feature_utils as feat_util

def get_feat_selection_waves(df, max_lvl=34):
    """
    Gets the feature selection widget.
    :param df:
    :param max_lvl:
    :return:
    """
    start_level = widgets.IntSlider(value=0, min=0, max=max_lvl, step=1, description='Start Level:',
                                    disabled=False, continuous_update=False, orientation='horizontal', readout=True,
                                    readout_format='d')
    end_level = widgets.IntSlider(value=0, min=0, max=max_lvl, step=1, description='End Level:',
                                  disabled=False, continuous_update=False, orientation='horizontal', readout=True,
                                  readout_format='d')
    level_selection = widgets.GridBox([start_level, end_level])

    def change_start_level(change):
        end_level.min = start_level.value
        if end_level.value < start_level.value:
            end_level.value = start_level.value

    start_level.observe(change_start_level, names="value")

    lvl_feats = sorted(set([''.join(f.split('_')[1:])
                            for f in df.columns if f.startswith('lvl')]))
    sess_feats = sorted(
        set([f[7:] for f in df.columns if f.startswith('session')]))
    other_feats = sorted(set([f for f in df.columns if not f.startswith(
        'lvl') and not f.startswith('session')]))
    selection_widget = widgets.GridBox([multi_checkbox_widget(lvl_feats, 'lvl'),
                                        multi_checkbox_widget(
                                            sess_feats, 'sess'),
                                        multi_checkbox_widget(
                                            other_feats, 'other'),
                                        level_selection],
                                       layout=widgets.Layout(grid_template_columns=f"repeat(3, 500px)"))

    return selection_widget
