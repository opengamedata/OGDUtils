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

# consider making a general version with parameter for filename, index columns
# def getLakelandDecJanLogDF():
#     """

#     :return: (df, metadata List[str])
#     """
#     # define paths for DecJanLog
#     _proc_zip_url_dec = 'https://opengamedata.fielddaylab.wisc.edu/data/LAKELAND/LAKELAND_20191201_to_20191231_de09c18_proc.zip'
#     _proc_zip_path_jan = 'Data/Raw Log Data/LAKELAND_20200101_to_20200131_a9720c1_proc.zip'
#     # get the data
#     metadata = []
#     zipfile_dec, meta = openZipFromURL(_proc_zip_url_dec)
#     metadata.extend(meta)
#     zipfile_jan, meta = openZipFromPath(_proc_zip_path_jan)
#     metadata.extend(meta)
#     # put the data into a dataframe
#     df = pd.DataFrame()
#     for zf in [zipfile_dec, zipfile_jan]:
#         with zf.open(zf.namelist()[0]) as f:
#             df = pd.concat([df, pd.read_csv(f, index_col=['sessID', 'num_play'], comment='#')], sort=True)
#     df['sessID'] = [x[0] for x in df.index]
#     df['num_play'] = [x[1] for x in df.index]
#     return df, metadata


def get_lakeland_default_filter(lvlstart: Optional[int] = None, lvlend: Optional[bool] = None, no_debug: Optional[bool] = True,
                                min_sessActiveEventCount: Optional[int] = 10,
                                min_lvlstart_ActiveEventCount: Optional[int] = 3,
                                min_lvlend_ActiveEventCount: Optional[int] = 3, min_sessDuration: Optional[int] = 300, max_sessDuration: Optional[int] = None, cont: Optional[bool] = False) -> List[str]:
    """

    :param lvlstart: levelstart to be used for other parameters (None if not used)
    :param lvlend: levelend to be used for other parameters (None if not used)
    :param no_debug: boolean whether or not to use only players that have used SPYPARTY or only not used SPYPARTY  (None if not used)
    :param min_sessActiveEventCount:  (None if not used)
    :param min_lvlstart_ActiveEventCount:  (None if not used)
    :param min_lvlend_ActiveEventCount:  (None if not used)
    :param min_sessDuration:  (None if not used)
    :param max_sessDuration:  (None if not used)
    :param cont:  (None if not used)
    :return:
    """
    get_lakeland_default_filter()
    query_list = []

    if no_debug:
        query_list.append('debug == 0')
    if min_sessActiveEventCount is not None:
        query_list.append(
            f'sess_ActiveEventCount >= {min_sessActiveEventCount}')
    if lvlstart is not None and min_lvlstart_ActiveEventCount is not None:
        query_list.append(
            f'lvl{lvlstart}_ActiveEventCount >= {min_lvlstart_ActiveEventCount}')
    if lvlend is not None and min_lvlend_ActiveEventCount is not None:
        query_list.append(
            f'lvl{lvlend}_ActiveEventCount >= {min_lvlend_ActiveEventCount}')
    if min_sessDuration is not None:
        query_list.append(f'sessDuration >= {min_sessDuration}')
    if max_sessDuration is not None:
        query_list.append(f'sessDuration <= {max_sessDuration}')
    if cont is not None:
        query_list.append(f'_continue == {int(cont)}')

    return query_list

# def describe_lvl_feats_lakeland(df, fbase_list, lvl_range, level_time=300, level_overlap=30):
#     """
#     Calculates sum/avg of given level base features (fnames without lvlN_ prefix) in the level range.
#     Will automatically filter out players who did not complete the given level range in the df
#     May have a bug.

#     :param level_time: number of seconds per level (window)
#     :param level_overlap: number of overlap seconds per level (window)
#     :rtype: (df, List[str]) where the new df includes sum_ and avg_lvl_A_to_B.
#     :param df: dataframe to pull from and append to
#     :param fbase_list: list of feature bases (fnames without lvlN_ prefix)
#     :param lvl_range: range of levels to choose. typically range(min_level, max_level+1)
#     """
#     metadata = []
#     metadata.append(f'*arg* lvlfeats = {fbase_list}')
#     metadata.append(f'*arg* lvlrange = {lvl_range}')
#     if not fbase_list:
#         return df, metadata
#     lvl_start, lvl_end = lvl_range[0], lvl_range[-1]
#     query = f'sessDuration > {(level_time - level_overlap) * (lvl_end) + level_time}'
#     df = df.query(query)
#     metadata.append(
#         f'Describe Level Feats lvls {lvl_start} to {lvl_end}. Assuming WINDOW_SIZE_SECONDS={level_time} and WINDOW_OVERLAP_SECONDS={level_overlap}, filtered by ({query})')
#     fromlvl, tolvl = lvl_range[0], lvl_range[-1]
#     sum_prefix = f'sum_lvl_{fromlvl}_to_{tolvl}_'
#     avg_prefix = f'avg_lvl_{fromlvl}_to_{tolvl}_'
#     for fn in fbase_list:
#         tdf = df[[f'lvl{i}_{fn}' for i in lvl_range]].fillna(0).copy()
#         df[sum_prefix + fn] = tdf.sum(axis=1)
#         df[avg_prefix + fn] = tdf.mean(axis=1)
#     return df, metadata

def get_feat_selection_lakeland(df,  max_lvl=9):
    """
    Gets the feature selection widget.
    :param df:
    :param max_lvl:
    :return:
    """
    start_level = widgets.IntSlider(value=0, min=0, max=max_lvl, step=1, description='Start Level:',
                                    disabled=False, continuous_update=False, orientation='horizontal', readout=True, readout_format='d')
    end_level = widgets.IntSlider(value=0, min=0, max=max_lvl, step=1, description='End Level:',
                                  disabled=False, continuous_update=False, orientation='horizontal', readout=True, readout_format='d')
    level_selection = widgets.GridBox([start_level, end_level])

    def change_start_level(change):
        end_level.min = start_level.value
        if end_level.value < start_level.value:
            end_level.value = start_level.value
    start_level.observe(change_start_level, names="value")

    lvl_feats = sorted(set([f[5:] for f in df.columns if f.startswith('lvl')]))
    sess_feats = sorted(
        set([f[5:] for f in df.columns if f.startswith('sess_')]))
    other_feats = sorted(set([f for f in df.columns if not f.startswith(
        'lvl') and not f.startswith('sess_')]))
    selection_widget = widgets.GridBox([multi_checkbox_widget(lvl_feats, 'lvl'),
                                        multi_checkbox_widget(
                                            sess_feats, 'sess'),
                                        multi_checkbox_widget(
                                            other_feats, 'other'),
                                        level_selection],
                                       layout=widgets.Layout(grid_template_columns=f"repeat(3, 500px)"))

    return selection_widget
