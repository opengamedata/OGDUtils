# google imports

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
import feature_utils as feat_util


def response_boxplot(df, category, verbose=False):
    print('\n'+category)
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    qs = ['EFL_yes_no', 'skill_low_med_high', 'enjoy_high_med_low_none']
    for i, f in enumerate(['R0_quiz_response', 'R1_quiz_response', 'R2_quiz_response', ]):
        if verbose:
            print(qs[i])
        bp = df.boxplot(column=category, by=df[f].astype(
            'category'), ax=axs[i])
        bp.set_xlabel('')
        for choice in range(df[f].min(), df[f].max()+1):
            query = f"{f}=={choice}"
            cat_df = df.query(query)[category]
            num_chose = len(cat_df)
            mean = cat_df.mean()
            std = cat_df.std()
            if verbose:
                print(
                    f'{f} # chose {choice}: {num_chose} ({round(num_chose/len(df)*100)}%). Avg {mean}, std {std}.')
    plt.suptitle(f'{category} Boxplot')
    fig.show()


def group_by_func(df, func, title='', show=True):
    r0_groups = {0: 'native', 1: 'nonnative'}
    r1_groups = {0: 'not very good skill',
                 1: 'okay skill', 2: 'very good skill'}
    r2_groups = {0: 'really enjoy', 1: 'enjoy', 2: 'okay', 3: 'not enjoy'}
    def group_string(r0, r1, r2): return ', '.join(
        [r0_groups[r0], r1_groups[r1], r2_groups[r2]])
    result_dfs = [pd.DataFrame(index=r1_groups.values(), columns=r2_groups.values(
    )), pd.DataFrame(index=r1_groups.values(), columns=r2_groups.values())]
    if show:
        print(f'{"-"*6}  {title}  {"-"*6}')
    for r0 in [0, 1]:
        subtitle = "Nonnatives" if r0 else "Natives"
        if show:
            print(f'\n{subtitle}:')
        tdf0 = df.query(f"R0_quiz_response == {r0}")
        for r1 in [0, 1, 2]:
            tdf1 = tdf0.query(f"R1_quiz_response == {r1}")
            for r2 in [0, 1, 2, 3]:
                tdf2 = tdf1.query(f"R2_quiz_response == {r2}")
                result_dfs[r0].loc[r1_groups[r1], r2_groups[r2]
                                   ] = func(df, tdf0, tdf1, tdf2)
        if show:
            display(result_dfs[r0])
    return result_dfs


def standard_group_by_func(fulldf, per_category_stats_list=None):
    per_category_stats_list = None or ['sess_count_clicks',
                                       'sess_count_hovers',
                                       'sess_meaningful_action_count',
                                       'sess_EventCount',
                                       'sess_count_notebook_uses',
                                       'sess_avg_time_between_clicks',
                                       'sess_first_enc_words_read',
                                       'sess_first_enc_boxes_read',
                                       'sess_num_enc',
                                       'sess_first_enc_duration',
                                       'sess_first_enc_avg_wps',
                                       'sess_first_enc_var_wps',
                                       'sess_first_enc_avg_tbps',
                                       'sess_first_enc_var_tbps',
                                       'sess_start_obj',
                                       'sess_end_obj',
                                       'start_level',
                                       'max_level',
                                       'sessDuration']
    dfs_list = []
    title_list = []

    def df_func(df, tdf0, tdf1, tdf2): return len(tdf2)
    title = 'count'
    dfs = group_by_func(fulldf, df_func, title)
    dfs_list.append(dfs)
    title_list.append(title)

    def df_func(df, tdf0, tdf1, tdf2): return round(len(tdf2)/len(df)*100, 2)
    title = 'percent total pop'
    dfs = group_by_func(fulldf, df_func, title)
    dfs_list.append(dfs)
    title_list.append(title)

    def df_func(df, tdf0, tdf1, tdf2): return round(len(tdf2)/len(tdf0)*100, 2)
    title = 'percent native class pop'
    dfs = group_by_func(fulldf, df_func, title)
    dfs_list.append(dfs)
    title_list.append(title)

    for category in per_category_stats_list:
        df_func = get_avg_std_df_func(category)
        title = f'(avg, std) {category}'
        dfs = group_by_func(fulldf, df_func, title)
        dfs_list.append(dfs)
        title_list.append(title)
    return title_list, dfs_list


def get_avg_std_df_func(category_name):
    def inner(df, tdf0, tdf1, tdf2):
        mean = tdf2[category_name].mean()
        std = tdf2[category_name].std()
        if not pd.isna(mean):
            mean = round(mean, 2)
        if not pd.isna(std):
            std = round(std, 2)
        return (mean, std)
    return inner


def html_stats(df):
    html_strs = ['<div class="container">', '<h3>{Stats}</h3>']
    qs = ['EFL_yes_no', 'skill_low_med_high', 'enjoy_high_med_low_none']
    html_strs.append(f'<p> Total pop {len(df)} </p>')
    for i, f in enumerate(['R0_quiz_response', 'R1_quiz_response', 'R2_quiz_response', ]):
        html_strs.append(f'<p> {qs[i]}</p>')
        for choice in range(df[f].min(), df[f].max()+1):
            query = f"{f}=={choice}"
            cat_df = df.query(query)
            num_chose = len(cat_df)
            html_strs.append(
                f'<p>{f} # chose {choice}: {num_chose} ({round(num_chose/len(df)*100)}%).</p>')
    return '\n'.join(html_strs+['</div>'])


def full_html(base_df, title_list, dfs_list, suptitle=None):
    HEADER = '''<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Document</title>
</head>

<body>
  <style>
    .flex-container {
      display: flex;
      flex-wrap: wrap;
    }

    .container {
      border: thick solid black;
      padding: 10px;
      margin: 5px;
    }

    .container table:nth-of-type(2) td {
      background-color: rgb(161, 161, 230);
    }

    .container table:nth-of-type(2) th {
      background-color: rgb(20, 20, 194);
      color: white;
    }

    .container table:nth-of-type(2n-1) td {
      background-color: rgb(235, 158, 158);
    }

    .container table:nth-of-type(2n-1) th {
      background-color: rgb(160, 11, 11);
      color: white;
    }
    .break {
  flex-basis: 100%;
  height: 0;
}
  </style>
  <div class="flex-container">'''
    FOOTER = '''  </div>
    </body>

    </html>'''

    def table_header(title): return f'''    <div class="container">
        <h3>{title}</h3>'''
    table_footer = '''    </div>'''
    def table_html(title, dfs): return '\n'.join([table_header(
        title), "<p>Natives:</p>", dfs[0].to_html(), "<p>Nonnatives:</p>", dfs[1].to_html(), table_footer])

    if suptitle is not None:
        suptitle = f'<h2>{suptitle}</h2>\n<div class="break"></div> <!-- break -->'
    else:
        suptitle = ''
    return '\n'.join([HEADER, suptitle, html_stats(base_df)] +
                     [table_html(t, dfs) for t, dfs in zip(title_list, dfs_list)] +
                     [FOOTER])


def download_full_html(base_df, title_list, dfs_list, filename, suptitle=None):
    with open(filename, 'w+') as f:
        f.write(full_html(base_df, title_list, dfs_list, suptitle=suptitle))
        print("Wrote to", filename)
    files.download(filename)


onext_int_feats = [f'obj{i}_onext_int' for i in range(80)]
onext_int_cats = [["nan", 1],
                  ["nan", 11],
                  ["nan", 12, 86, 111, 125],
                  ["nan", 13, 14, 113, 116, 118],
                  ["nan", 14, 15, 113, 114, 116, 118],
                  ["nan", 13, 15, 113, 114, 116, 118],
                  ["nan", 16, 86, 115, 118, 132, 161],
                  ["nan", 17, 86, 115, 118, 128, 161],
                  ["nan", 18, 86, 115, 118, 161],
                  ["nan", 19, 86, 117, 118, 127, 133, 134, 161],
                  ["nan", 20, 133, 134, 136],
                  ["nan", 2, 80, 81, 82, 83],
                  ["nan", 21, 86, 117, 127, 136, 137, 161],
                  ["nan", 22, 137, 141],
                  ["nan", 23, 24, 86, 117, 127, 136, 161],
                  ["nan", 23, 24, 117, 127, 136, 161],
                  ["nan", 25, 86, 117, 118, 127, 136, 140, 147, 151, 161],
                  ["nan", 26, 142, 145],
                  ["nan", 27, 143],
                  ["nan", 28, 86, 117, 118, 136, 140, 150, 161],
                  ["nan", 29, 119, 130],
                  ["nan", 29, 30, 35, 86, 117, 118, 126, 136, 140, 149],
                  ["nan", 3, 80, 82, 83, 86, 87, 88, 93],
                  ["nan", 31, 38],
                  ["nan", 32, 153],
                  ["nan", 33, 154],
                  ["nan", 34, 155],
                  ["nan", 35, 156],
                  ["nan", 36, 157],
                  ["nan", 37, 158],
                  ["nan", 30],
                  ["nan", 39, 163],
                  ["nan", 40, 160],
                  ["nan", 3],
                  ["nan", 41, 164, 166],
                  ["nan", 42, 166],
                  ["nan", 30],
                  ["nan", 44, 85, 125],
                  ["nan", 29, 45, 47, 84, 118, 125, 136, 140, 149, 168, 169, 184],
                  ["nan", 45, 46, 169, 170],
                  ["nan", 29, 45, 47, 92, 118, 136, 140, 149, 169, 184],
                  ["nan", 29, 45, 48, 92, 118, 140, 149, 168, 184],
                  ["nan", 46, 49, 168],
                  ["nan", 46, 50, 168, 170],
                  ["nan", 5, 80, 82, 83, 86, 89, 91, 95, 97, 125],
                  ["nan", 29, 51, 92, 118, 136, 140, 149, 168, 184],
                  ["nan", 52, 92, 118, 136, 149, 171, 184],
                  ["nan", 53, 54, 92, 118, 136, 140, 149, 184],
                  ["nan", 53, 54, 55, 59, 60, 90, 92, 94,
                      118, 136, 140, 149, 168, 184],
                  ["nan", 53, 55, 59, 60, 90, 92, 94, 118, 136, 140, 149, 184],
                  ["nan", 55, 56, 59, 60, 149, 174],
                  ["nan", 57, 59, 60, 174],
                  ["nan", 58, 59, 60, 136, 172, 174, 184],
                  ["nan", 29, 59, 60, 61, 92, 118, 136, 149, 168, 172, 184],
                  ["nan", 55, 56, 57, 58, 60, 61, 140, 172, 174, 184],
                  ["nan", 6, 80, 82, 83, 86, 98, 100, 125],
                  ["nan", 55, 56, 57, 58, 59, 61, 92, 118,
                      136, 140, 149, 172, 174, 184],
                  ["nan", 59, 62, 136, 140, 149, 172, 173, 175, 184],
                  ["nan", 63, 64, 176],
                  ["nan", 64, 66, 149, 175, 184],
                  ["nan", 29, 65, 66, 92, 118, 136, 140, 172, 175, 177, 184],
                  ["nan", 66, 67, 68, 92, 118, 136, 140, 146, 175, 177, 184],
                  ["nan", 67, 144],
                  ["nan", 29, 64, 65, 68, 92, 118, 131, 136,
                      140, 148, 149, 172, 175, 177, 184],
                  ["nan", 92, 118, 122, 123, 124, 131, 136, 140,
                      146, 148, 168, 172, 175, 177, 184],
                  ["nan", 70],
                  ["nan", 7],
                  ["nan", 71, 178],
                  ["nan", 72, 179],
                  ["nan", 73, 180],
                  ["nan", 74, 181],
                  ["nan", 75, 182],
                  ["nan", 69],
                  ["nan", 77, 78, 185],
                  ["nan", 78, 185],
                  ["nan", 79],
                  [0],
                  ["nan", 8],
                  ["nan", 9, 103],
                  ["nan", 104, 105, 108]]

QA_1_feats = [f'Q{i}_A1' for i in range(19)]
QA_1_cats = [['0', 'A', 'B', 'C', 'D'],
             ['0', 'A', 'B', 'C', 'D'],
             ['0', 'A', 'B', 'C', 'D'],
             ['0', 'A', 'B', 'C', 'D'],
             ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q'],
             ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q'],
             ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q'],
             ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q'],
             ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q'],
             ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q'],
             ['0', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q'],
             ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q'],
             ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q'],
             ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q'],
             ['0', 'A', 'B', 'C', 'D', 'F', 'G', 'I', 'M', 'N', 'O', 'P', 'Q',
              'R', 'S', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f'],
             ['0', 'Q', 'V', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f'],
             ['0', '?', 'X', 'Y', 'Z', 'b', 'c', 'd', 'e', 'f'],
             ['0', 'X', 'Y', 'b', 'c', 'd', 'e', 'f'],
             ['0', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f']]


def get_preprocessor(df, scaler=StandardScaler(), imputer=SimpleImputer(strategy='constant'), bool_dtype='int64'):
    """
    By default has a number of steps:
    1. drops columns from use in preprocessor if present:
       - [f'Q{q}_answers' for q in range(19)]
       - ["play_year", "play_month", "play_day", "play_hour", "play_minute", "play_second"]
       - ["_continue", "continue", "save_code", "music", "hq", "fullscreen", "persistentSessionID"]
    2. Creates a preprocessor for all non-y columns and non-boolean columns with the following steps:
       a. Standard Scaler (0 mean, 1 std)
       b. Simple Imputer(strategy='constant') (fill NaN with 0)
    3. Fits the preprocessor to the given X
    4. returns the unfitted preprocessor (sklearn pipeline), and the unprocessed X dataframe
    :param df: jowilder dataframe
    :param scaler: sklearn compatible scaler
    :param imputer: sklearn compatible imputer
    :return: the unfitted preprocessor (sklearn pipeline), and the unprocessed X dataframe
    """
    df = df.drop(
        [f'Q{q}_answers' for q in range(19)] + ["play_year", "play_month", "play_day", "play_hour", "play_minute",
                                                "play_second",
                                                "_continue", "continue", "save_code", "music", "hq", "fullscreen",
                                                "persistentSessionID", ], axis=1, errors='ignore').copy()
    y_cols, bool_cols, num_cols = separate_columns(df, bool_dtype=bool_dtype)
    X = df.loc[:, num_cols+bool_cols]

    # too complicated to allow for pipeline order
    # pipeline_strings = [pipeline_order[i:i+2] for i in range(0,len(pipeline_order),2)]
    # transformers = []
    # num_sa, num_sc, num_im = 0,0,0
    # for s in pipeline_strings:
    #     if s == 'Sa':
    #         transformer = make_pipeline(sampler)
    #         cols = num_cols + bool_cols
    #         name = f'{s}{num_sa}'
    #         num_sa += 1
    #     elif s == 'Sc':
    #         transformer = scaler
    #         name = f'{s}{num_sc}'
    #         cols = num_cols
    #         num_sc += 1
    #     elif s == 'Im':
    #         transformer = imputer
    #         name = f'{s}{num_im}'
    #         cols = num_cols
    #         num_im += 1
    #     else:
    #         raise ValueError("Pipeline substrings must be Sa Sc or Im")
    #     transformers.append((name, transformer, cols))

    def col_str_to_int(col_strs): return [
        X.columns.get_loc(s) for s in col_strs]
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', make_pipeline(scaler, imputer), col_str_to_int(num_cols)),
            ('bool', 'passthrough', col_str_to_int(bool_cols))
        ],
        remainder='drop')
    return column_transformer, X


def get_ys(df):
    """

    :rtype: dictionary of y columns (df series). keys: y0,y1,y2,y1_bin,y2_bin,y1_bin_x,y2_bin_x
    """
    ys = {}
    for key, y_col in [
        ('y0', 'R0_quiz_response'),
        ('y1', 'R1_quiz_response'),
        ('y2', 'R2_quiz_response'),
        ('y1_bin', 'R1_quiz_response_bin'),
        ('y1_bin_0v12', 'R1_quiz_response_0v12'),
        ('y1_bin_01v2', 'R1_quiz_response_01v2'),
        ('y1_bin_x', 'R1_quiz_response_bin_x'),
        ('y2_bin', 'R2_quiz_response_bin'),
        ('y2_bin_x', 'R2_quiz_response_bin_x'),
        ('y2_bin_0v123', 'R2_quiz_response_bin0v123'),
        ('y2_bin_01v23', 'R2_quiz_response_bin01v23'),
        ('y2_bin_012v3', 'R2_quiz_response_bin012v3'),
    ]:
        if y_col in df.columns:
            ys[key] = df.loc[:, y_col].astype('category').copy()
    return ys


def separate_columns(df, bool_dtype='int64', expect_bool_cols = True) -> (list, list, list):
    """

    :param df:
    :param bool_dtype: Defaults to 'int64'. Should be int64 if coming from import csv otherwise could be 'uint8'
    if coming from the pd dummies.
    :return: tuple of lists of column names for y_columns, bool_columns, and integer_columns
    """
    y_cols = [col for col in df.columns if 'quiz_response' in col]
    bool_cols = [col for col in df.select_dtypes(include=[bool_dtype])
                 if np.isin(df[col].dropna().unique(), [0, 1]).all() and
                 col not in y_cols]
    num_cols = [
        col for col in df.columns if col not in bool_cols and col not in y_cols]
    if not bool_cols and expect_bool_cols:
        print('Warning! No bool columns. Consider changing bool_dtype="int_64" to "uint8"')
    return y_cols, bool_cols, num_cols


end_obj_to_last_Q = {
    9: 0,
    10: 3,
    11: 3,
    12: 3,
    13: 3,
    14: 3,
    15: 3,
    16: 3,
    17: 3,
    18: 3,
    19: 3,
    20: 3,
    21: 3,
    22: 3,
    23: 3,
    24: 3,
    25: 3,
    26: 3,
    27: 3,
    28: 3,
    29: 3,
    30: 3,
    31: 3,
    32: 4,
    33: 5,
    34: 6,
    35: 7,
    36: 8,
    37: 9,
    38: 9,
    39: 10,
    40: 11,
    41: 12,
    42: 13,
    43: 13,
    44: 13,
    45: 13,
    46: 13,
    47: 13,
    48: 13,
    49: 13,
    50: 13,
    51: 13,
    52: 13,
    53: 13,
    54: 13,
    55: 13,
    56: 13,
    57: 13,
    58: 13,
    59: 13,
    60: 13,
    61: 13,
    62: 13,
    63: 13,
    64: 13,
    65: 13,
    66: 13,
    67: 13,
    68: 13,
    69: 13,
    70: 13,
    71: 14,
    72: 15,
    73: 16,
    74: 17,
    75: 18,
    76: 18,
    77: 18,
    78: 18,
    79: 18,
}

end_obj_to_last_lvl = {
    0:	0,
    1:	0,
    2:	0,
    3:	1,
    4:	2,
    5:	2,
    6:	3,
    7:	3,
    8:	4,
    9: 4,
    10: 4,
    11: 4,
    12: 5,
    13: 6,
    14: 6,
    15: 6,
    16: 6,
    17: 6,
    18: 6,
    19: 7,
    20: 7,
    21: 8,
    22: 8,
    23: 9,
    24: 9,
    25: 9,
    26: 10,
    27: 10,
    28: 11,
    29: 11,
    30: 12,
    31: 12,
    32: 12,
    33: 12,
    34: 12,
    35: 12,
    36: 12,
    37: 12,
    38: 12,
    39: 12,
    40: 12,
    41: 12,
    42: 12,
    43: 13,
    44: 13,
    45: 14,
    46: 15,
    47: 15,
    48: 16,
    49: 16,
    50: 17,
    51: 17,
    52: 18,
    53: 18,
    54: 18,
    55: 18,
    56: 18,
    57: 18,
    58: 18,
    59: 18,
    60: 18,
    61: 18,
    62: 19,
    63: 19,
    64: 19,
    65: 20,
    66: 20,
    67: 21,
    68: 21,
    69: 22,
    70: 22,
    71: 22,
    72: 22,
    73: 22,
    74: 22,
    75: 22,
    76: 22,
    77: 23,
    78: 23,
    79: 23,
}


class GridSearcher():

    def __init__(self, csv_fpath=None, df=None, preprocessor=None, fillna=0, meta=[], expect_bool_cols=True):
        # either give csv_fpath or df.
        assert csv_fpath or not df.empty
        print(f'Loading from {csv_fpath}...')
        # load df
        if df is None:
            print(f'Loading from {csv_fpath}...')
            self.df, self.meta = feat_util.open_csv_from_path_with_meta(
                csv_fpath, index_col=0)
        else:
            self.df, self.meta = df, meta

        # set X and ys, and preprocessor
        if not preprocessor:
            self.preprocessor, self.X = get_preprocessor(self.df)
            self.X = self.X.fillna(fillna)
        else:
            _, bool_cols, num_cols = separate_columns(self.df, expect_bool_cols=expect_bool_cols)
            self.X = df[bool_cols+num_cols]
            self.preprocessor = preprocessor
        self.ys = get_ys(self.df)

        # set object vars
        self.model_dict = {}
        self.cur_model = None

    def split_data(self):
        nonnull_X, nonnull_y = feat_util.remove_nan_labels(self.X, self.y)
        X_train, X_test, y_train, y_test = train_test_split(
            nonnull_X, nonnull_y, test_size=0.2, random_state=1)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def set_y(self, y_key=None, other_col=None):
        if y_key:
            print(f'Switching to {y_key}...')
            self.y = self.ys[y_key]
        elif other_col:
            self.y = self.X[other_col]
            self.X = self.X.drop(other_col, axis=1)
        else:
            print("Did not change y. Invalid inputs.")
        self.split_data()

    def run_fit(self, classifier, sampler=None, verbose=False, preprocess_twice=True, sampler_index=None, full_pipeline=False):
        # fit self.cur_model as a pipeline of the given preprocessor, sampler, preprocessor, classifer
        # if preprocess_twice is false, self.cur_model is sampler, preprocessor, classifier
        # if full_pipeline and sampler index, self.cur_model is the classifier 
        # (must be a pipeline containing a sampler or a placeholder (None) for the sampler)
        if full_pipeline:
            assert sampler_index is not None
            clf = classifier
        elif preprocess_twice:
            clf = make_pipeline(self.preprocessor, sampler,
                                copy.deepcopy(self.preprocessor), classifier)
            sampler_index = 1
        else:
            clf = make_pipeline(sampler, self.preprocessor, classifier)
            sampler_index = 0

        self._sampling_pipeline = clf[:sampler_index+1]
        self._classifying_pipeline = clf[sampler_index+1:]
        if clf[sampler_index] is not None:
            self.X_train_sampled, self.y_train_sampled = self._sampling_pipeline.fit_resample(
                self.X_train, self.y_train)
        else:
            self.X_train_sampled, self.y_train_sampled = self.X_train, self.y_train
            clf = self._classifying_pipeline

        # model_name = f'{sampler} {classifier}'
        # if verbose:
        #     print(f'Running {model_name}.')
        self._classifying_pipeline.fit(
            self.X_train_sampled, self.y_train_sampled)
        self.cur_model = clf
        # if verbose:
        #     print("model trained to: %.3f" %
        #           clf.score(self.X_train, self.y_train))
        #     print("model score: %.3f" % clf.score(self.X_test, self.y_test))
        return clf

    def metrics(self, graph_dir=None, graph_prefix=None, binary_classification=True):
        # return list of (metric: float, metric_name: str) tuples of metrics of given classifier (default: self.cur_model)
        # can only do metrics for binary classification as of right now
        assert binary_classification
        metric_list = []
        clf = self.cur_model

        # label metrics
        if graph_prefix:
            for flipped_labels in [False, True]:
                flipped_labels_suffix = '' if not flipped_labels else '_flipped'
                fig, axes = plt.subplots(3, 3, figsize=(20, 20))
                for i, (yarray, Xarray, label) in enumerate([(self.y_test, self.X_test, 'test'),
                                                             (self.y_train_sampled,
                                                              self.X_train_sampled, 'train'),
                                                             (self.y_train,
                                                              self.X_train, 'train_raw'),
                                                             ]):
                    for j, (graph_type, func) in enumerate([
                        ('', plot_confusion_matrix),
                        ('_PR', plot_precision_recall_curve),
                        ('_ROC', plot_roc_curve),
                    ]):
                        ax = axes[j, i]
                        graph_yarray = yarray.astype(bool)
                        if flipped_labels:
                            graph_yarray = ~graph_yarray
                        disp = func(clf, Xarray, graph_yarray, ax=ax)
                        title = f'{label}{graph_type}{flipped_labels_suffix}'
                        ax.set_title(title)
                        if graph_type in ['_PR', '_ROC']:
                            ax.set_xlim(-0.05, 1.05)
                            ax.set_ylim(-0.05, 1.05)
                            ax.set_aspect('equal', adjustable='box')
                suptitle = f'{graph_prefix}{flipped_labels_suffix}'
                plt.suptitle(suptitle)
                savepath = os.path.join(graph_dir, f'{suptitle}.png')
                fig.savefig(savepath, dpi=100)
                plt.close()

        for i, (yarray, Xarray, label) in enumerate([(self.y_test, self.X_test, 'test'),
                                                     (self.y_train_sampled,
                                                      self.X_train_sampled, 'train'),
                                                     (self.y_train,
                                                      self.X_train, 'train_raw'),
                                                     ]):

            y_pred = clf.predict(Xarray)
            y_prob = clf.predict_proba(Xarray)[:, 1]
            y_true = yarray
            X_shape = Xarray.shape
            metric_list.extend(feat_util.binary_metric_list(
                y_true=y_true, y_pred=y_pred, y_prob=y_prob, X_shape=X_shape,
                label_prefix=f'{label}_'
            ))



        return metric_list

    def model_stats(self, classifier=None, graph=True):
        # counter, auc, and optional graph of given classifer (default: self.cur_model)
        classifier = classifier or self.cur_model
        y_prob = classifier.predict_proba(self.X_test)[:, 1]
        print(f"dimension y_prob: {y_prob.shape}")
        print(f"dimension y_test: {self.y_test.shape}")
        print(f'Predicts:', Counter(list(classifier.predict(self.X_test))))
        print(f'True Labels:', Counter(self.y_test))
        if graph:
            fpr, tpr, thres = roc_curve(self.y_test, y_prob)
            plt.plot(fpr, tpr, color='green')
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')
            plt.show()
        roc_auc = roc_auc_score(self.y_test, y_prob)
        print(f"ROC-AUC Score: {roc_auc}")

    def classification_report(self):
        # classification report on current model
        y_true = self.y_test
        y_pred = self.cur_model.predict(self.X_test)
        print(classification_report(y_true, y_pred))


class JWWindowSelector:

    ycols = ['R0_quiz_response','R1_quiz_response','R2_quiz_response','R1_quiz_response_bin',
            'R1_quiz_response_0v12','R1_quiz_response_01v2','R1_quiz_response_bin_x',
            'R2_quiz_response_bin','R2_quiz_response_bin_x','R2_quiz_response_bin0v123',
            'R2_quiz_response_bin01v23','R2_quiz_response_bin012v3']
    INTERACTION = 0
    LEVEL = 1
    QUIZ = 2
    OBJECTIVE = 3

    def __init__(self, csv_fpath=None, df=None, meta=None):
        assert csv_fpath is not None or df is not None
        # load df
        if df is None:
            print(f'Loading from {csv_fpath}...')
            self.df, self.meta = feat_util.open_csv_from_path_with_meta(
                csv_fpath, index_col=0)
        else:
            self.df = df
            self.meta = meta or []

        self.df_cols = list(df.columns)

    @staticmethod
    def get_abbrev(window_type):
        if window_type == JWWindowSelector.INTERACTION:
            return 'int'
        if window_type == JWWindowSelector.LEVEL:
            return 'lvl'
        if window_type == JWWindowSelector.QUIZ:
            return 'q'
        if window_type == JWWindowSelector.OBJECTIVE:
            return 'obj'

    @staticmethod
    def get_prefix(n, window_type):
        if window_type == JWWindowSelector.INTERACTION:
            return f'int{n}_i'
        if window_type == JWWindowSelector.LEVEL:
            return f'lvl{n}_'
        if window_type == JWWindowSelector.QUIZ:
            return f'Q{n}_'
        if window_type == JWWindowSelector.OBJECTIVE:
            return f'obj{n}_o'

    @staticmethod
    def get_window_range(window_type, skip_Q23=False):
        if window_type == JWWindowSelector.INTERACTION:
            return range(189)
        if window_type == JWWindowSelector.LEVEL:
            return range(24)
        if window_type == JWWindowSelector.QUIZ:
            if not skip_Q23:
                return range(19)
            else:
                return [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
        if window_type == JWWindowSelector.OBJECTIVE:
            return range(80)


    def cols_startwith(self, prefix):
        return [c for c in self.df_cols if c.startswith(prefix)]
    def get_feats(self, n, window_type):
        prefix = self.get_prefix(n, window_type)
        feats = self.cols_startwith(prefix)
        return feats

    def get_filter_queries(self, n, window_type, max_seconds_per_word=2):
        prefix = JWWindowSelector.get_prefix(n, window_type)
        queries = [f"R1_quiz_response == R1_quiz_response"]
        if window_type in [JWWindowSelector.INTERACTION, JWWindowSelector.LEVEL]:
            queries.extend([
                        f"{prefix}first_enc_duration == {prefix}first_enc_duration",
                        f"{prefix}first_enc_duration > 0",
            ])
        if window_type == JWWindowSelector.QUIZ:
            queries.extend([
                f'{prefix}A1_nan!=1'
            ])
        elif window_type == JWWindowSelector.INTERACTION:
            num_words = self.df[f"int{n}_ifirst_enc_words_read"].max()
            queries.extend([
                    f"{prefix}first_enc_words_read == {num_words}",
                    f"{prefix}time_to > 0",
                    f"{prefix}first_enc_duration < {prefix}first_enc_words_read*{max_seconds_per_word}",
                    ])
        elif window_type == JWWindowSelector.OBJECTIVE:
            if n < 79:
                queries.append(f'obj{n}_onext_int_nan==0')
                queries.append(f"obj{n}_otime_to_next_obj < 600")
                queries.append(f"obj{n}_otime_to_next_obj > 0 ")

        elif window_type == JWWindowSelector.LEVEL:
            queries.append(f"{prefix}time_in_level < 1200")
            queries.append(f"{prefix}time_in_level > 0")

        queries.extend([f"R{i}_quiz_response == R{i}_quiz_response" for i in [0,1,2]])
        return queries

    def get_base_meta(self):
        return self.meta
    
    @staticmethod
    def join_XY(X,Y):
        return X.join(Y)
    
    def get_X_Y_meta(self, n, window_type, max_seconds_per_word=2,nbins=0, drop_first_next_int_col = True):
        meta = []
        prefix = JWWindowSelector.get_prefix(n, window_type)
        Xfeats = self.get_feats(n, window_type)
        meta.append(f'Using feats: {Xfeats}')

        if window_type==JWWindowSelector.INTERACTION:
            total_words = self.df[f"int{n}_ifirst_enc_words_read"].max() 
            if total_words is np.nan:
                return None, None, meta
            elif total_words < 10:
                print('Total words < 10!')
        queries = self.get_filter_queries(n, window_type, max_seconds_per_word=max_seconds_per_word)
        filtered_df, filtered_df_meta = feat_util.filter_df(self.df[Xfeats+JWWindowSelector.ycols], query_list=queries, verbose=True, fillna=None)
        meta.extend(filtered_df_meta)
        X = filtered_df[Xfeats].fillna(0).copy()
        meta.append(f'Filled X with 0')
        Y = filtered_df[JWWindowSelector.ycols].copy()
        drop_cols = []
        if window_type in [JWWindowSelector.INTERACTION, JWWindowSelector.LEVEL]:
            drop_cols = [
                f"{prefix}first_enc_boxes_read",
                f"{prefix}first_enc_words_read",
            ]
        if window_type==JWWindowSelector.INTERACTION:
            drop_cols.extend([
            f"{prefix}time_to",
            f"{prefix}total_duration"
            ])
        if window_type==JWWindowSelector.OBJECTIVE:
            drop_cols.append(f"{prefix}next_int_nan")

        # if window_type==JWWindowSelector.QUIZ:
        #     drop_cols.append(f"{prefix}answers")

        X = X.drop(columns=drop_cols)
        meta.append(f"Dropped drop_cols: {drop_cols}")
        constant_cols = X.columns[X.nunique()==1]
        X = X.drop(columns=constant_cols)
        meta.append(f'Dropped constant_cols: {constant_cols}')
        if not len(X):
            return None, None, meta 
        if window_type == JWWindowSelector.OBJECTIVE and drop_first_next_int_col:
            next_int_cols = [c for c in X.columns if 'next_int' in c]
            if next_int_cols:
                X = X.drop(columns=next_int_cols[0])
                meta.append(f'Dropped onehot column {next_int_cols[0]} from {next_int_cols}')

        ## does not bin by default
        if nbins:
            est = KBinsDiscretizer(n_bins=nbins, encode='onehot-dense', strategy='quantile')
            bin_feats = [f'{prefix}first_enc_avg_tbps', 
                        f'{prefix}first_enc_avg_wps',
                        # f'{prefix}first_enc_duration',
                        f'{prefix}first_enc_var_tbps',
                        f'{prefix}first_enc_var_wps']
            bin_feats = [c for c in bin_feats if c in X.columns]
            if bin_feats:
                    
                Xt = est.fit_transform(X[bin_feats])
                new_feat_names = [f'{feat}>{x:.2f}' for bins,feat in zip(est.bin_edges_,bin_feats) for x in list(bins)[:-1]]

                Xt_df = pd.DataFrame(Xt, index=X.index, columns=new_feat_names)
                X = X.join(Xt_df)
                X = X.drop(columns=bin_feats)
                meta.append(f'Quantized n_bins={nbins} feats {bin_feats} to {new_feat_names}')

        return (X, Y, meta)

    def get_X_Y_meta_range(self, ns, window_type, max_seconds_per_word=2,nbins=0, drop_first_next_int_col = True, verbose=True):
        X, Y, meta = None, None, []
        for n in ns:
            tX, tY, tmeta = self.get_X_Y_meta(n, window_type, max_seconds_per_word=max_seconds_per_word, nbins=nbins, drop_first_next_int_col=drop_first_next_int_col)
            X, Y, meta = JWWindowSelector.join_X_Y_meta(X, Y, meta, tX, tY, tmeta, copy=False)
            print('Join Size:', X.shape)
        X, Y = X.copy(), Y.copy()
        return X, Y, meta

    @staticmethod
    def join_X_Y_meta(X1, Y1, meta1, X2, Y2, meta2, copy=True):
        meta = meta1+meta2
        if X1 is None:
            X = X2
            Y = Y2
        elif X2 is None:
            X = X1
            Y = Y1
        else:
            X = X1.join(X2, how='inner')
            Y = Y1.loc[X.index, :]
            meta = meta1+['--Inner Join--']+meta2+[f'Resultant Join Shape: {X.shape}']
        if copy and X is not None:
            X, Y = X.copy(), Y.copy()
        return X, Y, meta
        
                

