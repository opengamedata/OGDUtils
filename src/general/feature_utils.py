"""
Note: Utils in this file are 

Usage:
If the first cell ran correctly, changing the CWD to the Jupyter file and adding '.' to sys path, then.
import Notebooks.Clustering.cluster_utils as cu

Otherwise:
from google.colab import drive
drive.mount('/content/drive')

import os
FIELDDAY_DIR = '/content/drive/My Drive/Field Day' # the field day directory on the mounted drive
JUPYTER_DIR = os.path.join(FIELDDAY_DIR,'Research and Writing Projects/2020 Lakeland EDM/Jupyter')
os.chdir(JUPYTER_DIR)

import sys
sys.path.append('.')
import Notebooks.Clustering.cluster_utils as cu

"""
import copy
import os
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import ipywidgets as widgets
from matplotlib import pyplot as plt
from scipy import stats
from typing import Iterable, Optional, List, Tuple, Union
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from datetime import datetime
import pickle
from collections import Counter



def print_options(meta):
    """
    Takes in meta text and outputs text for an options group.
    :param meta: meta text. Expected format will be like:

        Metadata:
        Import from fhttps://opengamedata.fielddaylab.wisc.edu/data/LAKELAND/LAKELAND_20191201_to_20191231_de09c18_proc.zip
        Import from fData/Raw Log Data/LAKELAND_20200101_to_20200131_a9720c1_proc.zip
        *arg* filter_args = {'query_list': ['debug == 0', 'sess_ActiveEventCount >= 10', 'sessDuration >= 300', '_continue == 0'], 'one_query': False, 'fillna': 0, 'verbose': True}
        Query: Intial Shape, output_shape: (32227, 1647)
        Query: debug == 0, output_shape: (32221, 1647)
        Query: sess_ActiveEventCount >= 10, output_shape: (26934, 1647)
        Query: sessDuration >= 300, output_shape: (16109, 1647)
        Query: _continue == 0, output_shape: (10591, 1647)
        Filled NaN with 0
        *arg* new_feat_args = {'verbose': False, 'avg_tile_hover_lvl_range': None}
        *arg* lvlfeats = ['count_blooms', 'count_deaths', 'count_farmfails', 'count_food_produced', 'count_milk_produced']
        *arg* lvlrange = range(0, 2)
        Describe Level Feats lvls 0 to 1. Assuming WINDOW_SIZE_SECONDS=300 and WINDOW_OVERLAP_SECONDS=30, filtered by (sessDuration > 570)
        *arg* finalfeats = ['avg_lvl_0_to_1_count_deaths', 'avg_lvl_0_to_1_count_farmfails', 'avg_lvl_0_to_1_count_food_produced', 'avg_lvl_0_to_1_count_milk_produced']
        Original Num Rows: 6712
        *arg* zthresh = 3
        Removed points with abs(ZScore) >= 3. Reduced num rows: 6497

    where all args are denoted by an initial *arg* and values are after =.


    """
    if type(meta) == str:
        meta = meta.split('\n')
    inner = ',\n\t'.join(["'GAME'", "'NAME'"] + [l[6:].split(' = ')[1]
                                                 for l in meta if l.startswith('*arg*')] + ['[]'])
    print(f'options({inner}\n)')


# split out query creation per-game
def filter_df(df: pd.DataFrame, query_list: List[str], one_query: bool = False, fillna: object = 0, verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """

    :param df: dataframe to filter
    :param query_list: list of queries for filter
    :param one_query: bool to do the query (faster) as one query or seperate queries (slower, gives more info)
    :param fillna: value to fill NaNs with
    :param verbose: whether to input information
    :return: (df, List[str])
    """
    df = df.rename({'continue': '_continue'}, axis=1)
    filter_args = locals()
    filter_args.pop('df')
    filter_meta = [f'*arg* filter_args = {filter_args}']

    def append_meta_str(q, shape):
        outstr = f'Query: {q}, output_shape: {shape}'
        filter_meta.append(outstr)
        if verbose:
            print(outstr)

    append_meta_str('Intial Shape', df.shape)

    if not one_query:
        for q in query_list:
            df = df.query(q)
            append_meta_str(q, df.shape)
    else:  # do the whole query at once
        full_query = ' & '.join([f"({q})" for q in query_list])
        print('full_query:', full_query)
        df = df.query(full_query)
        append_meta_str(full_query, df.shape)

    if fillna is not None:
        df = df.fillna(fillna)
        filter_meta.append(f'Filled NaN with {fillna}')
    return df.rename({'_continue': 'continue'}), filter_meta


def create_new_base_features(df, verbose=False):
    """

    Currently a stub. Used to create new features from existing ones. See create_new_base_features_lakeland for example.
    :param df:
    :param verbose:
    :return:
    """
    new_base_feature_args = locals()
    new_base_feature_args.pop('df')
    new_feat_meta = [f'*arg* new_feat_args = {new_base_feature_args}']

    return df, new_feat_meta

def describe_lvl_feats(df, fbase_list, lvl_range):
    """
    Calculates sum/avg of given level base features (fnames without lvlN_ prefix) in the level range.
    May have a bug.

    :rtype: (df, List[str]) where the new df includes sum_ and avg_lvl_A_to_B
    :param df: dataframe to pull from and append to
    :param fbase_list: list of feature bases (fnames without lvlN_ prefix)
    :param lvl_range: range of levels to choose. typically range(min_level, max_level+1)
    """
    metadata = []
    metadata.append(f'*arg* lvlfeats = {fbase_list}')
    metadata.append(f'*arg* lvlrange = {lvl_range}')
    if not fbase_list:
        return df, metadata

    # TODO: Add filter for levels we don't want, like the one from lakeland
    # query = f'sessDuration > {(level_time - level_overlap) * (lvl_end) + level_time}'
    # df = df.query(query)
    # metadata.append(
    #     f'Describe Level Feats lvls {lvl_start} to {lvl_end}. Assuming WINDOW_SIZE_SECONDS={level_time} and WINDOW_OVERLAP_SECONDS={level_overlap}, filtered by ({query})')

    fromlvl, tolvl = lvl_range[0], lvl_range[-1]
    sum_prefix = f'sum_lvl_{fromlvl}_to_{tolvl}_'
    avg_prefix = f'avg_lvl_{fromlvl}_to_{tolvl}_'
    for fn in fbase_list:
        tdf = df[[f'lvl{i}_{fn}' for i in lvl_range]].fillna(0)
        df[sum_prefix + fn] = tdf.sum(axis=1)
        df[avg_prefix + fn] = tdf.mean(axis=1)
    return df, metadata


def describe_range_feats(df, range_feats_and_range, cc_prefix_max_list):
    """
    Calculates sum/avg of given level base features (fnames without lvlN_ prefix) in the level range.
    May have a bug.

    :rtype: (df, List[str]) where the new df includes sum_ and avg_lvl_A_to_B
    :param df: dataframe to pull from and append to
    :param fbase_list: list of feature bases (fnames without lvlN_ prefix)
    :param lvl_range: range of levels to choose. typically range(min_level, max_level+1)
    """
    metadata = []
    metadata.append(f'*arg* range_feats_and_range = {range_feats_and_range}')
    metadata.append(f'*arg* cc_prefix_max_list = {cc_prefix_max_list}')
    if not range_feats_and_range:
        return df, metadata

    # TODO: Add filter for levels we don't want, like the one from lakeland
    # query = f'sessDuration > {(level_time - level_overlap) * (lvl_end) + level_time}'
    # df = df.query(query)
    # metadata.append(
    #     f'Describe Level Feats lvls {lvl_start} to {lvl_end}. Assuming WINDOW_SIZE_SECONDS={level_time} and WINDOW_OVERLAP_SECONDS={level_overlap}, filtered by ({query})')

    range_prefix_max_list = [('lvl', None)]+cc_prefix_max_list
    for i in range(len(range_feats_and_range)):
        range_feats, rang = range_feats_and_range[i]
        prefix, _ = range_prefix_max_list[i]
        fromval, toval = rang[0], rang[-1]
        sum_prefix = f'sum_{prefix}_{fromval}_to_{toval}_'
        avg_prefix = f'avg_{prefix}_{fromval}_to_{toval}_'
        for fn in range_feats:
            tdf = df[[f'{prefix}{i}_{fn}' for i in rang]].fillna(0)
            df[sum_prefix + fn] = tdf.sum(axis=1)
            df[avg_prefix + fn] = tdf.mean(axis=1)
    return df, metadata


def get_feat_selection(df, session_prefix, max_lvl, cc_prefix_max_list=None):
    """
    Gets the feature selection widget.
    :param df:
    :param max_lvl:
    :return:
    """

    cc_prefix_max_list = cc_prefix_max_list or []
    checkbox_widgets = []
    slider_widgets = []
    feats = set()

    for prefix, max_val in [('lvl', max_lvl)] + cc_prefix_max_list:
        start_val = widgets.IntSlider(value=0, min=0, max=max_val, step=1, description=f'Start {prefix}:',
                                      disabled=False, continuous_update=False, orientation='horizontal', readout=True,
                                      readout_format='d')
        end_val = widgets.IntSlider(value=0, min=0, max=max_val, step=1, description=f'End {prefix}:',
                                    disabled=False, continuous_update=False, orientation='horizontal', readout=True,
                                    readout_format='d')
        val_selection = widgets.GridBox([start_val, end_val])
        slider_widgets.append(val_selection)
        val_feats_set = set(['_'.join(f.split('_')[1:])
                             for f in df.columns if f.startswith(prefix)])
        feats = feats.union(
            [f'{prefix}{n}_{v}' for n in range(max_val+1) for v in val_feats_set])
        val_feats = sorted(val_feats_set)
        val_feats_checkbox = multi_checkbox_widget(val_feats, prefix)
        checkbox_widgets.append(val_feats_checkbox)

    other_feats = sorted(set(df.columns).difference(feats))
    selection_widget = widgets.GridBox(checkbox_widgets+slider_widgets+[multi_checkbox_widget(other_feats, 'other')],
                                       layout=widgets.Layout(grid_template_columns=f"repeat({len(slider_widgets)}, 500px)"))

    return selection_widget


def get_selected_feature_list(selection_widget, session_prefix, cc_prefix_max_list=None):
    """

    :param selection_widget:
    :return: list of features selected
    """
    cc_prefix_max_list = cc_prefix_max_list or []
    prefix_list = ['lvl']+[prefix_max[0] for prefix_max in cc_prefix_max_list]
    other_feats = [
        s.description for s in selection_widget.children[-1].children[1].children if s.value]
    range_feats_and_range = get_range_feats_and_range(selection_widget)
    range_feats_list = []
    for i in range(len(range_feats_and_range)):
        prefix = prefix_list[i]
        feats = range_feats_and_range[i][0]
        rang = range_feats_and_range[i][1]
        range_feats_list += [f'{prefix}{n}_{f}' for f in feats for n in rang]
    return range_feats_list + other_feats


def get_range_feats_and_range(selection_widget) -> Union[List[str], Iterable]:
    """

    :param selection_widget:
    :return: List of fbases from selection_widget and level range
    """
    ret = []
    widgets = selection_widget.children
    assert len(widgets) % 2
    num_range_groups = (len(widgets)-1)//2
    for i in range(num_range_groups):
        checkbox = widgets[i]
        slider = widgets[i+num_range_groups]
        start_widget = slider.children[0]
        end_widget = slider.children[1]
        feat_list = [
            s.description for s in checkbox.children[1].children if s.value]
        val_range = range(start_widget.value, end_widget.value + 1)
        ret.append((feat_list, val_range))

    return ret


def multi_checkbox_widget(descriptions, category):
    """ Widget with a search field and lots of checkboxes """
    search_widget = widgets.Text(
        layout={'width': '400px'}, description=f'Search {category}:')
    options_dict = {description: widgets.Checkbox(description=description, value=False,
                                                  layout={'overflow-x': 'scroll', 'width': '400px'}, indent=False) for
                    description in descriptions}
    options = [options_dict[description] for description in descriptions]
    options_widget = widgets.VBox(
        options, layout={'overflow': 'scroll', 'height': '400px'})
    multi_select = widgets.VBox([search_widget, options_widget])

    # Wire the search field to the checkboxes
    def on_text_change(change):
        search_input = change['new']
        if search_input == '':
            # Reset search field
            for d in descriptions:
                options_dict[d].layout.visibility = 'visible'
                options_dict[d].layout.height = 'auto'
        elif search_input[-1] == '$':
            search_input = search_input[:-1]
            # Filter by search field using difflib.
            for d in descriptions:
                if search_input in d:
                    options_dict[d].layout.visibility = 'visible'
                    options_dict[d].layout.height = 'auto'
                else:
                    options_dict[d].layout.visibility = 'hidden'
                    options_dict[d].layout.height = '0px'
            # close_matches = [d for d in descriptions if search_input in d] #difflib.get_close_matches(search_input, descriptions, cutoff=0.0)
            # new_options = [options_dict[description] for description in close_matches]
        # options_widget.children = new_options

    search_widget.observe(on_text_change, names='value')
    return multi_select


def reduce_feats(df, featlist):
    """
    Takes in a df and outputs only the given columns in featlist
    :param df:
    :param featlist:
    :return:
    """
    return df[featlist].copy(), [f'*arg* finalfeats = {featlist}']


def reduce_outliers(df, z_thresh, show_graphs=True, outpath=None):
    """
    Takes in df and z_thresh, shows box plots, and outputs graph with points of zscore>z_thresh removed.
    Does not always work properly. Does not seem to tolerate NaNs.
    TODO: fix.
    :param df:
    :param z_thresh:
    :param show_graphs:
    :return:
    """
    meta = []
    meta.append(f"Original Num Rows: {len(df)}")
    meta.append(f"*arg* zthresh = {z_thresh}")
    title = f'Raw Boxplot Original Data n={len(df)}'
    df.plot(kind='box', title=title, figsize=(20, 5))
    if outpath:
        savepath = os.path.join(outpath, f'Raw Boxplot Original.png')
        plt.savefig(savepath)
    plt.close()

    if z_thresh is None:
        return df, meta

    z = np.abs(stats.zscore(df))
    no_outlier_df = df[(z < z_thresh).all(axis=1)]
    meta.append(
        f'Removed points with abs(ZScore) >= {z_thresh}. Reduced num rows: {len(no_outlier_df)}')
    title = f'Raw Boxplot ZThresh={z_thresh} n={len(no_outlier_df)}'
    no_outlier_df.plot(kind='box', title=title, figsize=(20, 5))
    if outpath:
        savepath = os.path.join(outpath, f'Raw Boxplot Zthresh Removed.png')
        plt.savefig(savepath)
    plt.close()
    return no_outlier_df, meta


jw_cc_max = [('obj', 80), ('int', 188), ('Q', 18)]


def full_filter(df, import_meta, options, outpath) -> Tuple[pd.DataFrame, List[str]]:
    """
    Takes in a df, metadata, and options group.
    Outputs the filtered df and the meta.
    :param get_df_func:
    :param options:
    :return:
    """
    # df, import_meta = get_df_func()
    filtered_df, filter_meta = filter_df(df, **options.filter_args)
    game = options.game.upper()
    # if game == 'LAKELAND':
    #     new_feat_df, new_feat_meta = create_new_base_features_lakeland(filtered_df, **options.new_feat_args)
    #     aggregate_df, aggregate_meta = describe_lvl_feats_lakeland(new_feat_df, options.lvlfeats, options.lvlrange)
    # elif game == 'CRYSTAL':
    #     new_feat_df, new_feat_meta = create_new_base_features_crystal(filtered_df, **options.new_feat_args)
    #     aggregate_df, aggregate_meta = describe_lvl_feats_crystal(new_feat_df, options.lvlfeats, options.lvlrange)
    # elif game == 'WAVES':
    #     new_feat_df, new_feat_meta = create_new_base_features_waves(filtered_df, **options.new_feat_args)
    #     aggregate_df, aggregate_meta = describe_lvl_feats_waves(new_feat_df, options.lvlfeats, options.lvlrange)
    # else:
    #     assert False
    new_feat_df, new_feat_meta = create_new_base_features(
        filtered_df, **options.new_feat_args)
    aggregate_df, aggregate_meta = describe_lvl_feats(
        new_feat_df, options.lvlfeats, options.lvlrange)
    reduced_df, reduced_meta = reduce_feats(aggregate_df, options.finalfeats)
    # hack while NaNs are popping up in aggregate df or newfeatdf TODO: Fix this. It never used to be an issue.
    reduced_df = reduced_df.fillna(0)
    final_df, outlier_meta = reduce_outliers(
        reduced_df, options.zthresh, outpath=outpath)
    final_meta = import_meta + filter_meta + new_feat_meta + \
        aggregate_meta + reduced_meta + outlier_meta
    return final_df, final_meta


def save_csv_and_meta(df, meta_list, save_dir, csv_name, meta_name=None, permissions='w+', add_columns=True):
    if csv_name.endswith(('.tsv', '.csv')):
        extension = csv_name[-4:]
        csv_name = csv_name[:-4]
    else:
        extension = '.csv'
    separator = '\t' if extension == '.tsv' else ','

    # hardcopy
    meta_list = [x for x in meta_list]
    meta_list.append(f'OUTPUT_SHAPE: {df.shape}')
    meta_list.append(f'OUTPUT_FILE: {csv_name}')
    meta_list.append(f'CSV OUTPUT_DATE: {datetime.now()}')
    if add_columns:
        meta_list.append(f'OUTPUT_COLUMNS: {sorted(list(df.columns))}')

    meta_name = meta_name or csv_name + '_meta.txt'
    meta_text = save_meta(meta_list, save_dir, meta_name, permissions=permissions)

    with open(os.path.join(save_dir, csv_name)+extension, permissions) as f:
        for l in meta_text.splitlines():
            f.write(f'# {l}\n')
        f.write('\n')
        df.to_csv(f, sep=separator)

    return None, []

def save_meta(meta_list, save_dir, meta_name, permissions='w+'):
    meta_text = 'Metadata:\n'+'\n'.join(meta_list+[
        f'META OUTPUT_DATE: {datetime.now()}'
    ])
    with open(os.path.join(save_dir, meta_name), permissions) as f:
        f.write(meta_text)
    return meta_text


def open_csv_from_path_with_meta(csv_fpath, index_col=0):
    metadata = []
    with open(csv_fpath) as f:
        for line in f.readlines():
            if line.startswith('#'):
                metadata.append(line[2:].strip())
            else:
                break
    df = pd.read_csv(csv_fpath, comment='#', index_col=index_col)
    return df, metadata

def remove_nan_labels(X, y):
    nonnull_indices = ~y.isna()
    ret_X = X.loc[nonnull_indices, :].copy()
    ret_y = y.loc[nonnull_indices].copy()
    return ret_X, ret_y

def get_PSPC_pipeline(classifier, preprocessor=None, sampler=None):
    """
Returns a Preporcessor Sampler Preprocessor Classifier pipeline
imblearn.pipeline.make_pipeline(preprocessor, sampler, copy.deepcopy(preprocessor),classifier)
    :param classifier: sklearn compatible classifier
    :param preprocessor: sklearn compatible classifier (Make sure it won't change the data if applied iteratively!)
    :param sampler: sklearn compatible sampler
    :return:
    """
    clf = imblearn.pipeline.make_pipeline(preprocessor, sampler, copy.deepcopy(preprocessor),classifier)
    return clf




def save_model(savedir, savename, model, X_test, y_test, meta_list=None):
    name, ext = os.path.splitext(savename)
    if meta_list:
        meta_list = [x for x in meta_list]
        meta_list.append(f'MODEL_USED: {model}')
        meta_list.append(f'TEST_SHAPE: {X_test.shape}')
        meta_list.append(f'MODEL_Ytest_Xtest_SAVEPATH: {savename}')
        meta_list.append(f'OUTPUT_FILE: {savename}')
        meta_list.append(f'MODEL_OUTPUT_DATE: {datetime.now()}')
        meta_list.append(f'TEST_COLUMNS: {sorted(list(X_test.columns))}')
        meta_name = name + '_meta.txt'
        save_meta(meta_list, savedir, meta_name, permissions='w+')
    with open(os.path.join(savedir, savename),'wb+') as f:
        pickle.dump((model, y_test, X_test), f)

def load_model(loadpath):
    ## Do NOT load ANYTHING that you don't 100% trust! Malicious contents could easily harm your computer.
    with open(loadpath, 'rb') as f:
        model, y_test, X_test = pickle.load(f)
    return model, X_test, y_test, 

def corr_heatmap(df,figsize=(20,20),max_corr=.3, max_rows=3000):
    corr = fast_corr(df, max_rows)
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=max_corr, vmin=-1*max_corr, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
def fast_corr(df, max_rows):
    num_rows = min(max_rows, len(df))
    corr_matrix = df.iloc[:num_rows,:].corr()
    return corr_matrix

def get_high_corr_columns(df, thresh=.95,max_rows=3000):
    corr_matrix = fast_corr(df,max_rows=max_rows).abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find index of feature columns with correlation greater or equal to thresh
    to_drop = [column for column in upper.columns if any(upper[column] >= thresh)]
    return to_drop

def fb(prec, recall, beta=1):
    if prec == 0 or recall == 0:
        return 0
    numerator = prec*recall
    denominator = prec*beta*beta + recall
    return (1+beta*beta)*numerator/denominator

def f1(prec, recall):
    return fb(prec, recall, beta=1)

def f2(prec, recall):
    return fb(prec, recall, beta=2)

def binary_metric_list(y_true, y_pred, y_prob, X_shape=None, label_prefix='', majority_class=1):
    metric_list = []

    baseline_pred = [majority_class]*len(y_true)

    auc = roc_auc_score(y_true=y_true, y_score=y_prob)
    metric_list.append((auc, f'{label_prefix}AUC'))
    f1macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    metric_list.append((f1macro, f'{label_prefix}f1_avg'))
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    metric_list.append((acc, f'{label_prefix}acc'))
    baseline_acc = accuracy_score(y_true=y_true, y_pred=baseline_pred)
    metric_list.append((acc, f'{label_prefix}acc'))
    metric_list.append((baseline_acc, f'{label_prefix}baseline_acc'))
    baseline_auc = roc_auc_score(y_true=y_true, y_score=baseline_pred)
    metric_list.append((baseline_auc, f'{label_prefix}baseline_auc'))
    dAcc = acc - baseline_acc
    dAuc = auc - baseline_auc
    metric_list.append((dAuc, f'{label_prefix}dAuc'))
    metric_list.append((dAcc, f'{label_prefix}dAcc'))


    counter = Counter(y_true)
    size_0s, size_1s = counter[0], counter[1]
    if X_shape:
        num_rows, num_cols = X_shape
        assert (size_0s + size_1s) == num_rows
        metric_list.append((num_rows, f'{label_prefix}total_size'))
        metric_list.append((num_cols, f'{label_prefix}num_feats'))


    metric_list.append((size_0s, f'{label_prefix}size_0s'))
    metric_list.append((size_1s, f'{label_prefix}size_1s'))

    confusion_mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_mat.ravel()        
    precision_1 = tp/(tp+fp)
    b_precision_1 = size_1s/(size_1s+size_0s)
    precision_0 = tn/(tn+fn)
    b_precision_0 = size_0s/(size_1s+size_0s)
    recall_1 = tp/(tp+fn)
    recall_0 = tn/(tn+fp)
    b_recall = 1
    f1_1 = f1(precision_1, recall_1)
    b_f1_1 = f1(b_precision_1, b_recall)
    f1_0 = f1(precision_0, recall_0)
    b_f1_0 = f1(b_precision_0, b_recall)
    f2_1 = f2(precision_1, recall_1)
    b_f2_1 = f2(b_precision_1, b_recall)
    f2_0 = f2(precision_0, recall_0)
    b_f2_0 = f2(b_precision_0, b_recall)
    f1_avg = (f1_1+f1_0)/2

    metric_list.extend([
        (tp, f'{label_prefix}tp'),
        (fp, f'{label_prefix}fp'),
        (tn, f'{label_prefix}tn'),
        (fn, f'{label_prefix}fn'),
        (precision_1, f'{label_prefix}prec_1'),
        (precision_0, f'{label_prefix}prec_0'),
        (recall_1, f'{label_prefix}recall_1'),
        (recall_0, f'{label_prefix}recall_0'),
        (f1_1, f'{label_prefix}f1_1'),
        (f1_0, f'{label_prefix}f1_0'),
        (f2_1, f'{label_prefix}f2_1'),
        (f2_0, f'{label_prefix}f2_0'),

        (b_precision_1, f'{label_prefix}prec_1_baseline'),
        (b_precision_0, f'{label_prefix}prec_0_baseline'),
        (b_f1_1, f'{label_prefix}f1_1_baseline'),
        (b_f1_0, f'{label_prefix}f1_0_baseline'),
        (b_f2_1, f'{label_prefix}f2_1_baseline'),
        (b_f2_0, f'{label_prefix}f2_0_baseline'),

        (precision_1 - b_precision_1, f'{label_prefix}dPrec_1'),
        (precision_0 - b_precision_0, f'{label_prefix}dPrec_0'),
        (f1_1 - b_f1_1, f'{label_prefix}dF1_1'),
        (f1_0 - b_f1_0, f'{label_prefix}dF1_0'),
        (f2_1 - b_f2_1, f'{label_prefix}dF2_1'),
        (f2_0 - b_f2_0, f'{label_prefix}dF2_0'),
    ])

    return metric_list


if __name__ == '__main__':
    pass
