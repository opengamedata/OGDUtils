from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


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
    plt.suptitle('')
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
