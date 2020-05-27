from collections import namedtuple

class FeatureSetOptions:
    feature_set_options = namedtuple('FeatureSetOptions', ['game','name','filter_args','new_feat_args','lvlfeats','lvlrange','finalfeats','zthresh','finalfeats_readable'])
    jowilder_demo_set = feature_set_options('jowilder',
                              'demo',
                              {'query_list':['sess_avg_num_tiles_hovered_before_placing_home > 1'],
                               'verbose':False, 
                               'fillna':0},
                               {'avg_tile_hover_lvl_range': range(0,1)},
                              ['count_buy_home', 'count_buy_farm','count_buy_livestock','count_buys'],
                              range(0, 1),
                              ['weighted_avg_lvl_0_to_0_avg_num_tiles_hovered_before_placing_farm',
                                'sum_lvl_0_to_0_count_buy_home',
                              'sum_lvl_0_to_0_count_buy_farm',
                              'sum_lvl_0_to_0_count_buy_livestock',
                              'sum_lvl_0_to_0_count_buys'],
                              3,
                              ['hovers\nbefore\nfarm','home','farm','livestock','buys']
    )