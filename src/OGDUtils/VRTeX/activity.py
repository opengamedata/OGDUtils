import json
import sys
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import numpy as np
import numpy.matlib as npm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
print(sys.version)



def dot_pdt(dfs):
  for i in range(len(dfs)):
    
    shifted_lst = dfs[i]['3d_normalized'].shift(-1).tolist()
    origin = dfs[i]['3d_normalized'].values.tolist()
    shifted_lst[-1] = np.array([0,0,0])
    dot_lst = []
    for j in range(len(origin)):
        dot_lst.append(np.dot(origin[j], shifted_lst[j]))
    dfs[i]['dot_product'] = dot_lst
  return dfs

from scipy.spatial.distance import pdist
#added assortment by timestamp for later graph purpose


def pairwise_distance(dfs):
  for i in range(len(dfs)):
    shifted_lst = dfs[i]['position'].shift(-1).tolist()
    origin = dfs[i]['position'].values.tolist()
    shifted_lst[-1] = np.array([0,0,0])
    dot_lst = []
    for j in range(len(origin)):
        positions = np.vstack([origin[j], shifted_lst[j]])
        distances = pdist(positions)
        dot_lst.append(distances.item())
    dfs[i]['distance_product'] = dot_lst


  return dfs


def fst_n_min(dfs, n_min=1):
  ret_val = dfs.copy()
  for i in range(len(dfs)):
    ret_val[i] = dfs[i][dfs[i]['timesincelaunch']<=n_min*60]
  return ret_val
  
view_1st_5_min = fst_n_min(df_pos_rot, n_min=5)
print(view_1st_5_min)

def create_player_plot(player_number,event_data):

    #view_1st_5_min = fst_n_min(df_pos_rot, n_min=5)
    #right_1st_5_min = fst_n_min(right_pos_pairwise, n_min=5)
    #left_1st_5_min = fst_n_min(left_pos_pairwise, n_min=5)
    
    # create a 4x5 grid of subplots

    fig = make_subplots(rows=3, cols=1, shared_yaxes=True, shared_xaxes=True, subplot_titles=('View Data', 'Left Hand', 'Right Hand'))
    
    fig.add_trace(go.Scatter(x=df_pos_rot[player_number]['timesincelaunch'][:-1], y=1-df_pos_rot[player_number]['dot_product'][:-1], mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=left_pos_pairwise[player_number]['timesincelaunch'][:-1], y=left_pos_pairwise[player_number]['distance_product'][:-1], mode='lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=right_pos_pairwise[player_number]['timesincelaunch'][:-1], y=right_pos_pairwise[player_number]['distance_product'][:-1], mode='lines'), row=3, col=1)
    
    fig.update_layout(title=f'View Vector and Hand Movement for Player {player_number}', showlegend=False)
    
    player_id = gaze_package_lst_splt[player_number]["player_id"].iloc[0]
    other_data_for_player = event_data[event_data['player_id'] == player_id]
    event_times_for_player = other_data_for_player['timesincelaunch'].tolist()
    print(f"percent waddle happens overall: {len(event_times_for_player)/len(gaze_package_lst_splt[player_number]['player_id'])*100}")
    if event_times_for_player is not []:
        for event_time in event_times_for_player:
            fig.add_vline(x=event_time, line_width=2, line_dash="dash", line_color="yellow")
    
    fig.show()


def quaternion_to_rotation_matrix(q):

    #maybe transform into a 4*4 matrixs
    
    """
    Convert a quaternion into a rotation matrix.
    
    Parameters:
    q (tuple or list): The quaternion in the format (w, x, y, z).
    
    Returns:
    numpy.ndarray: The 3x3 rotation matrix.
    """
    
    w, x, y, z = q
    
    # Compute the product of the quaternion with its conjugate
    norm_sq = w**2 + x**2 + y**2 + z**2
    
    # Compute the rotation matrix elements
    r00 = 1.0 - 2.0*(y**2 + z**2) / norm_sq
    r01 = 2.0*(x*y - w*z) / norm_sq
    r02 = 2.0*(x*z + w*y) / norm_sq
    r10 = 2.0*(x*y + w*z) / norm_sq
    r11 = 1.0 - 2.0*(x**2 + z**2) / norm_sq
    r12 = 2.0*(y*z - w*x) / norm_sq
    r20 = 2.0*(x*z - w*y) / norm_sq
    r21 = 2.0*(y*z + w*x) / norm_sq
    r22 = 1.0 - 2.0*(x**2 + y**2) / norm_sq
    
    # Formulate the rotation matrix
    R = np.array([[r00, r01, r02],
                  [r10, r11, r12],
                  [r20, r21, r22]])
    
    return R


# see only x factor of roatition (),with waddle as well - from event data  
# how to get individual rotation component from quaternion - get x component 
player_number = 2
event_data = waddle_lst

fig = make_subplots(rows=1, cols=1, shared_yaxes=True, shared_xaxes=True, subplot_titles=('View Data', 'Left Hand', 'Right Hand'))



fig.add_trace(go.Scatter(x=df_pos_rot[player_number]['timesincelaunch'][:-1], y=df_pos_rot[player_number]["rotation_x"][:-1], mode='lines'), row=1, col=1)

fig.update_layout(title=f'View Vector and Hand Movement for Player {player_number}', showlegend=False)
    
player_id = gaze_package_lst_splt[player_number]["player_id"].iloc[0]
other_data_for_player = event_data[event_data['player_id'] == player_id]
event_times_for_player = other_data_for_player['timesincelaunch'].tolist()

if event_times_for_player is not []:
    for event_time in event_times_for_player:
        fig.add_vline(x=event_time, line_width=2, line_dash="dash", line_color="yellow")
    
fig.show()