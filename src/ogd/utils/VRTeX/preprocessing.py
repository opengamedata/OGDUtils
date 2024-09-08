
import json
import sys
from typing import List

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import numpy as np
import numpy.matlib as npm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def GetStandardColumns(df_of_events:pd.DataFrame, date_format="mixed") -> pd.DataFrame:
    """
    Given a dataframe of OGD event data, returns a new dataframe with only the columns useful for per-event analysis.
    Specifically, these are columns for session ID, timestamp, event name, event data, game state, and time since launch

    :param df_of_events: _description_
    :type df_of_events: pd.DataFrame
    :return: _description_
    :rtype: pd.DataFrame
    """
    df_of_events=df_of_events.sort_values(["session_id","index"])
    df_of_events.loc[:,"game_state"]=df_of_events["game_state"].apply(json.loads)
    df_of_events.loc[:,"event_data"]=df_of_events["event_data"].apply(json.loads)
    #get timestamp from game_state
    df_of_events['timesincelaunch'] = df_of_events['game_state'].apply(lambda x: x.get('seconds_from_launch', 0))
    df_copy = df_of_events[['session_id','timestamp','event_name','event_data','game_state',"timesincelaunch"]]
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'],format=date_format,yearfirst=True)
    return df_copy

def ExtractTelemetryRows(df_of_events:pd.DataFrame, target_event_name:str, package_name) -> pd.DataFrame:
    """
    Given a dataframe of OGD event data, extracts packaged per-frame data from the event data column, creating new rows from each frame in the packages.

    :param df_of_events: _description_
    :type df_of_events: pd.DataFrame
    :param target_event_name: _description_
    :type target_event_name: str
    :param package_name: _description_
    :type package_name: _type_
    :return: _description_
    :rtype: pd.DataFrame
    """
    # Keep event_name with 'data_name' & headset_on
    package_lst = df_of_events.copy()
    
    # Use the isin function to filter rows where event_name is either data_name or 'headset_on'
    package_lst = package_lst[package_lst['event_name'].isin([target_event_name, 'headset_on'])].copy()
    
    # Create a new column 'headset_on_counter' that increments whenever 'headset_on' event is encountered
    package_lst['headset_on_counter'] = np.where(package_lst['event_name'] == 'headset_on', 1, 0)
    package_lst['headset_on_counter'] = package_lst['headset_on_counter'].cumsum()
    package_lst = package_lst[package_lst['event_name'] == target_event_name]

    # Unpack value from event_data
    
    package_lst['event_data'] = package_lst['event_data'].apply(lambda x: json.loads(x[package_name]))

    package_lst['position'] = package_lst['event_data'].apply(lambda x: [item['pos'] for item in x])
    package_lst['rotation'] = package_lst['event_data'].apply(lambda x: [item['rot'] for item in x])

    # Compute the difference in 'timesincelaunch' for each session
    package_lst['timesincelaunch_diff'] = package_lst.groupby('session_id')['timesincelaunch'].diff().fillna(package_lst['timesincelaunch'])
    package_lst['timesincelaunch_initial'] = (package_lst['timesincelaunch'] - package_lst['timesincelaunch_diff']).fillna(0)
    
    # Calculate number of items in the 'event_data' package for each row
    package_lst['num_items'] = package_lst['event_data'].apply(len)
    package_lst['timesincelaunch_diff_split'] = package_lst['timesincelaunch_diff'] / package_lst['num_items']
    #print(package_lst[['timesincelaunch_diff',"timesincelaunch","timesincelaunch_initial","timesincelaunch_diff_split"]].head())
    # Create a list from 0 to 'num_items' for each row
    #Using the apply function to get the index numbers of occurrences of 'pos' in each row
    index_numbers = package_lst['event_data'].apply(lambda x: [i for i, item in enumerate(x) if 'pos' in item])
    #print(package_lst[['timesincelaunch_diff',"timesincelaunch","timesincelaunch_initial","timesincelaunch_diff_split"]].head(n=30))
    
    # Using explode to create a new row for each element in the lists
    exploded_index_numbers = index_numbers.explode()
    exploded_index_numbers.reset_index(drop=True, inplace=True)
    #print(package_lst[['timesincelaunch_diff',"timesincelaunch","timesincelaunch_initial","timesincelaunch_diff_split"]].head(n=5))
    
    # Flatten the 'pos' & 'rot' list and create new rows
    package_lst = package_lst.explode(['position','rotation'])   
    package_lst.reset_index(drop=True, inplace=True)

    package_lst["sequence"]=exploded_index_numbers
 
    # Accumulate the split differences to the initial 'timesincelaunch' values
    package_lst['timesincelaunch'] = package_lst['timesincelaunch_initial'] + package_lst['timesincelaunch_diff_split'] * (package_lst['sequence']+1)
    #print(package_lst[['timesincelaunch_diff',"timesincelaunch","timesincelaunch_initial","timesincelaunch_diff_split"]].head(n=30))
    package_lst['player_id'] = package_lst['session_id'].astype(str) + '-' + package_lst['headset_on_counter'].astype(str)

    return package_lst


def ExtractEventRows(df_of_events:pd.DataFrame, target_event_name:str) -> pd.DataFrame:
    """
    Given a dataframe of OGD events, returns a new dataframe with only the "headset_on" and given target events.

    :param df_of_events: _description_
    :type df_of_events: pd.DataFrame
    :param target_event_name: _description_
    :type target_event_name: str
    :return: _description_
    :rtype: pd.DataFrame
    """
    # Keep event_name with 'data_name' & headset_on
    package_lst = df_of_events.copy()
    
    # Use the isin function to filter rows where event_name is either data_name or 'headset_on'    
    package_lst = package_lst[package_lst['event_name'].isin([target_event_name, 'headset_on'])].copy()
    
    # Create a new column 'headset_on_counter' that increments whenever 'headset_on' event is encountered
    package_lst['headset_on_counter'] = np.where(package_lst['event_name'] == 'headset_on', 1, 0)
    package_lst['headset_on_counter'] = package_lst['headset_on_counter'].cumsum()
    package_lst = package_lst[package_lst['event_name'] == target_event_name]
    package_lst = package_lst[package_lst['event_name'] == target_event_name]
    package_lst['player_id'] = package_lst['session_id'].astype(str) + '-' + package_lst['headset_on_counter'].astype(str)
    return package_lst

def SplitDataframeByPlayerID(df:pd.DataFrame) -> List[pd.DataFrame]:
    """
    Given a dataframe with a player ID column, generates a new list of dataframes, separated by the player ID.

    :param df: _description_
    :type df: pd.DataFrame
    :return: _description_
    :rtype: List[pd.DataFrame]
    """
    player_groups = df.groupby("player_id")

    split_dataframes = []

    for player_id, group_df in player_groups:
        split_dataframes.append(group_df)

    return split_dataframes

def SplitDataframeBySessionID(df:pd.DataFrame) -> List[pd.DataFrame]:
    """
    Given a dataframe with a session ID column, generates a new list of dataframes, separated by the session ID.

    :param df: _description_
    :type df: pd.DataFrame
    :return: _description_
    :rtype: List[pd.DataFrame]
    """
    player_groups = df.groupby("session_id")

    split_dataframes = []

    for player_id, group_df in player_groups:
        split_dataframes.append(group_df)

    return split_dataframes


  # https://github.com/christophhagen/averaging-quaternions

# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation
def averageQuaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = np.outer(q,q) + A
    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0].A1)

#average by n rows
    
def average_dataframes_by_rows(df:pd.DataFrame, num_rows:int) -> pd.DataFrame:
    df_grouped = df.groupby(df.index // num_rows)
    avg_df = df_grouped.agg({
        'session_id': 'first',
        'event_name': 'first',
        'timesincelaunch': np.mean,
        'position': lambda x: np.mean(np.vstack(x), axis=0).tolist(),
        'rotation': lambda x: averageQuaternions(np.vstack(x)).tolist(),
        'game_state_pos': lambda x: np.mean(np.vstack(x), axis=0).tolist(),
        'game_state_rot': lambda x: averageQuaternions(np.vstack(x)).tolist()
        })
    avg_df['rotation_x'] = avg_df['rotation'].apply(lambda x: x[1])# Extracting the x-component

    return avg_df


#sampling by rows

# Sample every nth row
def sample_dataframes_by_rows(df:pd.DataFrame, num_rows:int) -> pd.DataFrame:
        # Calculate average for each group
    df_grouped = df.groupby(df.index // num_rows)
    smp_df = df_grouped.agg({
    'session_id': 'first',
    'event_name': 'first',
    'timesincelaunch':'first',
    'position':  'first',
    'rotation': 'first',
    'game_state_pos': 'first',
    'game_state_rot': 'first'
        })   
    return smp_df

def QuaternionToViewVector(quaternion, view_vector=np.array([0,1,0])):
  # define the quaternion vector
  q = np.array(quaternion)

  # normalize the quaternion vector
  q_norm = q / np.linalg.norm(q)

  # compute the rotation matrix
  qx, qy, qz, qw = q_norm
  R = np.array([[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]])

  # compute the 3D vector
  #origional [1,0,0]
  v = np.dot(R, view_vector)
  #change the vector to see how the graph changes - why that change occur 
  return(v)

def QuaternionsToViewVectors(series_of_quaternions, view_vector=np.array([0,1,0])):
    return series_of_quaternions.apply(lambda x : QuaternionToViewVector(x, view_vector=view_vector))

def NormalizeViewVector(view_vector:np.ndarray, target_scale:float):
    norm = np.linalg.norm(view_vector)  # calculate L2-norm of the vector
    return [(x / norm) * target_scale for x in view_vector]  # divide each element in the vector by the norm

def NormalizeViewVectors(series_of_vectors, target_scale):
    # apply the normalization function to the values in the 'col1' column
    return series_of_vectors.apply(lambda v: NormalizeViewVector(v, target_scale))







