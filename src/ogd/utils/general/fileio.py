# import standard libraries
import json
import shutil
from pathlib import Path
import urllib.request as urlrequest

from enum import Enum
from io import BytesIO
from typing import List, Optional, Tuple
from zipfile import ZipFile

# import 3rd-party libraries
import numpy as np
import pandas as pd

class FileTypes(Enum):
    SESSION = "sessions"
    PLAYER = "players"
    POPULATION = "population"
    EVENTS = "events"
    ALL_EVENTS = "all_events"

    def __str__(self):
        return self.value

class FileAPI:

    _api_server = 'http://ogd-services.fielddaylab.wisc.edu/'
    _api_path = 'wsgi-bin/opengamedata-website-api/production/app.wsgi/'

    @property
    def APIServer(self):
        return FileAPI._api_server

    @property
    def APIPath(self):
        return FileAPI._api_path

    @staticmethod
    def GetAvailableMonths(game_id:str, api_server:Optional[str] = None, api_path:Optional[str] = None) -> List[str]:
        """Function to retrieve a list of datasets available for a given game.

        :param game_id: The game whose dataset should be downloaded
        :type game_id: str
        :param api_server: A custom file server, if different from default, or use default server if None. The default can be accessed by the APIServer property.
        :type api_server: Optional[str], optional
        :param api_path: A custom path to the API on the file server, if different from default, or use default path if None. The default can be accessed by the APIPath property.
        :type api_path: Optional[str], optional
        :return: A list of month/year strings indicating the available datasets for the given game
        :rtype: List[str]
        """
        ret_val = []

        _server = api_server or FileAPI._api_server
        _path   = api_path   or FileAPI._api_path
        game_usage_link = f'{_server}{_path}getMonthlyGameUsage?game_id={game_id}'
        with urlrequest.urlopen(game_usage_link) as remote_list:
            json_data    = json.loads(remote_list.read())
            session_data = json_data.get('data', {}).get('sessions', {})
            ret_val = [f"{elem.get('month')}/{elem.get('year')}" for elem in session_data]
        return ret_val

    @staticmethod
    def DownloadZippedDataset(game_id:str, month:int, year:int, datatype:FileTypes, local_path:Path=Path('./'), api_server:Optional[str] = None, api_path:Optional[str] = None) -> Tuple[Optional[ZipFile], str]:
        """Function to retrieve a dataset for a given month/year of a given game.

        :param game_id: The game whose dataset should be downloaded
        :type game_id: str
        :param month: The month of the dataset to download
        :type month: int
        :param year: The year of the dataset to download
        :type year: int
        :param datatype: The type of dataset to download
        :type datatype: FileTypes
        :param local_path: Path to the local folder where the zipped dataset should be stored. Defaults to "./"
        :type local_path: str
        :param api_server: A custom file server, if different from default, or use default server if None. The default can be accessed by the APIServer property.
        :type api_server: Optional[str], optional
        :param api_path: A custom path to the API on the file server, if different from default, or use default path if None. The default can be accessed by the APIPath property.
        :type api_path: Optional[str], optional
        :return: The name of the dataset that was downloaded. This is needed to locate the tsv file within the downloaded zip.
        :rtype: str
        """
        dataset_name = "REMOTE DATASET NOT FOUND"
        zip_file     = None

        _server = api_server or FileAPI.APIServer
        _path   = api_path   or FileAPI.APIPath
        month_data_link = f'{_server}{_path}getGameFileInfoByMonth?game_id={game_id}&year={year}&month={month}'
        with urlrequest.urlopen(month_data_link) as remote_list:
            json_data    = json.loads(remote_list.read())
            # pprint(json_data)
            file_url = json_data.get('data', {}).get(f"{datatype}_file")
            zip_name = file_url.split('/')[-1]
            dataset_name = f"{'_'.join(zip_name.split('_')[:-2])}"
            zip_path = local_path / zip_name
            # dataset_name = f"{zip_name[:zip_name.rfind('_')]}"
            if not zip_path.is_file():
                print(f"Didn't find the file {zip_path} locally, downloading from {_server}...")
                with urlrequest.urlopen(file_url) as remote_file, open(zip_path, 'wb') as local_file:
                    shutil.copyfileobj(remote_file, local_file)
                    print(f"Successfully downloaded a copy of the file.")
            else:
                print(f"Found the file {zip_name} locally, nothing will be downloaded.")
            zip_file = ZipFile(Path(f'./{zip_name}'))
        return (zip_file, dataset_name)

def openZipFromURL(url):
    """

    :param url: url pointing to a zipfile
    :return: zipfile object, list of metadata lines
    """
    metadata = [f'Import from f{url}']
    resp = urlrequest.urlopen(url)
    zipfile = ZipFile(BytesIO(resp.read()))

    return zipfile, metadata


def openZipFromPath(path):
    """

    :param path: path pointing to a zipfile
    :return: zipfile object, list of metadata lines
    """
    metadata = [f'Import from f{path}']
    zipfile = ZipFile(path)

    return zipfile, metadata


def readCSVFromPath(path, index_cols):
    """

    :param path: path pointing to a csv
    :return: dataframe, List[str] of metadata lines
    """
    import os
    print(os.getcwd())
    metadata = [f'Import from f{path}']
    df = pd.read_csv(path, index_col=index_cols, comment='#')
    return df, metadata


def getZippedLogDFbyURL(proc_zip_urls, index_cols=['sessionID']):
    """

    :param proc_urls: List of urls to proc data file zips.
    :param index_cols: List of columns to be treated as index columns.
    :return: (df, metadata List[str])
    """
    # get the data
    metadata = []
    df = pd.DataFrame()
    for next_url in proc_zip_urls:
        zf, meta = openZipFromURL(next_url)
        # put the data into a dataframe
        with zf.open(zf.namelist()[0]) as f:
            df = pd.concat(
                [df, pd.read_csv(f, index_col=index_cols, comment='#')], sort=True)
        metadata.extend(meta)
    if len(index_cols) > 1:
        for i, col_name in enumerate(index_cols):
            df[col_name] = [x[i] for x in df.index]
    else:
        df[index_cols[0]] = [x for x in df.index]
    return df, metadata


def getLogDFbyPath(proc_paths, zipped=True, index_cols=['sessionID']):
    """

    :param proc_paths: List of paths to proc data files.
    :param zipped: True if files are zipped, false if just CSVs (default True).
    :param index_cols: List of columns to be treated as index columns.
    :return: (df, metadata List[str])
    """
    # get the data
    metadata = []
    df = pd.DataFrame()
    for next_path in proc_paths:
        if zipped:
            next_file, meta = openZipFromPath(next_path)
            # put the data into a dataframe
            with next_file.open(next_file.namelist()[0]) as f:
                df = pd.concat(
                    [df, pd.read_csv(f, index_col=index_cols, comment='#')], sort=True)
        else:  # CSVs, not zips
            next_file, meta = readCSVFromPath(next_path, index_cols)
            # put the data into a dataframe
            df = pd.concat([df, next_file], sort=True)
        metadata.extend(meta)
    if len(index_cols) > 1:
        for i, col_name in enumerate(index_cols):
            df[col_name] = [x[i] for x in df.index]
    else:
        df[index_cols[0]] = [x for x in df.index]
    return df, metadata