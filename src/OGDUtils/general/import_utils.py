from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
import urllib.request


def openZipFromURL(url):
    """

    :param url: url pointing to a zipfile
    :return: zipfile object, list of metadata lines
    """
    metadata = [f'Import from f{url}']
    resp = urllib.request.urlopen(url)
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