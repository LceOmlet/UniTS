#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
import pandas as pd
import os
import glob
# from ts_datasets.ts_datasets.base import BaseDataset, _main_fns_docstr
# from ..base import BaseDataset, _main_fns_docstr
from typing import Tuple
import logging
import sys
from pathlib import Path
import zipfile
import requests
from .utils.utils import normalize

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)



_intro_docstr = "Base dataset class for storing time series as ``pd.DataFrame`` s."
_intro_docstr = "Base dataset class for storing time series as ``pd.DataFrame`` s."

_main_fns_docstr = """
Each dataset supports the following features:

1.  ``__getitem__``: you may call ``ts, metadata = dataset[i]``. ``ts`` is a time-indexed ``pandas`` DataFrame, with
    each column representing a different variable (in the case of multivariate time series). ``metadata`` is a dict or
    ``pd.DataFrame`` with the same index as ``ts``, with different keys indicating different dataset-specific
    metadata (train/test split, anomaly labels, etc.) for each timestamp.
2.  ``__len__``:  Calling ``len(dataset)`` will return the number of time series in the dataset.
3.  ``__iter__``: You may iterate over the ``pandas`` representations of the time series in the dataset with
    ``for ts, metadata in dataset: ...``

.. note::

    For each time series, the ``metadata`` will always have the key ``trainval``, which is a 
    ``pd.Series`` of ``bool`` indicating whether each timestamp of the time series should be
    training/validation (if ``True``) or testing (if ``False``). 
"""


class BaseDataset:
    __doc__ = _intro_docstr + _main_fns_docstr

    time_series: list
    """
    A list of all individual time series contained in the dataset. Iterating over
    the dataset will iterate over this list. Note that for some large datasets, 
    ``time_series`` may be a list of filenames, which are read lazily either during
    iteration, or whenever ``__getitem__`` is invoked.
    """

    metadata: list
    """
    A list containing the metadata for all individual time series in the dataset.
    """

    def __init__(self):
        self.subset = None
        self.time_series = []
        self.metadata = []

    def __getitem__(self, i) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.time_series[i], self.metadata[i]

    def __len__(self):
        return len(self.time_series)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def describe(self):
        for ts_df in self.time_series:
            print(f"length of the data: {len(ts_df)}")
            print(f"timestamp index name: {ts_df.index.name}")
            print(f"number of data columns: {len(ts_df.columns)}")
            print("data columns names (the first 20): ")
            print(ts_df.columns[:20])
            print(f"number of null entries: {ts_df.isnull().sum()}")

_intro_docstr = """
Base dataset class for storing time series intended for anomaly detection.
"""

_extra_note = """

.. note::

    For each time series, the ``metadata`` will always have the key ``anomaly``, which is a 
    ``pd.Series`` of ``bool`` indicating whether each timestamp is anomalous.
"""


class TSADBaseDataset(BaseDataset):
    __doc__ = _intro_docstr + _main_fns_docstr + _extra_note

    @property
    def max_lead_sec(self):
        """
        The maximum number of seconds an anomaly may be detected early, for
        this dataset. ``None`` signifies no early detections allowed, or that
        the user may override this value with something better suited for their
        purposes.
        """
        return None

    @property
    def max_lag_sec(self):
        """
        The maximum number of seconds after the start of an anomaly, that we
        consider detections to be accurate (and not ignored for being too late).
        ``None`` signifies that any detection in the window is acceptable, or
        that the user may override this value with something better suited for
        their purposes.
        """
        return None

    def describe(self):
        anom_bds = []
        anom_locs = []
        anom_in_trainval = []
        for ts, md in self:
            boundaries = md.anomaly.iloc[1:] != md.anomaly.values[:-1]
            boundaries = boundaries[boundaries].index
            if len(boundaries) == 0:
                continue

            ts_len = ts.index[-1] - ts.index[0]
            if md.anomaly.iloc[0]:
                anom_bds.append((ts.index[0], boundaries[0]))
                anom_locs.append((boundaries[0] - ts.index[0]) / ts_len)
                anom_in_trainval.append(True)

            for t0, tf in zip(boundaries[:-1], boundaries[1:]):
                if md.anomaly[t0]:
                    anom_bds.append((t0, tf))
                    anom_locs.append((tf - ts.index[0]) / ts_len)
                    anom_in_trainval.append(bool(md.trainval[t0]))

            if md.anomaly[boundaries[-1]]:
                anom_bds.append((boundaries[-1], ts.index[-1]))
                anom_locs.append(1.0)
                anom_in_trainval.append(False)

        print("=" * 80)
        print(f"Time series in dataset have average length {int(np.mean([len(ts) for ts, md in self]))}.")
        print(f"Time series in dataset have {len(anom_bds) / len(self):.1f} anomalies on average.")
        print(
            f"{sum(anom_in_trainval) / len(anom_in_trainval) * 100:.1f}% of "
            f"anomalies are in the train/val split of their respective time "
            f"series."
        )
        print(f"Anomalies in dataset have average length {pd.Timedelta(np.mean([(tf - t0) for t0, tf in anom_bds]))}.")
        print(
            f"Average anomaly occurs {np.mean(anom_locs) * 100:.1f}% "
            f"(+/- {np.std(anom_locs) * 100:.1f}%) of the way through "
            f"its respective time series."
        )
        print("=" * 80)



class UCR(TSADBaseDataset):
    """
    Data loader for the Hexagon ML/UC Riverside Time Series Anomaly Archive.

    See `here <https://compete.hexagon-ml.com/practice/competition/39/>`_ for details.

    Hoang Anh Dau, Eamonn Keogh, Kaveh Kamgar, Chin-Chia Michael Yeh, Yan Zhu,
    Shaghayegh Gharghabi, Chotirat Ann Ratanamahatana, Yanping Chen, Bing Hu,
    Nurjahan Begum, Anthony Bagnall , Abdullah Mueen, Gustavo Batista, & Hexagon-ML (2019).
    The UCR Time Series Classification Archive. URL https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
    """

    def __init__(self, rootdir=None):
        super().__init__()
        if rootdir is None:
            fdir = os.path.dirname(os.path.abspath(__file__))
            merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
            rootdir = os.path.join(merlion_root, "data", "ucr")

        self.download(rootdir)
        self.time_series = sorted(
            glob.glob(
                os.path.join(
                    rootdir, "AnomalyDatasets_2021", "UCR_TimeSeriesAnomalyDatasets2021", "FilesAreInHere", "UCR_Anomaly_FullData", "*.txt"
                )
            )
        )

    def __getitem__(self, i):
        fname = self.time_series[i]
        split, anom_start, anom_end = [int(x) for x in fname[: -len(".txt")].split("_")[-3:]]
        name = fname.split("_")[-4]
        arr = np.loadtxt(fname)
        trainval = [i < split for i in range(len(arr))]
        anomaly = [anom_start <= i <= anom_end for i in range(len(arr))]
        index = pd.date_range(start=0, periods=len(arr), freq="1min")
        df = pd.DataFrame({name: arr}, index=index)
        return (
            df,
            pd.DataFrame(
                {
                    "anomaly": [anom_start - 100 <= i <= anom_end + 100 for i in range(len(arr))],
                    "trainval": [i < split for i in range(len(arr))],
                },
                index=index,
            ),
        )

    def download(self, rootdir):
        filename = "UCR_TimeSeriesAnomalyDatasets2021.zip"
        url = f"https://www.cs.ucr.edu/~eamonn/time_series_data_2018/{filename}"

        os.makedirs(rootdir, exist_ok=True)
        compressed_file = os.path.join(rootdir, filename)

        # Download the compressed dataset
        if not os.path.exists(compressed_file):
            logger.info("Downloading " + url)
            with requests.get(url, stream=True) as r:
                with open(compressed_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            f.flush()

        # Uncompress the downloaded zip file
        if not os.path.isfile(os.path.join(rootdir, "_SUCCESS")):
            logger.info(f"Uncompressing {compressed_file}")
            with zipfile.ZipFile(compressed_file, "r") as zip_ref:
                zip_ref.extractall(rootdir)
            Path(os.path.join(rootdir, "_SUCCESS")).touch()

def subsequences(sequence, window_size, time_step):
    # An array of non-contiguous memory is converted to an array of contiguous memory
    sq = np.ascontiguousarray(sequence)
    a = (sq.shape[0] - window_size + time_step) % time_step
    # label array
    if sq.ndim == 1:
        shape = (int((sq.shape[0] - window_size + time_step) / time_step), window_size)
        stride = sq.itemsize * np.array([time_step * 1, 1])
        if a != 0:
            sq = sq[:sq.shape[0] - a]
    # data array
    elif sq.ndim == 2:
        shape = (int((sq.shape[0] - window_size + time_step) / time_step), window_size, sq.shape[1])
        stride = sq.itemsize * np.array([time_step * sq.shape[1], sq.shape[1], 1])
        if a != 0:
            sq = sq[:sq.shape[0] - a, :]
    else:
        print('Array dimension error')
        os.exit()
    # print(sq.strides)
    sq = np.lib.stride_tricks.as_strided(sq, shape=shape, strides=stride)
    return sq

def data_generator1(train_data, test_data, train_labels, test_labels, configs):
    train_time_series_ts = train_data
    test_time_series_ts = test_data

    split = len(train_time_series_ts)

    ts = np.concatenate([train_time_series_ts, test_time_series_ts], axis=0)

    train_time_series, test_time_series = ts[:split], ts[split:]

    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)
    # print("*"*200 + "window_size: " + str(configs["window_size"]))
    train_x = subsequences(train_time_series, configs["window_size"], configs["time_step"])
    test_x = subsequences(test_time_series, configs["window_size"], configs["time_step"])
    train_y = subsequences(train_labels, configs["window_size"], configs["time_step"])
    test_y = subsequences(test_labels, configs["window_size"], configs["time_step"])

    # train_y_window = np.zeros(train_x.shape[0])
    # test_y_window = np.zeros(test_x.shape[0])
    train_anomaly_window_num = 0
    # print(f"total ys: {sum(test_labels)}, length y: {len(test_labels)}")
    # print(f"total ys: {sum(test_y)}, length y: {test_y.shape}")
    # raise RuntimeError()

    train_y_window = np.sum(train_y[:, :configs["time_step"]], axis=-1) >= 1
    train_anomaly_window_num = np.sum(train_y_window)
    test_y_window = np.sum(test_y[:, :configs["time_step"]], axis=-1) >= 1
    test_anomaly_window_num = np.sum(test_y_window)
    print(test_y_window.shape, train_y_window.shape)
    print(f"train anomaly window num: {train_anomaly_window_num}, "
          f"test anomaly window num: {np.sum(test_y_window)}")
    # raise RuntimeError()
    # for i, item in enumerate(train_y[:]):
    #     if sum(item[:configs["time_step"]]) >= 1:
    #         train_anomaly_window_num += 1
    #         train_y_window[i] = 1
    #     else:
    #         train_y_window[i] = 0
    # for i, item in enumerate(test_y[:]):
    #     if sum(item[:configs["time_step"]]) >= 1:

    #         test_y_window[i] = 1
    #     else:
    #         test_y_window[i] = 0
    # train_x, val_x, train_y, val_y = train_test_split(train_x, train_y_window, test_size=0.2, shuffle=False)
    train_x = train_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))
    # train_x = train_x[:256]
    # test_x = test_x[:256]
    # train_y_window = train_y_window[:256]
    # test_y_window = test_y_window[:256]
    split = [list(range(len(train_x))), list(range(len(train_x), len(train_x) + len(test_x)))]
    X_ = np.concatenate((train_x, test_x), axis=0)
    y_ = np.concatenate((train_y_window, test_y_window), axis=0)


    return X_, y_, split, test_anomaly_window_num