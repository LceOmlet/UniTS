import os
import pickle

import yaml
import numpy as np

SCALER_FNAME = "scaler.pkl"

def fill_out_with_Nan(data,max_length):
    #via this it can works on more dimensional array
    pad_length = max_length-data.shape[-1]
    if pad_length == 0:
        return data
    else:
        pad_shape = list(data.shape[:-1])
        pad_shape.append(pad_length)
        Nan_pad = np.empty(pad_shape)*np.nan
        return np.concatenate((data, Nan_pad), axis=-1)
    
def get_data_and_label_from_ts_file(file_path,label_dict):
    # print(file_path)
    # exit()
    with open(file_path, encoding='utf-8') as file:
        lines = file.readlines()
        Start_reading_data = False
        Label_list = []
        Data_list = []
        max_length = 0
        for line in lines:
            if Start_reading_data == False:
                if '@data'in line:
                    Start_reading_data = True
            else:
                temp = line.split(':')
                Label_list.append(label_dict[temp[-1].replace('\n','')])
                data_tuple= [np.expand_dims(np.fromstring(channel, sep=','), axis=0) for channel in temp[:-1]]
                max_channel_length = 0
                for channel_data in data_tuple:
                    if channel_data.shape[-1]>max_channel_length:
                        max_channel_length = channel_data.shape[-1]
                data_tuple = [fill_out_with_Nan(data,max_channel_length) for data in data_tuple]
                data = np.expand_dims(np.concatenate(data_tuple, axis=0), axis=0)
                Data_list.append(data)
                if max_channel_length>max_length:
                    max_length = max_channel_length
        
        Data_list = [fill_out_with_Nan(data,max_length) for data in Data_list]
        X =  np.concatenate(Data_list, axis=0)
        Y =  np.asarray(Label_list)
        
        return np.float32(X), Y

def get_label_dict(file_path):
    label_dict ={}
    with open(file_path, encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if '@classLabel' in line:
                label_list = line.replace('\n','').split(' ')[2:]
                # print(line)
                # exit()
                for i in range(len(label_list)):
                    label_dict[label_list[i]] = i 
                
                break
    return label_dict   

def load_yaml_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    return loaded


def load_data(data_dir: str, dataset: str) -> np.ndarray:
    """
    Load data from a dataset located in a directory.

    Args:
        data_dir (str): The directory where the dataset is located.
        dataset (str): The name of the dataset file (without the .npz extension).

    Returns:
        np.ndarray: The loaded dataset.
    """
    return get_npz_data(os.path.join(data_dir, f"{dataset}.npz"))


def save_data(data: np.ndarray, output_file: str) -> None:
    """
    Save data to a .npz file.

    Args:
        data (np.ndarray): The data to save.
        output_file (str): The path to the .npz file to save the data to.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(output_file, data=data)


def get_npz_data(input_file: str) -> np.ndarray:
    """
    Load data from a .npz file.

    Args:
        input_file (str): The path to the .npz file.

    Returns:
        np.ndarray: The data array extracted from the .npz file.
    """
    loaded = np.load(input_file)
    return loaded["data"]


def split_data(
    data: np.ndarray, valid_perc: float, shuffle: bool = True, seed: int = 123
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the data into training and validation sets.

    Args:
        data (np.ndarray): The dataset to split.
        valid_perc (float): The percentage of data to use for validation.
        shuffle (bool, optional): Whether to shuffle the data before splitting.
                                  Defaults to True.
        seed (int, optional): The random seed to use for shuffling the data.
                              Defaults to 123.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the training data and
                                       validation data arrays.
    """
    N = data.shape[0]
    N_train = int(N * (1 - valid_perc))

    if shuffle:
        np.random.seed(seed)
        data = data.copy()
        np.random.shuffle(data)

    train_data = data[:N_train]
    valid_data = data[N_train:]
    return train_data, valid_data


class MinMaxScaler:
    """Min Max normalizer.
    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """

    def fit_transform(self, data):
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data

    def fit(self, data):
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        return self

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data

    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data


def inverse_transform_data(data, scaler):
    return scaler.inverse_transform(data.copy())


def scale_data(train_data, valid_data):
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_valid_data = scaler.transform(valid_data)
    return scaled_train_data, scaled_valid_data, scaler


def save_scaler(scaler: MinMaxScaler, dir_path: str) -> None:
    """
    Save a MinMaxScaler to a file.

    Args:
        scaler (MinMaxScaler): The scaler to save.
        dir_path (str): The path to the directory where the scaler will be saved.

    Returns:
        None
    """
    os.makedirs(dir_path, exist_ok=True)
    scaler_fpath = os.path.join(dir_path, SCALER_FNAME)
    with open(scaler_fpath, "wb") as file:
        pickle.dump(scaler, file)


def load_scaler(dir_path: str) -> MinMaxScaler:
    """
    Load a MinMaxScaler from a file.

    Args:
        dir_path (str): The path to the file from which the scaler will be loaded.

    Returns:
        MinMaxScaler: The loaded scaler.
    """
    scaler_fpath = os.path.join(dir_path, SCALER_FNAME)
    with open(scaler_fpath, "rb") as file:
        scaler = pickle.load(file)
    return scaler
