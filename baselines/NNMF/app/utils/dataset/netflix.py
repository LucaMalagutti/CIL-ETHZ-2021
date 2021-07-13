import pandas as pd

from . import utils
from .utils import NETFLIX as KEY

_COL_NAMES = ['user_id', 'item_id', 'rating']
_DELIMITER = ','

def load_netflix_data():
    base_file_name, test_file_name = utils.split_data(
        KEY, 'data.csv', ('base', 'test'), rate=0.9)
    train_file_name, valid_file_name = utils.split_data(
        KEY, base_file_name, ('train', 'valid'), rate=0.98)
        
    train_data = pd.read_csv(
        utils.get_file_path(KEY, train_file_name),
        delimiter=_DELIMITER,
        header=0,
        names=_COL_NAMES)
    
    valid_data = pd.read_csv(
        utils.get_file_path(KEY, valid_file_name),
        delimiter=_DELIMITER,
        header=0,
        names=_COL_NAMES)

    test_data = pd.read_csv(
        utils.get_file_path(KEY, test_file_name),
        delimiter=_DELIMITER,
        header=0,
        names=_COL_NAMES)

    return {
        'train': train_data,
        'valid': valid_data,
        'test': test_data,
    }

