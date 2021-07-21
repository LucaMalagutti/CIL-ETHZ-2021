from . import utils
from .netflix import load_netflix_data
from .utils import ML_1M, ML_100K, NETFLIX


def load_data(kind):
    utils.make_dir_if_not_exists("data")

    if kind == NETFLIX:
        return load_netflix_data()
    else:
        raise NotImplementedError("Kind '{}' is not implemented yet.".format(kind))


def get_N_and_M(kind):
    if kind == NETFLIX:
        return 10000, 1000
    else:
        raise NotImplementedError("Kind '{}' is not implemented yet.".format(kind))
