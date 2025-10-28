from enum import Enum


class DataSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
