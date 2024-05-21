import os
import numpy as np

__version__ = '0.4'

from .OrvilleImageDB import OrvilleImageDB

_BAD_FREQ_PATH = os.path.dirname(os.path.abspath(__file__))
_BAD_FREQ_PATH = os.path.join(_BAD_FREQ_PATH, 'data', 'bad_freq.txt')

BAD_FREQ_LIST = np.array([], dtype=np.float32)
if os.path.exists(_BAD_FREQ_PATH):
    BAD_FREQ_LIST = np.loadtxt(_BAD_FREQ_PATH)
