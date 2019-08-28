from __future__ import absolute_import

from . import fuzzy_matching
from . import crf
from . import utils
from .standardizer import standardize, standardize4testing
from .crf.tagger import detect_entity 
from .crf import crf_based_standardization as cbs

__version__ = '0.0.1'