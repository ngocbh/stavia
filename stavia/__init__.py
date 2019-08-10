from __future__ import absolute_import

from . import fuzzy_matching
from . import crf
from . import utils
from .standardizer import standardize, standardize4testing
from .crf.tagger import detect_entity

__version__ = '0.0.1'