# coding: utf-8
# pylint: disable=wrong-import-position
"""GluonCV: a deep learning vision toolkit powered by Gluon."""
from __future__ import absolute_import

__version__ = '0.7.0'

from mxnet.gluon import data
from mxnet.gluon import model_zoo
from mxnet.gluon import nn
from mxnet.gluon import utils
from gluoncv import model_zoo, data, utils, nn, loss
from gluoncv.utils.version import _require_mxnet_version, _deprecate_python2

_deprecate_python2()
_require_mxnet_version('1.4.0')