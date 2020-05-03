from __future__ import print_function
import logging
logging.basicConfig(level=logging.INFO)
import os
import time
from collections import OrderedDict
import skimage.io as io
import mxnet as mx
from mxnet.test_utils import download
mx.random.seed(127)

# setup the contexts; will use gpus if avaliable, otherwise cpu
gpus = mx.test_utils.list_gpus()
contexts = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]

from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils

net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)



if __name__ == '__main__':
    print("hello world")