"""02. Predict with pre-trained Faster RCNN models
==============================================

This article shows how to play with pre-trained Faster RCNN model.
First let's import some necessary libraries:

"""
from __future__ import print_function

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
import numpy as np
from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
from gluoncv.model_zoo.rcnn.faster_rcnn import *
import time

"""

Teacher : 'faster_rcnn_resnet101_v1d_coco'
Student : 'faster_rcnn_resnet50_v1b_coco'

"""


"""
======================================= Faster RCNN ================================================
"""

######################################################################
# Load a pretrained model
# -------------------------
#
# Let's get an Faster RCNN model trained on Pascal VOC
# dataset with ResNet-50 backbone. By specifying
# ``pretrained=True``, it will automatically download the model from the model
# zoo if necessary. For more pretrained models, please refer to
# :doc:`../../model_zoo/index`.
#
# The returned model is a HybridBlock :py:class:`gluoncv.model_zoo.FasterRCNN`
# with a default context of `cpu(0)`.

net = model_zoo.get_model('faster_rcnn_resnet101_v1d_coco', pretrained=True)


######################################################################
# Pre-process an image
# --------------------
#
# Next we download an image, and pre-process with preset data transforms.
# The default behavior is to resize the short edge of the image to 600px.
# But you can feed an arbitrarily sized image.
#
# You can provide a list of image file names, such as ``[im_fname1, im_fname2,
# ...]`` to :py:func:`gluoncv.data.transforms.presets.rcnn.load_test` if you
# want to load multiple image together.
#
# This function returns two results. The first is a NDArray with shape
# `(batch_size, RGB_channels, height, width)`. It can be fed into the
# model directly. The second one contains the images in numpy format to
# easy to be plotted. Since we only loaded a single image, the first dimension
# of `x` is 1.
#
# Please beware that `orig_img` is resized to short edge 600px.


im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/biking.jpg?raw=true',
                          path='biking.jpg')
x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)


print()
print("===================== Let's compare the two models ==============================")
print()

######################################################################
# Inference and display
# ---------------------
#
# The Faster RCNN model returns
# predicted class IDs (batch_size, num_bboxes, 1)
# confidence scores (batch_size, num_bboxes, 1)
# bounding boxes coordinates. (batch_size, num_bboxes, 4)
#
# We can use :py:func:`gluoncv.utils.viz.plot_bbox` to visualize the
# results. We slice the results for the first image and feed them into `plot_bbox`:

start = time.time()
box_ids, scores, bboxes, cls_score = net(x)
end = time.time()
print(f'Resnet 101 took {end-start} seconds')

# save the resulting predictions
nd.save('resnet101_predictions', [cls_score])



# bboxes, scores, box_ids = nd.load('resnet101_predictions')  # load the predictions
ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)
plt.show()


net2 = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=True)

start = time.time()
box_ids2, scores2, bboxes2, cls_score2 = net(x)
end = time.time()
print(f'Resnet 50 took {end-start} seconds')

# bboxes, scores, box_ids = nd.load('resnet50_predictions')  # load the predictions
ax2 = utils.viz.plot_bbox(orig_img, bboxes2[0], scores2[0], box_ids2[0], class_names=net2.classes)
plt.show()


"""
========================== Compare the output during training and the output of the teacher =========================
"""
