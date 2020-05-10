"""06. Train Faster-RCNN end-to-end on PASCAL VOC
================================================

This tutorial goes through the basic steps of training a Faster-RCNN [Ren15]_ object detection model
provided by GluonCV.

Specifically, we show how to build a state-of-the-art Faster-RCNN model by stacking GluonCV components.

It is highly recommended to read the original papers [Girshick14]_, [Girshick15]_, [Ren15]_
to learn more about the ideas behind Faster R-CNN.
Appendix from [He16]_ and experiment detail from [Lin17]_ may also be useful reference.

.. hint::

    You can skip the rest of this tutorial and start training your Faster-RCNN model
    right away by downloading this script:

    :download:`Download train_faster_rcnn.py<../../../scripts/detection/faster_rcnn/train_faster_rcnn.py>`

    Example usage:

    Train a default resnet50_v1b model with Pascal VOC on GPU 0:

    .. code-block:: bash

        python train_faster_rcnn.py --gpus 0

    Train a resnet50_v1b model on GPU 0,1,2,3:

    .. code-block:: bash

        python train_faster_rcnn.py --gpus 0,1,2,3 --network resnet50_v1b

    Check the supported arguments:

    .. code-block:: bash

        python train_faster_rcnn.py --help


.. hint::

    Since lots of contents in this tutorial is very similar to :doc:`./train_ssd_voc`, you can skip any part
    if you feel comfortable.

"""

from gluoncv.data import VOCDetection

# typically we use 2007+2012 trainval splits for training data
train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])

# and use 2007 test as validation data
val_dataset = VOCDetection(splits=[(2007, 'test')])

print('Training images:', len(train_dataset))
print('Validation images:', len(val_dataset))


# For Faster-RCNN networks, the only data augmentation is horizontal flip.
from gluoncv.data.transforms import presets
from gluoncv import utils
from mxnet import nd

short, max_size = 600, 1000  # resize image to short side 600 px, but keep maximum length within 1000
utils.random.seed(233)  # fix seed in this tutorial

from gluoncv.data.batchify import Tuple, Append, FasterRCNNTrainBatchify
from mxnet.gluon.data import DataLoader

batch_size = 2  # for tutorial, we use smaller batch-size
num_workers = 0  # you can make it larger(if your CPU has more cores) to accelerate data loading

from gluoncv import model_zoo

net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained_base=False)
# print(net)

import mxnet as mx

net.initialize()

##############################################################################
# Faster-RCNN returns three values, where ``cids`` are the class labels,
# ``scores`` are confidence scores of each prediction,
# and ``bboxes`` are absolute coordinates of corresponding bounding boxes.

from mxnet import autograd
from gluoncv.data.transforms import presets
from gluoncv.utils import viz
import matplotlib
import matplotlib.pyplot as plt

# the loss to penalize incorrect foreground/background prediction
rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
# the loss to penalize inaccurate anchor boxes
rpn_box_loss = mx.gluon.loss.HuberLoss(rho=1 / 9.)  # == smoothl1
# the loss to penalize incorrect classification prediction.
rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
# and finally the loss to penalize inaccurate proposals
rcnn_box_loss = mx.gluon.loss.HuberLoss()  # == smoothl1

# If we provide network to the training transform function, it will compute training targets
train_transform = presets.rcnn.FasterRCNNDefaultTrainTransform(short, max_size, net, flip_p=0)
# Return images, labels, rpn_cls_targets, rpn_box_targets, rpn_box_masks loosely
batchify_fn = FasterRCNNTrainBatchify(net)
# For the next part, we only use batch size 1
batch_size = 1
train_loader = DataLoader(train_dataset.transform(train_transform), batch_size, shuffle=False,
                          batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)

teacher = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
for ib, batch in enumerate(train_loader):
    print(ib)
    if ib > 3:
        break
    with autograd.record():
        for image_idx, (data_batch, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks) in enumerate(zip(*batch)):
            with autograd.pause():
                teacher_img = train_dataset[batch_size * ib + image_idx][0]
                transformed_img = presets.rcnn.transform_test(teacher_img)[0]
                ids, scores, bboxes, cls_probailities = teacher(transformed_img)
                teacher_label = train_dataset[batch_size * ib + image_idx][1]

            # for i, score in enumerate(scores[0]):
            #     if not score < 0.5 and not ids[0][i] < 0:
                    # print(f'{i} {score} {bboxes[0][i]} {ids[0][i]}')

            label = label.expand_dims(0)
            gt_label = label[:, :, 4:5]
            gt_box = label[:, :, :4]

            train_image = data_batch
            train_image = train_image.transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
            train_image = (train_image * 255).asnumpy().astype('uint8')
            matplotlib.use('TkAgg')
            ax = viz.plot_bbox(train_image, teacher_label[:, 0:4], mx.ndarray.ones(teacher_label.shape[0]), teacher_label[:, 4], class_names=teacher.classes)
            plt.show()

            # network forward
            cls_preds, box_preds, roi, samples, matches, rpn_score, rpn_box, anchors, cls_targets, \
                box_targets, box_masks, _ = net(data_batch.expand_dims(0), bboxes, ids)


            # # losses of rpn
            # rpn_score = rpn_score.squeeze(axis=-1)
            # num_rpn_pos = (rpn_cls_targets >= 0).sum()
            # rpn_loss1 = rpn_cls_loss(rpn_score, rpn_cls_targets,
            #                          rpn_cls_targets >= 0) * rpn_cls_targets.size / num_rpn_pos
            # rpn_loss2 = rpn_box_loss(rpn_box, rpn_box_targets,
            #                          rpn_box_masks) * rpn_box.size / num_rpn_pos

            # losses of rcnn
            num_rcnn_pos = (cls_targets >= 0).sum()
            rcnn_loss1 = rcnn_cls_loss(cls_preds, cls_targets,
                                       cls_targets >= 0) * cls_targets.size / cls_targets.shape[
                             0] / num_rcnn_pos
            rcnn_loss2 = rcnn_box_loss(box_preds, box_targets, box_masks) * box_preds.size / \
                         box_preds.shape[0] / num_rcnn_pos

        # some standard gluon training steps:
        # autograd.backward([rpn_loss1, rpn_loss2, rcnn_loss1, rcnn_loss2])
        # trainer.step(batch_size)
