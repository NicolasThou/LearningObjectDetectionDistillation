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

import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet.gluon.data import DataLoader
import gluon
from gluoncv.data import VOCDetection
from gluoncv import utils
from gluoncv.data.batchify import Tuple, Append, FasterRCNNTrainBatchify
from gluoncv import model_zoo
from gluoncv.data.transforms import presets
from gluoncv import data
from gluoncv.utils import viz
import matplotlib
import matplotlib.pyplot as plt


def extract_boxes(scores, labels):
    assert len(scores) == len(labels)

    idx = []
    for i in range(len(scores)):
        if scores[i] < 0.5 or labels[i] < 0:
            continue

        idx.append(i)
        # print(f'{scores[i]} {labels[i]}')

    return idx


def inverse_transformation(image):
    image = image.transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
    image = (image * 255).asnumpy()
    return image


train_dataset = data.COCODetection(splits=['instances_train2017'])
# train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])

teacher = model_zoo.get_model('faster_rcnn_resnet101_v1d_coco', pretrained=True)
# teacher = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)

student = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=False)
student.initialize()


# the loss to penalize incorrect foreground/background prediction
rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
# the loss to penalize inaccurate anchor boxes
rpn_box_loss = mx.gluon.loss.HuberLoss(rho=1 / 9.)  # == smoothl1
# the loss to penalize incorrect classification prediction.
rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
# and finally the loss to penalize inaccurate proposals
rcnn_box_loss = mx.gluon.loss.HuberLoss()  # == smoothl1

# If we provide network to the training transform function, it will compute training targets
train_transform = presets.rcnn.FasterRCNNDefaultTrainTransform(net=student, flip_p=0)

# Return images, labels, rpn_cls_targets, rpn_box_targets, rpn_box_masks loosely
batchify_fn = FasterRCNNTrainBatchify(student)

batch_size = 1  # for tutorial, we use smaller batch-size
train_loader = DataLoader(train_dataset.transform(train_transform), batch_size, shuffle=False,
                          batchify_fn=batchify_fn, last_batch='rollover', num_workers=0)

trainer = gluon.Trainer(student.collect_params(), 'sgd',
                            {'learning_rate': 0.01, 'wd': 0.0005, 'momentum': 0.9})

matplotlib.use('TkAgg')
for batch_idx, batch in enumerate(train_loader):
    if batch_idx > 5:
        break
    with autograd.record():
        for image_idx, (data_batch, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks) in enumerate(zip(*batch)):
            # teacher prediction
            with autograd.pause():
                teacher_img, teacher_label = train_dataset[batch_size * batch_idx + image_idx]
                transformed_img, original_teacher_img = presets.rcnn.transform_test(teacher_img)
                ids, scores, bboxes, cls_prob = teacher(transformed_img)

            idx = extract_boxes(scores[0], ids[0])
            bboxes = bboxes[:, idx, :]
            scores = scores[:, idx, :]
            ids = ids[:, idx, :]

            label = label.expand_dims(0)
            gt_label = label[:, :, 4:5]
            gt_box = label[:, :, :4]

            # inverse transformation to get image
            train_image = inverse_transformation(data_batch)

            # viz.plot_bbox(original_teacher_img, gt_box[0], mx.ndarray.ones(gt_box[0].shape[0]), gt_label[0], class_names=teacher.classes)
            # viz.plot_bbox(original_teacher_img, bboxes[0], scores[0], ids[0], class_names=teacher.classes)
            # viz.plot_bbox(train_image, gt_box[0], mx.ndarray.ones(gt_box[0].shape[0]), gt_label[0], class_names=teacher.classes)
            viz.plot_bbox(train_image, bboxes[0], scores[0], ids[0], class_names=teacher.classes)
            # viz.plot_bbox(teacher_img, teacher_label[:, :4], mx.ndarray.ones(teacher_label.shape[0]), teacher_label[:, 4], class_names=teacher.classes)
            plt.show()

            # network forward
            cls_preds, box_preds, roi, samples, matches, rpn_score, rpn_box, anchors, cls_targets, \
                box_targets, box_masks, _ = student(data_batch.expand_dims(0), bboxes, ids)

            # # losses of rpn
            rpn_score = rpn_score.squeeze(axis=-1)
            num_rpn_pos = (rpn_cls_targets >= 0).sum()
            rpn_loss1 = rpn_cls_loss(rpn_score, rpn_cls_targets,
                                     rpn_cls_targets >= 0) * rpn_cls_targets.size / num_rpn_pos
            rpn_loss2 = rpn_box_loss(rpn_box, rpn_box_targets,
                                     rpn_box_masks) * rpn_box.size / num_rpn_pos

            # losses of rcnn
            num_rcnn_pos = (cls_targets >= 0).sum()
            rcnn_loss1 = rcnn_cls_loss(cls_preds, cls_targets,
                                       cls_targets >= 0) * cls_targets.size / cls_targets.shape[
                             0] / num_rcnn_pos
            rcnn_loss2 = rcnn_box_loss(box_preds, box_targets, box_masks) * box_preds.size / \
                         box_preds.shape[0] / num_rcnn_pos

            autograd.backward([rpn_loss1, rpn_loss2, rcnn_loss1, rcnn_loss2])

        trainer.step(batch_size)
