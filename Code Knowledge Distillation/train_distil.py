import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd, gluon
from mxnet.gluon.data import DataLoader
from gluoncv import model_zoo, data, utils
from gluoncv.data import VOCDetection
from gluoncv.data.transforms import presets
from gluoncv.data.batchify import Tuple, Append, FasterRCNNTrainBatchify
from gluoncv.utils import viz
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import tkinter
import time
import os
from joblib import dump, load


def train(student, teacher):
    student.initialize()
    batch_size = 1

    # train_dataset = data.COCODetection(splits=['instances_train2017'])
    train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
    train_transform = presets.rcnn.FasterRCNNDefaultTrainTransform(net=student)
    batchify_fn = Tuple(Append(), Append())
    train_loader = DataLoader(train_dataset.transform(train_transform), batch_size=batch_size, shuffle=False,
                              batchify_fn=FasterRCNNTrainBatchify(student), last_batch='rollover')

    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_box_loss = mx.gluon.loss.HuberLoss()  # == smoothl1

    trainer = gluon.Trainer(student.collect_params(), 'sgd',
                            {'learning_rate': 0.01, 'wd': 0.0005, 'momentum': 0.9})

    for batch_idx, batch in enumerate(train_loader):
        # batch[0] : images
        # batch[1] : labels (1, X (anchors boxes), 5 (bbox coordinates + class))
        if batch_idx > 0:
            break
        with autograd.record():
            # for image_idx, student_img in enumerate(batch[0]):
            image_idx = 0
            for student_img, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks in zip(*batch):
                with autograd.pause():
                    teacher_img = train_dataset[batch_size*batch_idx + image_idx][0]
                    teacher_img = data.transforms.presets.rcnn.transform_test(teacher_img)[0]
                    ids, scores, bboxes, cls_probailities = teacher(teacher_img)

                # network forward
                cls_pred, box_pred, _, _, _, _, _, _, cls_targets, box_targets, box_masks, _ = \
                    student(student_img.expand_dims(0), bboxes, ids)

                # losses
                num_rcnn_pos = (cls_targets >= 0).sum()
                rcnn_loss1 = rcnn_cls_loss(cls_pred, cls_targets,
                                           cls_targets >= 0) * cls_targets.size / cls_targets.shape[0] / num_rcnn_pos
                rcnn_loss2 = rcnn_box_loss(box_pred, box_targets, box_masks) * box_pred.size / box_pred.shape[0] / num_rcnn_pos

                autograd.backward([rcnn_loss1, rcnn_loss2])
                image_idx += 1

            trainer.step(batch_size=32)


teacher = model_zoo.get_model('faster_rcnn_resnet101_v1d_coco', pretrained=True)
student = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=False)

train(student, teacher)
