import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet.gluon.data import DataLoader
from mxnet.gluon import Trainer
from gluoncv.data import VOCDetection
from gluoncv import utils
from gluoncv.data.batchify import Tuple, Append, FasterRCNNTrainBatchify
from gluoncv import model_zoo
from gluoncv.data.transforms import presets
from gluoncv import data
from gluoncv.utils import viz
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
import tkinter
import numpy as np
import time


def extract_boxes(scores, labels):
    assert len(scores) == len(labels)

    idx = []
    for i in range(len(scores)):
        if scores[i] < 0.09 or labels[i] < 0:
            continue
        idx.append(i)
        # print(scores[i])

    return idx


def inverse_transformation(image):
    image = image.transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
    image = (image * 255).asnumpy()
    return image


def teacher_bounded_regression_loss(y, teacher_pred, student_pred):
    l2_loss = mx.gluon.loss.L2Loss()
    y_teacher = l2_loss(y, teacher_pred)
    y_student = l2_loss(y, student_pred)
    if y_student > y_teacher:
        return y_student
    else:
        return 0


train_dataset = data.COCODetection(splits=['instances_train2017'])

model = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=False)
model.load_params('params/model_distil_49.params')

train_transform = presets.rcnn.FasterRCNNDefaultTrainTransform(flip_p=0)

train_loader = DataLoader(train_dataset.transform(train_transform), batch_size=1, shuffle=False,
                          last_batch='rollover', num_workers=0)

matplotlib.use('TkAgg')
for batch_idx, batch in enumerate(train_loader):
    if batch_idx > 5:
        break
    for data_img, data_label in zip(*batch):
        # teacher_img, teacher_label = train_dataset[batch_idx]
        # transformed_img, original_teacher_img = presets.rcnn.transform_test(teacher_img)
        # ids, scores, bboxes, teacher_prob = teacher(transformed_img)
        ids, scores, bboxes, teacher_prob = model(data_img.expand_dims(0))

        idx = extract_boxes(scores[0], ids[0])
        bboxes = bboxes[:, idx, :]
        scores = scores[:, idx, :]
        ids = ids[:, idx, :]

        data_label = data_label.expand_dims(0)
        gt_label = data_label[:, :, 4:5]
        gt_box = data_label[:, :, :4]

        print(bboxes[0])
        print(gt_box[0])

        train_image = inverse_transformation(data_img)  # inverse transformation to get image
        # viz.plot_bbox(original_teacher_img, gt_box[0], mx.ndarray.ones(gt_box[0].shape[0]), gt_label[0], class_names=teacher.classes)
        # viz.plot_bbox(original_teacher_img, bboxes[0], scores[0], ids[0], class_names=teacher.classes)
        viz.plot_bbox(train_image, gt_box[0], mx.ndarray.ones(gt_box[0].shape[0]), gt_label[0], class_names=model.classes)
        plt.show()
        viz.plot_bbox(train_image, bboxes[0], mx.ndarray.ones(bboxes[0].shape[0]), ids[0], class_names=model.classes)
        # viz.plot_bbox(teacher_img, teacher_label[:, :4], mx.ndarray.ones(teacher_label.shape[0]), teacher_label[:, 4], class_names=teacher.classes)
        plt.show()
