import mxnet as mx
from mxnet import nd
from mxnet.gluon.data import DataLoader
from gluoncv import utils
from gluoncv import model_zoo
from gluoncv.data.transforms import presets
from gluoncv import data
from gluoncv.utils import viz
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time


def extract_boxes(scores, labels, classes):
    assert len(scores) == len(labels)

    idx = []
    for i in range(len(scores)):
        if labels[i] < 0 or scores[i] < 0.03:
            continue
        print(f'{scores[i].asnumpy().item()} | {classes[int(labels[i].asnumpy().item())]}')
        idx.append(i)

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
model.load_parameters('params/model_1399.params', ignore_extra=True)
distil_model = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=False)
distil_model.load_parameters('params/model_distil_1399.params', ignore_extra=True)

train_transform = presets.rcnn.FasterRCNNDefaultTrainTransform(flip_p=0)

train_loader = DataLoader(train_dataset.transform(train_transform), batch_size=1, shuffle=True, last_batch='rollover')

matplotlib.use('TkAgg')
for batch_idx, batch in enumerate(train_loader):
    if batch_idx > 5:
        break
    for data_img, data_label in zip(*batch):
        ids, scores, bboxes, _ = model(data_img.expand_dims(0))
        distil_ids, distil_scores, distil_bboxes, _ = distil_model(data_img.expand_dims(0))

        print('No distil mode')
        idx = extract_boxes(scores[0], ids[0], model.classes)
        if len(idx) == 0:
            bboxes, scores, ids = [], [], []
        else:
            bboxes = bboxes[:, idx, :]
            scores = scores[:, idx, :]
            ids = ids[:, idx, :]

        print('Distil model')
        distil_idx = extract_boxes(distil_scores[0], distil_ids[0], distil_model.classes)
        if len(distil_ids) == 0:
            distil_bboxes, distil_scores, distil_ids = [], [], []
        else:
            distil_bboxes = distil_bboxes[:, distil_idx, :]
            distil_scores = distil_scores[:, distil_idx, :]
            distil_ids = distil_ids[:, distil_idx, :]

        data_label = data_label.expand_dims(0)
        gt_label = data_label[:, :, 4:5]
        gt_box = data_label[:, :, :4]

        train_image = inverse_transformation(data_img)  # inverse transformation to get image
        # viz.plot_bbox(train_image, gt_box[0], mx.ndarray.ones(gt_box[0].shape[0]), gt_label[0], class_names=model.classes)
        viz.plot_bbox(train_image, bboxes[0], mx.ndarray.ones(bboxes[0].shape[0]), ids[0], class_names=model.classes)
        viz.plot_bbox(train_image, distil_bboxes[0], mx.ndarray.ones(distil_ids[0].shape[0]), distil_ids[0], class_names=model.classes)
        plt.show()
