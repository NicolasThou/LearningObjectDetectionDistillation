from __future__ import print_function

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd, gluon
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import DataLoader
import gluoncv
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms import presets
from gluoncv.data.batchify import Tuple, Append, FasterRCNNTrainBatchify
from gluoncv.model_zoo.rcnn.faster_rcnn import *
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from joblib import dump, load


def compare_prediction_time(teacher, student):
    teacher_records = []
    student_records = []
    teacher_chrono = 0
    student_chrono = 0
    for i, batch_file in enumerate(os.listdir('batchs')):
        if i > 3:
            break
        print(i)
        batch = load(os.path.join('batchs', batch_file))

        for image, label in zip(batch[0], batch[1]):
            # teacher
            start = time.time()
            teacher(image)
            end = time.time()
            teacher_chrono += (end - start)

            # student
            start = time.time()
            student(image)
            end = time.time()
            student_chrono += (end - start)

        teacher_records.append(teacher_chrono)
        student_records.append(student_chrono)

    plt.plot(range(len(teacher_records)), teacher_records, label='Teacher')
    plt.plot(range(len(student_records)), student_records, label='Student')
    plt.title('Time comparison for predictions between teacher and student')
    plt.xlabel('Number of batchs')
    plt.ylabel('Time for predicting all the batchs (sec)')
    plt.legend()
    plt.show()


# # training set
# train_dataset = data.COCODetection(splits=['instances_train2017'])
# train_transform = presets.rcnn.FasterRCNNDefaultTrainTransform()
# batchify_fn = Tuple(Append(), Append())
# train_loader = DataLoader(train_dataset.transform(train_transform), batch_size=32, shuffle=True,
#                           batchify_fn=batchify_fn, last_batch='rollover')
#
# for i, batch in enumerate(train_loader):
#     if i > 49:
#         break
#     print(i)
#     # batch[0] : images
#     # batch[1] : labels
#     dataset_to_dump = [batch[0], batch[1]]
#     dump(dataset_to_dump, f'batchs/batch_{i}.joblib')

# networks

student = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=True)
teacher = model_zoo.get_model('faster_rcnn_resnet101_v1d_coco', pretrained=True)

batch = load('batchs/batch_0.joblib')
out = teacher(batch[0])
print(out)
