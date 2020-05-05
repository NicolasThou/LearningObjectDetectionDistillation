from mxnet import nd
import numpy as np
from mxnet.ndarray import *


if __name__ == '__main__':
    print("hello")
    print(24656/5)
    pred_student = nd.random.uniform(-1, 1, (2, 5, 4))
    target = nd.random.uniform(-1, 1, (2, 5, 4))
    pred_teacher = nd.random.uniform(-1, 1, (2, 5, 4))

    difference_student = pred_student - target
    difference_teacher = pred_teacher - target
    inter1 = square(difference_student)
    inter2 = square(difference_teacher)
    student = sum(inter1, axis=2)
    teacher = sum(inter2, axis=2)
    print('Student :',student)
    print('Teacher',teacher)

    batch, nb_box = target.shape[0], target.shape[1]
    print('batch, nb_box', batch, nb_box)
    result = zeros((batch, nb_box))
    print(result)
    for i in range(batch):
        for j in range(nb_box):
            if student[i][j] + 0.5 > teacher[i][j]:
                result[i][j] = student[i][j]
            else:
                result[i][j] = 0

    result = mean(result, axis=1)
    print(result)