from __future__ import absolute_import
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, gluon, nd
import random
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like
from mxnet.ndarray import *

""" ====================================== Classification Part ==================================== """


class DistillationLoss(gluon.HybridBlock):
    """SoftmaxCrossEntrolyLoss with Teacher model prediction

    L = (1 - weight) L_hard + weight. L_soft

    Parameters
    ----------
    temperature : float, default 1
        The temperature parameter to soften teacher prediction.
    hard_weight : float, default 0.5
        The weight for loss on the one-hot label.
    sparse_label : bool, default True
        Whether the one-hot label is sparse.
    """
    def __init__(self, temperature=1, hard_weight=0.5, sparse_label=True, **kwargs):
        super(DistillationLoss, self).__init__(**kwargs)
        self._temperature = temperature
        self._hard_weight = hard_weight
        with self.name_scope():
            self.soft_loss = ClassWeightCrossEntropy(sparse_label=False, **kwargs)
            self.hard_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label, **kwargs)

    def hybrid_forward(self, F, pred, label, soft_target):
        # pylint: disable=unused-argument
        """Compute loss"""
        label = _reshape_like(F, label, soft_target)
        pred = _reshape_like(F, pred, label)
        soft_loss = self.soft_loss(pred, soft_target)
        hard_loss = self.hard_loss(pred, label)
        return (1 - self._hard_weight) * soft_loss + self._hard_weight * hard_loss


class ClassWeightCrossEntropy(gluon.HybridBlock):
    """ClassWeightCrossEntrolyLoss with Teacher model prediction

    L_soft = -Sum(wi Pt.log(Ps))

    Parameters
    ----------

    """
    def __init__(self, **kwargs):
        super(ClassWeightCrossEntropy, self).__init__(**kwargs)


    def hybrid_forward(self, F, pred_student_confidence_score, soft_target):
        # pylint: disable=unused-argument
        """
        Compute loss
        pred_student_confidence_score and soft_target supposed to be in the same shape

        soft_target : dim [batch size, number of boxes, 1] one probability

        """

        soft_target = _reshape_like(F, soft_target, pred)

        return


""" ====================================== Regression Part ==================================== """


class RegressionLoss(gluon.HybridBlock):
    """
    RegressionLoss with Teacher model prediction

    L = L_smoothL1 + weight. L_BoundedRegression

    Parameters
    ----------
    hard_weight : float, default 0.5
        The weight for loss on bounded regression
    """
    def __init__(self, hard_weight=0.5, sparse_label=True, **kwargs):
        super(RegressionLoss, self).__init__(**kwargs)
        self._hard_weight = hard_weight
        with self.name_scope():
            self.lsmooth_loss = SmoothL1Loss(**kwargs)
            self.lbounded_loss = TeacherBoundedRegressionLoss(**kwargs)

    def hybrid_forward(self, F, pred_student, target, pred_teacher):
        """Compute loss"""
        target = _reshape_like(F, target, pred_student)
        pred_teacher = _reshape_like(F, pred_teacher, target)
        lsmooth_loss = self.lsmooth_loss(pred_student, target)
        lbounded_loss = self.lbounded_loss(pred_student, target, pred_teacher)
        return lsmooth_loss + self._hard_weight * lbounded_loss


class TeacherBoundedRegressionLoss(Loss):
    """

    Calculates Teacher Bounded Regression Loss
    L_BoundedRegression = Norme_L2(Rs - y)^2 or 0

    Inputs:
        - pred_student : prediction tensor with arbitrary shape
        - target : target tensor with the same size as pred.
        - pred_teacher : prediction tensor with arbitrary shape

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, m=0.2, **kwargs):
        super(TeacherBoundedRegressionLoss, self).__init__(m=0.2, **kwargs)
        self._m = m

    def hybrid_forward(self, F, pred_student, target, pred_teacher):
        """
        - difference between coordinate of predicted box and target box
        - square for each coordinate
        - sum each coordinate per boxe
        - assign the value according to the teacher bounding regression (Cf report)
        - mean per batch

        """
        target = _reshape_like(F, target, pred_student)
        target = _reshape_like(F, target, pred_teacher)
        difference_student = pred_student - target
        difference_teacher = pred_teacher - target
        inter1 = square(difference_student)
        inter2 = square(difference_teacher)
        student = sum(inter1, axis=2)
        teacher = sum(inter2, axis=2)


        batch, nb_box = target.shape[0], target.shape[1]
        result = zeros((batch, nb_box))
        for i in range(batch):
            for j in range(nb_box):
                if student[i][j] + self._m > teacher[i][j]:
                    result[i][j] = student[i][j]
                else:
                    result[i][j] = 0

        result = mean(result, axis=1)
        return result


class SmoothL1Loss(Loss):
    """

    Calculates smoothed L1 loss

    L_smootL1 = Sum(smooth(prediction_coordinate - target_coordinate))

    Parameters
    ----------
    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **target**: target tensor with the same size as pred.

    Outputs:
        - **loss**: loss tensor with shape (batch_size,)

    """

    def __init__(self, **kwargs):
        super(SmoothL1Loss, self).__init__(**kwargs)


    def hybrid_forward(self, F, pred_student, target):
        """
        pred : prediction offset : dim [Batch Size, Number of Boxes, 4]
        target : box targets:  dim [Batch Size, Number of Boxes, 4]
        """
        target = _reshape_like(F, target, pred_student)
        difference = pred_student - target
        inter = smooth_l1(difference, scalar=1)
        inter2 = sum(inter, axis=2)
        result = mean(inter2, axis=1)

        return result