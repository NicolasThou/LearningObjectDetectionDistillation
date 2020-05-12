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
import numpy as np
matplotlib.use('TkAgg')


def extract_boxes(scores, labels):
    assert len(scores) == len(labels)

    idx = []
    for i in range(len(scores)):
        if i in [45000, 45001, 45002, 46000, 50000, 55000]:
            print(f'{i} | {scores[i].asnumpy().item()}')

        if scores[i] < 0.1 or labels[i] < 0:
            continue

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
# train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])

teacher = model_zoo.get_model('faster_rcnn_resnet101_v1d_coco', pretrained=True)
# teacher = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)

distil_student = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=False)
distil_student.initialize()
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
# disable horizontal flip so image given to teacher match image from batch
train_transform = presets.rcnn.FasterRCNNDefaultTrainTransform(net=student, flip_p=0)

# Return images, labels, rpn_cls_targets, rpn_box_targets, rpn_box_masks loosely
batchify_fn = FasterRCNNTrainBatchify(student)

batch_size = 32
train_loader = DataLoader(train_dataset.transform(train_transform), batch_size, shuffle=False,
                          batchify_fn=batchify_fn, last_batch='rollover', num_workers=0)

trainer = Trainer(student.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 0.0005, 'momentum': 0.9})
distil_trainer = Trainer(distil_student.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 0.0005, 'momentum': 0.9})

temp = 3
teacher.temperature = temp
writer = SummaryWriter()
for batch_idx, batch in enumerate(train_loader):
    if batch_idx > 0:
        break
    with autograd.record():
        loss = []
        for image_idx, (data_batch, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks) in enumerate(zip(*batch)):
            # teacher predictions
            with autograd.pause():
                teacher_img, teacher_label = train_dataset[batch_size * batch_idx + image_idx]
                transformed_img, original_teacher_img = presets.rcnn.transform_test(teacher_img)
                ids, scores, bboxes, teacher_prob = teacher(transformed_img)

            # teacher bounding boxes and labels
            idx = extract_boxes(scores[0], ids[0])
            bboxes = bboxes[:, idx, :]
            scores = scores[:, idx, :]
            ids = ids[:, idx, :]

            # dataset bounding boxes and labels
            label = label.expand_dims(0)
            gt_label = label[:, :, 4:5]
            gt_box = label[:, :, :4]

            # train_image = inverse_transformation(data_batch)  # inverse transformation to get image
            # viz.plot_bbox(original_teacher_img, gt_box[0], mx.ndarray.ones(gt_box[0].shape[0]), gt_label[0], class_names=teacher.classes)
            # viz.plot_bbox(original_teacher_img, bboxes[0], scores[0], ids[0], class_names=teacher.classes)
            # viz.plot_bbox(train_image, gt_box[0], mx.ndarray.ones(gt_box[0].shape[0]), gt_label[0], class_names=teacher.classes)
            # viz.plot_bbox(train_image, bboxes[0], scores[0], ids[0], class_names=teacher.classes)
            # viz.plot_bbox(teacher_img, teacher_label[:, :4], mx.ndarray.ones(teacher_label.shape[0]), teacher_label[:, 4], class_names=teacher.classes)
            # plt.show()

            # networks forward
            cls_preds_soft, box_preds_soft, _, _, _, _, _, _, cls_targets_soft, box_targets_soft, box_masks_soft, _ = distil_student(data_batch.expand_dims(0), bboxes, ids)
            cls_preds_hard, box_preds_hard, _, _, _, rpn_score_hard, rpn_box_hard, _, cls_targets_hard, box_targets_hard, box_masks_hard, _ = distil_student(data_batch.expand_dims(0), gt_box, gt_label)
            cls_preds, box_preds, _, _, _, rpn_score, rpn_box, _, cls_targets, box_targets, box_masks, _ = student(data_batch.expand_dims(0), gt_box, gt_label)

            # losses of rpn
            rpn_score = rpn_score.squeeze(axis=-1)
            num_rpn_pos = (rpn_cls_targets >= 0).sum()
            rpn_loss1 = rpn_cls_loss(rpn_score, rpn_cls_targets, rpn_cls_targets >= 0) * rpn_cls_targets.size / num_rpn_pos
            rpn_loss2 = rpn_box_loss(rpn_box, rpn_box_targets, rpn_box_masks) * rpn_box.size / num_rpn_pos

            rpn_score_hard = rpn_score_hard.squeeze(axis=-1)
            num_rpn_pos_hard = (rpn_cls_targets >= 0).sum()
            rpn_loss1_hard = rpn_cls_loss(rpn_score_hard, rpn_cls_targets, rpn_cls_targets >= 0) * rpn_cls_targets.size / num_rpn_pos_hard
            rpn_loss2_hard = rpn_box_loss(rpn_box_hard, rpn_box_targets, rpn_box_masks) * rpn_box_hard.size / num_rpn_pos_hard

            # losses of rcnn
            num_rcnn_pos = (cls_targets >= 0).sum()
            rcnn_loss1 = rcnn_cls_loss(cls_preds, cls_targets,cls_targets >= 0) * cls_targets.size / cls_targets.shape[0] / num_rcnn_pos
            rcnn_loss2 = rcnn_box_loss(box_preds, box_targets, box_masks) * box_preds.size / box_preds.shape[0] / num_rcnn_pos

            num_rcnn_pos_hard = (cls_targets_hard >= 0).sum()
            rcnn_loss1_hard = rcnn_cls_loss(cls_preds_hard, cls_targets_hard, cls_targets_hard >= 0) * cls_targets_hard.size / cls_targets_hard.shape[0] / num_rcnn_pos_hard

            num_rcnn_pos_soft = (cls_targets_soft >= 0).sum()
            rcnn_loss1_soft = rcnn_cls_loss(cls_preds_soft/temp, cls_targets_soft, cls_targets_soft >= 0) * cls_targets_soft.size / cls_targets_soft.shape[0] / num_rcnn_pos_soft
            rcnn_loss2_soft = (rcnn_box_loss(box_preds_soft, box_targets_soft, box_masks_soft) * box_preds_soft.size / box_preds_soft.shape[0] / num_rcnn_pos_soft) + teacher_bounded_regression_loss(box_targets, box_targets_soft, box_preds_soft)

            # compute backward gradient
            autograd.backward([rpn_loss1, rpn_loss2, rcnn_loss1, rcnn_loss2, rpn_loss1_hard, rpn_loss2_hard, rcnn_loss1_hard, rcnn_loss1_soft, rcnn_loss2_soft])

            # add losses
            mu = 0.5
            loss.append([rpn_loss1.asnumpy().item(), rpn_loss2.asnumpy().item(),
                         rcnn_loss1.asnumpy().item(), rcnn_loss2.asnumpy().item(),
                         rpn_loss1_hard.asnumpy().item(), rpn_loss2_hard.asnumpy().item(),
                         mu*rcnn_loss1_hard.asnumpy().item() + (1-mu)*rcnn_loss1_soft.asnumpy().item(),
                         rcnn_loss2_soft.asnumpy().item()])
    # make an optimization step
    trainer.step(batch_size)
    distil_trainer.step(batch_size)

    if ((batch_idx+1) % 10) == 0:
        student.save_parameters(f'params/model_{batch_idx%10}.params')
        distil_student.save_parameters(f'params/model_distil_{batch_idx%10}.params')

    # update loss graphs
    loss = np.mean(loss, axis=0).tolist()
    print(f'batch : {batch_idx} | loss no distil : {sum(loss[:4])} | loss distil : {sum(loss[4:])}')

    writer.add_scalar('RPN/Classification loss no distil', loss[0], batch_idx)
    writer.add_scalar('RPN/Classification loss distil', loss[4], batch_idx)

    writer.add_scalar('RPN/Regression loss no distil', loss[1], batch_idx)
    writer.add_scalar('RPN/Regression loss distil', loss[5], batch_idx)

    writer.add_scalar('RCNN/Classification loss no distil', loss[2], batch_size)
    writer.add_scalar('RCNN/Classification loss distil', loss[6], batch_size)

    writer.add_scalar('RCNN/Regression loss no distil', loss[3], batch_size)
    writer.add_scalar('RCNN/Regression loss distil', loss[7], batch_size)
