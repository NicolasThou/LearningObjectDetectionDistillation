import mxnet as mx
from mxnet import nd
from mxnet.gluon.data import DataLoader
from gluoncv import model_zoo
from gluoncv.data.transforms import presets
from gluoncv import data
from gluoncv.utils import viz
import matplotlib
import matplotlib.pyplot as plt


def inverse_transformation(image):
    image = image.transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
    image = (image * 255).asnumpy()
    return image


dataset = data.COCODetection(splits=['instances_val2017'])

model = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=False)
model.load_parameters('params/model_1399.params', ignore_extra=True)

distil_model = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=False)
distil_model.load_parameters('params/model_distil_1399.params', ignore_extra=True)

transform = presets.rcnn.FasterRCNNDefaultValTransform()
data_loader = DataLoader(dataset.transform(transform), batch_size=1, shuffle=True, last_batch='keep')

matplotlib.use('TkAgg')
for batch_idx, batch in enumerate(data_loader):
    if batch_idx > 100:
        break
    for data_img, data_label, _ in zip(*batch):
        ids, scores, bboxes, _ = model(data_img.expand_dims(0))
        distil_ids, distil_scores, distil_bboxes, _ = distil_model(data_img.expand_dims(0))

        data_label = data_label.expand_dims(0)
        gt_label = data_label[:, :, 4:5]
        gt_box = data_label[:, :, :4]

        # we can change the threshold to display bounding boxes with lower scores
        # smaller training will result in smaller confidence score
        img = inverse_transformation(data_img)  # inverse transformation to get image
        viz.plot_bbox(img, gt_box[0], mx.ndarray.ones(gt_box[0].shape[0]), gt_label[0], class_names=dataset.CLASSES)
        viz.plot_bbox(img, bboxes[0], scores[0], ids[0], class_names=dataset.CLASSES, thresh=0.3)
        viz.plot_bbox(img, distil_bboxes[0], distil_scores[0], distil_ids[0], class_names=dataset.CLASSES, thresh=0.3)
        plt.show()
