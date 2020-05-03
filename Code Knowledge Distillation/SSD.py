import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import nd
from mxnet.ndarray.contrib import MultiBoxPrior
from mxnet.gluon import nn

n = 40
# shape: of the input example batch x channel x height x weight
x = nd.random_uniform(shape=(1, 3, n, n))
y = MultiBoxPrior(x, sizes=[.5, .25, .1], ratios=[1, 2, .5])

# each pixel (40x40 pixels) have 5 anchor boxes with 4 coordinates
boxes = y.reshape((n, n, 5, 4))

# We can visualize all anchor boxes generated for one pixel on a certain size feature map
def box_to_rect(box, color, linewidth=3):
    """convert an anchor box to a matplotlib rectangle"""
    box = box.asnumpy()
    return plt.Rectangle(
        (box[0], box[1]), (box[2]-box[0]), (box[3]-box[1]),
        fill=False, edgecolor=color, linewidth=linewidth)

colors = ['blue', 'green', 'red', 'black', 'magenta']
# display an image, here is all white, (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
plt.imshow(nd.ones((n, n, 3)).asnumpy())
# take all the anchor boxes for the pixel 20x20
anchors = boxes[20, 20, :, :]
#  anchors.shape[0] equal to 5 (number of anchor boxes)
for i in range(anchors.shape[0]):
    #  draw or add a patch on the figure, and anchors[i, :] * n to rescale the coordinates
    #  The coordinate values of the  𝑥  and  𝑦  axis are divided by the width and height of the image,
    #  respectively, so the value range is between 0 and 1.
    plt.gca().add_patch(box_to_rect(anchors[i, :] * n, colors[i]))
plt.show()

"""
For each anchor box, we want to predict the associated class label. We make this prediction 
by using a convolution layer. 
We choose a kernel of size 3×3 with padding size (1,1) so that the output will have the same width and height 
as the input. 
The confidence scores for the anchor box class labels are stored in channels.

"""

print("======================== Class Predictor =======================")

def class_predictor(num_anchors, num_classes):
    # nn.Conv2D creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
    # num_anchors * (num_classes + 1) is the number of filter/kernel, the number of output channel
    return nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)

cls_pred = class_predictor(5, 10)
cls_pred.initialize()
# example of input, 2 batch size, 3 channel
x = nd.zeros((2, 3, 20, 20))
# we pass each filter on this input, and the shape will be (2, 5 * 11, 20, 20)
print('Class prediction', cls_pred(x).shape)
print(cls_pred(x))

print("===============================================")
print("=================== Regression Predictor ============================")
# Predict the Anchor Boxes

def box_predictor(num_anchors):
    """return a layer to predict delta locations"""
    return nn.Conv2D(num_anchors * 4, 3, padding=1)

box_pred = box_predictor(10)
box_pred.initialize()
x = nd.zeros((2, 3, 20, 20))
print('Box prediction', box_pred(x).shape)

print("===============================================")
print("======================= Down Sample ========================")
"""
Convolution -> Batch Normalization -> ReLU -> Pooling layer to downsample the feature by half
"""
def down_sample(num_filters):
    """stack two Conv-BatchNorm-Relu blocks and then a pooling layer
    to halve the feature size"""
    out = nn.HybridSequential()
    for _ in range(2):
        out.add(nn.Conv2D(num_filters, 3, strides=1, padding=1))
        out.add(nn.BatchNorm(in_channels=num_filters))
        out.add(nn.Activation('relu'))
    out.add(nn.MaxPool2D(2))
    return out

blk = down_sample(10)
blk.initialize()
x = nd.zeros((2, 3, 20, 20))
print('Before', x.shape, 'after', blk(x).shape)

print("===============================================")
print("====================== Example Multi layer prediction =========================")

"""
A key property of SSD is that predictions are made at multiple layers with shrinking spatial size. 
Thus, we have to handle predictions from multiple feature layers. 
One idea is to concatenate them along convolutional channels, with each one predicting a 
corresponding value (class or box) for each default anchor.
 
We give class predictor as an example,  and box predictor follows the same rule.
"""

# a certain feature map with 20x20 spatial shape
feat1 = nd.zeros((2, 8, 20, 20))
print('Feature map 1', feat1.shape)

cls_pred1 = class_predictor(5, 10)
cls_pred1.initialize()
y1 = cls_pred1(feat1)

box_pred1 = box_predictor(5)
box_pred1.initialize()
y_reg = box_pred1(feat1)

print('Class prediction for feature map 1', y1.shape)
print('Reg prediction for feature map 1', y_reg.shape)

# down-sample
ds = down_sample(16)
ds.initialize()
feat2 = ds(feat1)
print('Feature map 2', feat2.shape)

cls_pred2 = class_predictor(3, 10)
cls_pred2.initialize()
y2 = cls_pred2(feat2)
print('Class prediction for feature map 2', y2.shape)

box_pred2 = box_predictor(3)
box_pred2.initialize()
y_reg2 = box_pred2(feat2)
print('Reg prediction for feature map 2', y_reg2.shape)


def flatten_prediction(pred):
    return nd.flatten(nd.transpose(pred, axes=(0, 2, 3, 1)))

def concat_predictions(preds):
    return nd.concat(*preds, dim=1)


flat_y1 = flatten_prediction(y1)
print('Flatten class prediction 1', flat_y1.shape)
flat_y2 = flatten_prediction(y2)
print('Flatten class prediction 2', flat_y2.shape)
print('Concat class predictions', concat_predictions([flat_y1, flat_y2]).shape)


flat_y_reg = flatten_prediction(y_reg)
print('Flatten reg prediction 1', flat_y_reg.shape)
flat_y_reg2 = flatten_prediction(y_reg2)
print('Flatten reg prediction 2', flat_y_reg2.shape)
print('Concat reg predictions', concat_predictions([flat_y_reg, flat_y_reg2]).shape)

print("===============================================")
print("====================== Extract Features =========================")

def body():
    """return the body network"""
    out = nn.HybridSequential()
    for nfilters in [16, 32, 64]:
        out.add(down_sample(nfilters))
    return out

bnet = body()
bnet.initialize()
x = nd.zeros((2, 3, 256, 256))
print('Body network', [y.shape for y in bnet(x)])

print("================================================================================")
print("====================== Create the Object Detection Network =========================")


# Architecture

def toy_ssd_model(num_anchors, num_classes):
    """return SSD modules"""
    downsamples = nn.Sequential()
    class_preds = nn.Sequential()
    box_preds = nn.Sequential()

    downsamples.add(down_sample(128))
    downsamples.add(down_sample(128))
    downsamples.add(down_sample(128))

    for scale in range(5):
        class_preds.add(class_predictor(num_anchors, num_classes))
        box_preds.add(box_predictor(num_anchors))

    return body(), downsamples, class_preds, box_preds

print(toy_ssd_model(5, 2))

# Forward

def toy_ssd_forward(x, body, downsamples, class_preds, box_preds, sizes, ratios):
    # extract feature with the body network
    x = body(x)

    # for each scale, add anchors, box and class predictions,
    # then compute the input to next scale
    default_anchors = []
    predicted_boxes = []
    predicted_classes = []

    for i in range(5):
        default_anchors.append(MultiBoxPrior(x, sizes=sizes[i], ratios=ratios[i]))
        predicted_boxes.append(flatten_prediction(box_preds[i](x)))
        predicted_classes.append(flatten_prediction(class_preds[i](x)))
        if i < 3:
            x = downsamples[i](x)
        elif i == 3:
            # simply use the pooling layer
            x = nd.Pooling(x, global_pool=True, pool_type='max', kernel=(4, 4))

    return default_anchors, predicted_classes, predicted_boxes

# Class of the object detection neural network

class ToySSD(gluon.Block):
    def __init__(self, num_classes, **kwargs):
        super(ToySSD, self).__init__(**kwargs)
        # anchor box sizes for 4 feature scales
        self.anchor_sizes = [[.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        # anchor box ratios for 4 feature scales
        self.anchor_ratios = [[1, 2, .5]] * 5
        self.num_classes = num_classes

        with self.name_scope():
            self.body, self.downsamples, self.class_preds, self.box_preds = toy_ssd_model(4, num_classes)

    def forward(self, x):
        default_anchors, predicted_classes, predicted_boxes = toy_ssd_forward(x, self.body, self.downsamples,
            self.class_preds, self.box_preds, self.anchor_sizes, self.anchor_ratios)
        # we want to concatenate anchors, class predictions, box predictions from different layers
        anchors = concat_predictions(default_anchors)
        box_preds = concat_predictions(predicted_boxes)
        class_preds = concat_predictions(predicted_classes)
        # it is better to have class predictions reshaped for softmax computation
        class_preds = nd.reshape(class_preds, shape=(0, -1, self.num_classes + 1))

        return anchors, class_preds, box_preds


# Test
# instantiate a ToySSD network with 10 classes
net = ToySSD(2)
net.initialize()
x = nd.zeros((1, 3, 256, 256))
default_anchors, class_predictions, box_predictions = net(x)
print('Outputs:', 'anchors', default_anchors.shape, 'class prediction', class_predictions.shape, 'box prediction', box_predictions.shape)