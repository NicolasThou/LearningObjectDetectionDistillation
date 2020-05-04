from __future__ import print_function
# Demo mode uses the validation dataset for training, which is smaller and faster to train.
demo = True
log_interval = 100

# Options are imperative or hybrid. Use hybrid for better performance.
mode = 'hybrid'

# training hyperparameters
batch_size = 256
if demo:
    epochs = 5
    learning_rate = 0.02
    wd = 0.002
else:
    epochs = 40
    learning_rate = 0.05
    wd = 0.002

# the class weight for hotdog class to help the imbalance problem.
positive_class_weight = 5


import logging
logging.basicConfig(level=logging.INFO)
import os
import time
from collections import OrderedDict
import skimage.io as io

import mxnet as mx
from mxnet.test_utils import download
mx.random.seed(127)

# setup the contexts; will use gpus if avaliable, otherwise cpu
gpus = mx.test_utils.list_gpus()
contexts = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]

dataset_files = {'train': ('not_hotdog_train-e6ef27b4.rec', '0aad7e1f16f5fb109b719a414a867bbee6ef27b4'),
                 'validation': ('not_hotdog_validation-c0201740.rec', '723ae5f8a433ed2e2bf729baec6b878ac0201740')}


if demo:
    training_dataset, training_data_hash = dataset_files['validation']
else:
    training_dataset, training_data_hash = dataset_files['train']

validation_dataset, validation_data_hash = dataset_files['validation']

def verified(file_path, sha1hash):
    import hashlib
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)
    matched = sha1.hexdigest() == sha1hash
    if not matched:
        logging.warn('Found hash mismatch in file {}, possibly due to incomplete download.'
                     .format(file_path))
    return matched

url_format = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/{}'
if not os.path.exists(training_dataset) or not verified(training_dataset, training_data_hash):
    logging.info('Downloading training dataset.')
    download(url_format.format(training_dataset),
             overwrite=True)
if not os.path.exists(validation_dataset) or not verified(validation_dataset, validation_data_hash):
    logging.info('Downloading validation dataset.')
    download(url_format.format(validation_dataset),
             overwrite=True)


# load dataset
train_iter = mx.io.ImageRecordIter(path_imgrec=training_dataset,
                                   min_img_size=256,
                                   data_shape=(3, 224, 224),
                                   rand_crop=True,
                                   shuffle=True,
                                   batch_size=batch_size,
                                   max_random_scale=1.5,
                                   min_random_scale=0.75,
                                   rand_mirror=True)
val_iter = mx.io.ImageRecordIter(path_imgrec=validation_dataset,
                                 min_img_size=256,
                                 data_shape=(3, 224, 224),
                                 batch_size=batch_size)

from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

# get pretrained squeezenet
net = models.squeezenet1_1(pretrained=True, prefix='deep_dog_', ctx=contexts)
# hot dog happens to be a class in imagenet.
# we can reuse the weight for that class for better performance
# here's the index for that class for later use
imagenet_hotdog_index = 713

deep_dog_net = models.squeezenet1_1(prefix='deep_dog_', classes=2)
deep_dog_net.collect_params().initialize(ctx=contexts)
deep_dog_net.features = net.features
print(deep_dog_net)

from skimage.color import rgba2rgb

def classify_hotdog(net, url, contexts):
    I = io.imread(url)
    if I.shape[2] == 4:
        I = rgba2rgb(I)
    image = mx.nd.array(I).astype(np.uint8)
    plt.subplot(1, 2, 1)
    plt.imshow(image.asnumpy())
    image = mx.image.resize_short(image, 256)
    image, _ = mx.image.center_crop(image, (224, 224))
    plt.subplot(1, 2, 2)
    plt.imshow(image.asnumpy())
    image = mx.image.color_normalize(image.astype(np.float32)/255,
                                     mean=mx.nd.array([0.485, 0.456, 0.406]),
                                     std=mx.nd.array([0.229, 0.224, 0.225]))
    image = mx.nd.transpose(image.astype('float32'), (2,1,0))
    image = mx.nd.expand_dims(image, axis=0)
    out = mx.nd.SoftmaxActivation(net(image.as_in_context(contexts[0])))
    print('Probabilities are: '+str(out[0].asnumpy()))
    result = np.argmax(out.asnumpy())
    outstring = ['Not hotdog!', 'Hotdog!']
    print(outstring[result])

classify_hotdog(deep_dog_net, '../img/real_hotdog.jpg', contexts)

# let's examine the output layer and find the last conv layer
print(net.output)

# the last conv layer is the second layer
pretrained_conv_params = net.output[0].params

# weights can then be found from the above parameter dict
pretrained_weight_param = pretrained_conv_params.get('weight')
pretrained_bias_param = pretrained_conv_params.get('bias')

# next, we locate the right slice that we're interested in.
hotdog_w = mx.nd.split(pretrained_weight_param.data(ctx=contexts[0]),
                       1000, axis=0)[imagenet_hotdog_index]
hotdog_b = mx.nd.split(pretrained_bias_param.data(ctx=contexts[0]),
                       1000, axis=0)[imagenet_hotdog_index]

# our classifier is for two classes. here, we reuse the hotdog class weight,
# and randomly initialize the 'not hotdog' class.
new_classifier_w = mx.nd.concat(mx.nd.random_normal(shape=hotdog_w.shape, scale=0.02, ctx=contexts[0]),
                                hotdog_w,
                                dim=0)
new_classifier_b = mx.nd.concat(mx.nd.random_normal(shape=hotdog_b.shape, scale=0.02, ctx=contexts[0]),
                                hotdog_b,
                                dim=0)

# finally, we initialize the parameter buffers and set the values.
# since classifier is a HybridSequential/Sequential, the following
# takes the zero-indexed 1-st layer of the classifier
final_conv_layer_params = deep_dog_net.output[0].params
final_conv_layer_params.get('weight').set_data(new_classifier_w)
final_conv_layer_params.get('bias').set_data(new_classifier_b)




# return metrics string representation
def metric_str(names, accs):
    return ', '.join(['%s=%f'%(name, acc) for name, acc in zip(names, accs)])
metric = mx.metric.create(['acc', 'f1'])

import mxnet.gluon as gluon
from mxnet.image import color_normalize

def evaluate(net, data_iter, ctx):
    data_iter.reset()
    for batch in data_iter:
        data = color_normalize(batch.data[0]/255,
                               mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1)),
                               std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1)))
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    out = metric.get()
    metric.reset()
    return out

import mxnet.autograd as autograd


def train(net, train_iter, val_iter, epochs, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': wd})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    best_f1 = 0
    val_names, val_accs = evaluate(net, val_iter, ctx)
    logging.info('[Initial] validation: %s'%(metric_str(val_names, val_accs)))
    for epoch in range(epochs):
        tic = time.time()
        train_iter.reset()
        btic = time.time()
        for i, batch in enumerate(train_iter):
            # the model zoo models expect normalized images
            data = color_normalize(batch.data[0]/255,
                                   mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1)),
                                   std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1)))
            data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with autograd.record():
                for x, y in zip(data, label):
                    z = net(x)
                    # rescale the loss based on class to counter the imbalance problem
                    L = loss(z, y) * (1+y*positive_class_weight)/positive_class_weight
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                for L in Ls:
                    L.backward()
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if log_interval and not (i+1)%log_interval:
                names, accs = metric.get()
                logging.info('[Epoch %d Batch %d] speed: %f samples/s, training: %s'%(
                               epoch, i, batch_size/(time.time()-btic), metric_str(names, accs)))
            btic = time.time()

        names, accs = metric.get()
        metric.reset()
        logging.info('[Epoch %d] training: %s'%(epoch, metric_str(names, accs)))
        logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        val_names, val_accs = evaluate(net, val_iter, ctx)
        logging.info('[Epoch %d] validation: %s'%(epoch, metric_str(val_names, val_accs)))

        if val_accs[1] > best_f1:
            best_f1 = val_accs[1]
            logging.info('Best validation f1 found. Checkpointing...')
            net.save_parameters('deep-dog-%d.params'%(epoch))

if mode == 'hybrid':
    deep_dog_net.hybridize()
if epochs > 0:
    deep_dog_net.collect_params().reset_ctx(contexts)
    train(deep_dog_net, train_iter, val_iter, epochs, contexts)