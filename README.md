	### Knowledge Distillation for Object Detection ##

In this folder you'll find the files to train and test object detection models using the Knowledge Distillation framework developped in [1].

First of all you need some dependancies to use the scripts. You have to install the following python library :
	- MXNet (pip install mxnet)
	- GluonCV (pip install --upgrade mxnet gluoncv)

Moroever, you need to install the COCO Object Detection dataset. You can do it with the script mscoco.py provided in the folder.

To use the predictions of the teacher you have to modify a file of the gluoncv library which is located at [$conda_environment_path]\Lib\site-packages\gluoncv\model_zoo\rcnn\faster_rcnn\faster_rcnn.py
We provide the modifed file in the library directory.

Then, when all the previous steps are done, you can train the models (a distilled and a standard) with the train.py file and test them with test.py.
For this do :
	- python train.py
	- python test.py

The models are stored in the folder 'params' and the loss plots in the runs folder (you can visualize them with tensorboard).

We provide two models already trained in : https://github.com/NicolasThou/LearningObjectDetectionDistillation

[1] Learning efficient object detection models with knowledge distillation, Guobin Chen et al.
