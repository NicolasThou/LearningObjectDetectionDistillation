from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils


if __name__ == '__main__':

    """
    =========================== pseudo code ============================
    
    1) Custom the pre-trained Faster R-CNN in order to have soft-label which means that for the output layer
    we need to put the activation function : softmax with Temperature
    
    2) Build the distill model, rely on fast R-CNN model or any object detection model but with less
    parameter that is to say : tiny wrt the number of neurons per layer (width for the teacher)
    
    3) Custom the backward pass, and the losses function :
    Weighted Cross Entropy Loss : Classification soft label
    Bounded Regression Loss : Regression soft label
    Ground Truth : SoftMax & SmoothL1 Loss
    
    4) Organized the data : we have for each unlabelled data/image, take the soft label of the Faster R-CNN
    i-e Regression & Classification soft label, and also a GroundTruth Label. 
    
    5)
    The distill model will learn how to predict GroundTruth Label and the prediction of the Teacher
    We have to merge the image into :
    [image,  GroundTruth Label, Regression & Classification soft label]

    
    6) Training of the Distill Model
    
    ---------------------------------------------------------------------------------------------------------
    
    # step 1 OK
    Fast_RCNN = get_model(activation_function_output = custom_function())
    
    
    # step 2 and step 3
    backward = custom_backward()
    distill_model = build_distill_model(output_dim = x, depth = 6, number_neurons_per_layer = 4, loss = custom_loss())
    
    
    # step 4
    training_set = get_data() # data = [image, Regression truth of the bounding box, GroundTruth Label]
    prediction = Fast_RCNN.predict(training_set)
    
    
    # step 5
    training_set_distill_model = merge(get_data(), prediction)
    
    # step 6
    disitill_model.train(training_set_distill_model)
    
    """

    
    print("hello world!")