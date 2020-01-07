# **Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview

In this project, an end-to-end learning for self driving car with Udacity's self-driving car simulator was build based on the convolution neural network(CNN) introduced by [NVIDIA](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The goal of this project is to train network to calculate the steering angle required for lane keeping from front camera image. 

|[LAKE TRACK](https://www.youtube.com/watch?v=Ac-268T252s&feature=youtu.be)           |
|:-------------------:|
![left][image6]       |

### The specific objectives

* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Details about files

This repository contains following files:
* `model.py` - This script used to create and train the model.
* `drive.py` - The script to drive the car.
* `model.h5` - The trained model.
* `README.md` - A report describes how to output the video.
The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

### Dependencies of current solution
This solution requires:
* Python
* Keras/Tensorflow 
* Numpy
* Simulator: [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)

## Data Preparation

### Data collection
To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here are the example images of track one from center, left, and right cameras:

|Left           |Center             |Right           |
|:-------------:|:-----------------:|:--------------:|
|![left][image1]|![center][image2]  |![right][image3]|

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn what to do if it gets off to the side of the road. Finally, one lap focusing on dirving smoothlt around curves were recorded.

### Data augumentation
For training, I used the following augumentation technique along with Python generator to boost the model:

* **Horizontal flip**. For every batch,I flip half of the frames horizontally and change the sign of the steering angle, thus yet increasing number of examples by a factor of 2.
* **Horizontal translation**. I translate image horizontally randomly with steering angle adjustment(0.002 per pixel shift), which is useful for difficult curve handling.
* **Random shadow**. I add a random vertical "shadow" by decreasing brightness of a frame slice, hoping to make the model invariant to actual shadows on the road.
* **Random brightness**. I randomly altered image brightness(lighter or darker)

Here some examples of the augmented images:

|![augumented image][image4]| ![augumented image][image7] |![augumented image][image8]|
|:-------------:|:-----------------:|:--------------:|

### Data preprocess
After the collection and augumentation process, I had 35916 data points. I then randomly shuffled the data set and put 20% of the data into a validation set.  The validation set helped determine if the model was over or under fitting. 

## Model

### Solution Design Approach

As the NVIDIA model is well documented, I was able to focus how to adjust the training images to produce the best result with some adjustments to the model to avoid overfitting and adding non-linearity to improve the prediction. I've added the following adjustments to the model.

* I added a Lambda layer to normalize input images, which can help to avoid saturation and make better gradient descent.
* I added a Cropping2D layer to crop input images so that the model won't be trained with the sky and the car front parts.  
* I've also included ELU for activation function for every convolution layer to introduce non-linearity.

### Model Architecture

The  [NVIDIA](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) model looks like as follows:

* Image normalization(Lambda layer and Cropping2D layer)
* Convolution: 5x5, filter: 24, strides: 2x2, subsampling: 2x2, activation: ELU
* Convolution: 5x5, filter: 36, strides: 2x2, subsampling: 2x2, activation: ELU
* Convolution: 5x5, filter: 48, strides: 2x2, subsampling: 2x2, activation: ELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
* Fully connected: neurons: 100
* Fully connected: neurons: 50
* Fully connected: neurons: 10
* Fully connected: neurons: 1 (output)

###  Training Strategy
I splitted the images into train and validation set in order to measure the performance at every epoch. Testing was done using the simulator. As for training,

* The model used mean squared error for the loss function to measure how close the model predicts to the given steering angle for each image.
* The model used an Adam optimizer, so the learning rate was not tuned manually

### Model Adjustment
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model based on  [NVIDIA](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) had a low mean squared error on the training set but a high mean squared error on the validation set.This implied that the model was overfitting. I tested this original model, the vehicle is able to drive autonomously around the track without leaving the road. 

|DATE             |BATCH SIZE    |EPOCHS        |LOSS(LAST EPOCH)     |VALIDATION LOSS(LAST EPOCH)    |TIME     |
|-----------------|-------------:|-------------:|--------------------:|-----------------:|--------------|
|2020-1-4         |32            |3             |8.7523e-04           |0.0264            |8645s         |
|2020-1-6         |128           |5             |1.7648e-04           |0.0232            |30229s        |

However, to combat the overfitting, I added an additional dropout layer to avoid overfitting after the convolution layers.In the end, the model looks like as follows:

![model archetecture][image5]
A model summary is as follows:

|LAYTER             |TYPE           |OUTPUT SHAPE          |PARAM #     |CONNECTED TO               |
|:------------------|:--------------|:---------------------|-----------:|:--------------------------|
|lambda_1           |Lambda         |(None, 160, 320, 3)   |0           |lambda_input_1[0][0]       |
|cropping2d_1       |Cropping2D     |(None, 90, 320, 3)    |0           |lambda_1[0][0]             |
|convolution2d_1    |Convolution2D  |(None, 85, 315, 24)   |1824        |cropping2d_1[0][0]         |
|maxpooling2d_1     |MaxPooling2D   |(None, 43, 158, 24)   |0           |convolution2d_1[0][0]      |
|convolution2d_2    |Convolution2D  |(None, 39, 154, 36)   |21636       |maxpooling2d_1[0][0]       |
|maxpooling2d_2     |MaxPooling2D   |(None, 20, 77, 36)    |0           |convolution2d_2[0][0]      |
|convolution2d_3    |Convolution2D  |(None, 16, 73, 48)    |43248       |maxpooling2d_2[0][0]       |
|maxpooling2d_3     |MaxPooling2D   |(None, 8, 37, 48)     |0           |convolution2d_3[0][0]      |
|convolution2d_4    |Convolution2D  |(None, 6, 35, 64)     |27712       |maxpooling2d_3[0][0]       |
|convolution2d_5    |Convolution2D  |(None, 4, 33, 64)     |36928       |convolution2d_4[0][0]      |
|dropout_1          |Dropout        |(None, 4, 33, 64)     |0           |convolution2d_5[0][0]      |
|flatten_1          |Flatten        |(None, 8448)          |0           |dropout_1[0][0]            |
|dense_1            |Dense          |(None, 100)           |844900      |flatten_1[0][0]            |
|dense_2            |Dense          |(None, 50)            |5050        |dense_1[0][0]              |
|dense_3            |Dense          |(None, 10)            |510         |dense_2[0][0]              |
|dense_4            |Dense          |(None, 1)             |11          |dense_3[0][0]              |

## Results

### Run the pretrained model
Once the model has been saved, start up the Udacity self-driving simulator, choose a scene and press the Autonomous Mode button. Then, run the model with `drive.py` as follows:

```sh
python drive.py model.h5
```
The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.
### Saving a video of the autonomous agent

The model can drive the course without bumping into the side ways(click the link for full video).
* [The Lake Track](https://www.youtube.com/watch?v=Ac-268T252s&feature=youtu.be)

## Discussion

As can be seen from the full videos, the car remains onthe center of the road on Track One for most of the path, but waits a little too long to turnnear the dirt track (but still manages to nonetheless).  However, current model works poorly in Track Two(jungle track). The performance of this pipeline could be better improved by:

* collecting more data from Track Two(jungle track) to boost the model. 
* using pre-trained weightsfrom a different architecture (transfer learning) which is then fit to our application 
* using either a driving wheel or a joystick to better manipulate the car during manual modeand generate smoother steering and hence better steering angles.

## Reference

* [1] NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
* [2] Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim
* [3] Bojarski, Mariusz, et al. "End to end learning for self-driving cars." arXiv preprint arXiv:1604.07316 (2016).
* [4] Bojarski, Mariusz, et al. "Explaining how a deep neural network trained with end-to-end learning steers a car." arXiv preprint arXiv:1704.07911 (2017).


[//]: # (Image References)
[image1]: ./img/left_2020_01_04_21_56_30_542.jpg "left"
[image2]: ./img/center_2020_01_04_21_56_30_542.jpg "center"
[image3]: ./img/right_2020_01_04_21_56_30_542.jpg "right"
[image4]: ./img/augumented01.jpg "augumented image"
[image5]: ./img/model.png "model archatecture"
[image6]: ./img/run1.gif "run"
[image7]: ./img/augumented02.jpg "augumented image"
[image8]: ./img/augumented03.jpg "augumented image"