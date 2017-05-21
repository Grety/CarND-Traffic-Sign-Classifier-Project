#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/grety/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used some python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410 
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![Initial Classes Representation][initial_classes_representation.png]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided NOT to convert the images to grayscale because in my intuitive opinion this would provide more information to the model. However I've read the opposite, grayscale images tend to perform better for traffic signs recognition. Later afte rmy experiments I found it doesn't give significant impact, so in my final result I stay with full color images

Data normalization makes absolute sense. Translating data to have mean equal zero makes models train faster. I've experimented with different formulas/ My own first guess was to normalize eac himage individually (this provides a kind of auto brightness correction effect) and translate to `mean=0`. While playing with different model layers I found that a magical formula found in Internet sometimes works well `images / 255. * 0.8 + 0.1`.

Generation of additional data was a crucial point in proceeding to required accuracy threshold. While I examined some early works of students (perhaps from udacity) which I could find in the Internet, I learned that with the new given validation set (`validation.p` pickle file) dramatically changes the situation. And models that trained pretty well (I even had one of my own, that showed 97% validation accuracy 92% test, and 100% on additional images!) on auto-generated validation set (old version of this lab), failed miserably (validation accuracy below 80%) with the new predefined validation set. I suspect some trick hidden in that data, but I failed to locate it. Brief analysis revealed that number of classes match, images of the mentioned set have reasonable quality.
Nevertheless, only generating a large set of jittered images (after getting aquainted with a paper by Pierre Sermanet and Yann LeCun I learned that jittered data should exceed original data in training set) allowed to pass the 93% accuracy threshold. Also "leveling" classes representation yields good results. Which means if some class has less samples in the training set, extra jittered images should be generated for this class.

To add more data to the the data set, I used random effects over existing images which enlist rotation (at random angle in range -5; +5 degrees), translation, scaling, sharpening and brightening.

The following chart shows difference between the original data set and the augmented data set.

![Augemented Training Set][Augmented_training_set.png]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 7x7     	| 2x2 stride, valid padding, outputs 13x13x16 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 4x4x32 	|
| Flatten	      		| outputs 1x800									|
| Dropout	      		| pass probability 0.8							|
| Fully connected		| output 256   									|
| Fully connected		| output 128   									|
| Fully connected		| output 43										|

The basis of this model is LeNet-5 with 2 convolutional layers and 3 fully connected ones. However the question remains about the layer parameters.
The basic idea behind my structure was to build a kind of pyramid, where size of each next layer is smaller than the previous one. In my intuition this should prevent overfitting inside the model.
I've achieved decent results with the first version of this model and started variating one layer at a time to understand impact on the model. I learned that adding/moving/changing `keep_prob` of the dropout layer doesn't significantly influence the model. Same does adding max pool layers. So I avoided having the latter in my model.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ADAM optimizer. Although it is mentioned that this optimizer has dynamically adapted learning rate I failed to confirm this. Examination of private variable `optimizer._lr` didn't make me believe that learning rate changes with epochs. I tried to build dynamically decreasing learning rate on my own but lack of good python skills prevented that. I ended up with a fixed rate of 0.005 which works pretty well within 10 to 20 epochs.
Batch size was chosen to be 256 as it allowed to train the model on my PC with 20 GB RAM and GTX 750 Ti GPU. I consider that if I had more time and went deeper into raster image recognition I would try different sizes as I failed to build a more heavy model (deeper convolutions). My computer just hang up trying to train the model.
I slightly modified code provided in lecture to use as many epochs as neede for training model. Thus in code I define minimal number of epochs to run, and a maximum one. Once minimal number of epochs is trained the process would stopas soon as accuracy starts declining. This approach allowed me to quickly check variations of models a was experimenting with and switch to intensive learning on models that proved to be good.
Hyperparameters (mu and sigma, randomization parameters for weights initial values) turned out to be a crucial point for me. Occasionally setting sigma to 0.3 instead of 0.1 completely ruined the model. I spent 2 days to debug and find the problem. On realizing how important this parameter is I spent some time to find optimal value, which for me turned out to be 0.07.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Steps taken to achieve results presetned below are discussed in previous paragraphs. it took a while to make solution for the updated task on this lab.

My final model results were:
* validation set accuracy of 93.7%
* test set accuracy of 93.6%

The fact that validation and test accuracy is not much different speaks for good model fit.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

For alternative images I've chosen Defined Traffic Signs from Belgium dataset:

![Yield][DefinedTS/13/13.png] ![Stop][[DefinedTS/13/14.png] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I find it interesting to test the model against such clean images without any noise. Good accuracy on this set should be evidence of proper model structure and training. On my first model (without using predefined validation set) I achieved 100% prediction accuracy which assumes that model learned "proper" knowledge, the sign info and not the noise to be part of recognized objects.
However on updated model I achieved only 70% result. This should be further investigated as it definitely shows some flwas of the model. Luckily this is not a requiremnt to be met for submission

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|Image                   	| Prediction 				|
|:-------------------------:|:-------------------------:| 
|Yield [13]              	| Yield [13]               	|
|Stop [14]               	| Speed limit (70km/h) [4] 	|
|Bumpy road [22]         	| Speed limit (20km/h) [0] 	|
|Wild animals crossing [31]	| Wild animals crossing [31]|
|Traffic signals [26]    	| Traffic signals [26]     	|
|Pedestrians [27]        	| Double curve [21]        	|
|Bicycles crossing [29]  	| General caution [18]     	|
|Double curve [21]       	| Slippery road [23]       	|
|Double curve [21]       	| Double curve [21]        	|
|No entry [17]           	| No entry [17]            	|

The model was able to correctly guess 5 of the 10 traffic signs, which gives an accuracy of 50%. This is definitely bad performance for noiseless images.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Model was pretty sure about its predictions. All but one predictions had probability 0.93-1.0. However, Pedestrians sign (which performed the worst on validation and test sets) was predicted wrong with rather spread probalities

The top three soft max probabilities for Pedestrians were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .67         			| Road narrows on the right						| 
| .17     				| Right-of-way at the next intersection			|
| .16					| Pedestrians									|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


