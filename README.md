# Description
This project is based off on an AI model trained using a CNN deep learning architecture to predict sentiment or emotion of the person in the user given image or video.

This project uses Tkinter to provide users with a basic GUI to upload an image or a video.

# Dataset Information
The dataset used in the project is the Cohn-Kanade dataset acquired from [ck dataset](http://www.consortium.ri.cmu.edu/ckagree/)

The class labels in the above dataset are "anger", "contempt", "disgust", "fear", "happy", "sadness" and "surprise".

You could train your model on different datasets too if you want!! Just run ckplus.py to create your own model.

# Requirements

There are some general library requirements for the project and some which are specific to individual methods. The general requirements are as follows.

* `opencv`
* `matplotlib`
* `numpy`


The library requirements specific to some methods are:
* `tensorflow`
* `face_recognition`

# Usage
1.> If you want to create your own model, run ckplus.py first choosing the required dataset and train your model.
		1.1> You can change the architecture of the model in models.py to suit your needs.
				 Training ckplus.py will display a graph which will provide you with a general overview of the accuracy between train and test	 					cases.
				 
2.> Run run.py to be greeted with a GUI where you can choose to upload an image, a video or test the model live using a webcam.

# Credits
Ckplus dataset provided us with about 1000 images to analyze and train the model for emotion analysis, so kudos to them.
