# Real-Fake-Emotions (Real versus Fake expressed emotions @ICCV17) by TUBITAK UZAY-METU 
This repo contains "TUBITAK UZAY-METU" submission for "Real versus Fake Expressed Emotions" challenge @ICCV2017. 

The proposed method consists of two main stages, feature extraction and classification. For the feature extraction step, we used the combination of a pretrained Caffe emotion model (C/C++) and TensorFlow LSTM structure (Python). Also, this step highly depends on internal code infrasture of TUBITAK UZAY which cannot be shared. Therefore, two directories, namely "tfcode" and "caffecode", are only provided to analyse the code consistency by the organizers as well as other developers to use this method.

In the classification step, we train a linear SVM (using LibSVM) with a precomputed features from the first step. This part is also written in C/C++ and it can be built by running "make" from a ubuntu terminal. "feature.zip" contains these precomputed features. You can also access pretrained SVM models from "svmmodels" directory.

For the challange, "val.py" generates "test_prediction.pkl" file for our predictions. 

In order to build and test the project, several steps need to be executed.

* Extract "feature.zip" to the project directory.
* Execute "make" on command line to build the code.
* Run "svm" file to visualize the validation and test results. 


