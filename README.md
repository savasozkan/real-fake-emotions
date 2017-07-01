# Real-Fake-Emotions (Real versus Fake expressed emotions @ICCV17) by TUBITAK UZAY-METU 
This repo contains "TUBITAK UZAY-METU" submission for "Real versus Fake Expressed Emotions" challenge @ICCV2017. 

The proposed method consists of two main stages, feature extraction and classification. For the feature extraction step, we used the combination of Caffe emotion model (C/C++) and TensorFlow LSTM structure (Python). Also, this step highly depends on the internal code infrasture of TUBITAK UZAY which cannot be shared/distributed over this page. Therefore, two directories, namely "tfcode" and "caffecode", are only provided to analyse the code consistency by the organizers and guide the developers for further applications.

In the classification step, we trained a linear SVM (using LibSVM) with precomputed features from the first step. This part is also written in C/C++ and it can be built by running "make" on a ubuntu terminal (it should be x64). "feature.zip" contains the precomputed features and you can access the pretrained SVM models from "svmmodels" directory.

"main.cpp" is the main code for the classification step and it comprises three modes, train a SVM model, e.g. train("anger");, obtain the validation predictions, e.g. validate("anger");, and obtain the test predictions, e.g. test("anger");. By default, it can only execute to generate validation and test predictions, and print to the screen for each emotion type independently. In order to activate train model, you should remove the comment syntax in "void main()" function. 

We also provide our final predictions in "val.py". You can generate "test_prediction.pkl" and compare the consistency of predictions by running "svm" binary file.

In order to build and test the project, several steps need to be done.

* Extract "feature.zip" to the project directory.
* Execute "make" on command line to build the code.
* Run "svm" binary file to visualize the validation and test results. 


