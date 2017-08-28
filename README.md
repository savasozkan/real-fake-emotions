# Real-Fake-Emotions (Real versus Fake expressed emotions @ICCV17) by TUBITAK UZAY-METU 
This repo contains "TUBITAK UZAY-METU" submission for "Real versus Fake Expressed Emotions" challenge @ICCV2017. 

The proposed method consists of two main stages, feature extraction and classification. For the feature extraction step, we used the combination of Caffe emotion model (C/C++) and TensorFlow LSTM structure (Python). Also, this step highly depends on the internal code infrastructure of TUBITAK UZAY which cannot be shared/distributed over this page. Therefore, two directories, namely "tfcode" and "caffecode", are only provided to analyse the code consistency by the organizers and guide the developers for further applications.

In the classification step, we trained a linear SVM (using LibSVM) with precomputed features at the first step. This part is also written in C/C++ and it can be built by running "make" on Ubuntu terminal (it should be x64). "feature.zip" contains these precomputed features and you can access the pretrained SVM models from "svmmodels" directory.

"main.cpp" is the main code for the classification step and it comprises three modes, training a SVM model, e.g. train("anger");, obtaining the validation predictions, e.g. validate("anger");, and obtaining the test predictions, e.g. test("anger");. By default, it can only generate validation and test predictions, and print to screen for each emotion type. In order to activate training mode, you should remove the comment syntax in "int main()" function for each type. 

We also provide our final predictions in "val.py". You can create "test_prediction.pkl" and verify our predictions.

In order to build and test the project, several steps need to be done.

* Extract "feature.zip" to the project directory.
* Execute "make" on command line to build the code. It will generate "svm" binary file.
* Run "svm" binary file to visualize the validation and test results. 

Lastly, this project was built and tested on Ubuntu 14.04 LTS (64-bit). 

## References

Please cite the following paper:

[1] Savas Ozkan and Gozde Bozdagi Akar, Relaxed Spatio-Temporal Deep Feature Aggregation for Real-Fake Expression Prediction, arXiv:1708.07335:
```
@article{ozkan17endnet,
  title={Relaxed Spatio-Temporal Deep Feature Aggregation for Real-Fake Expression Prediction},
  author={Ozkan, Savas and Akar, Gozde B},
  journal={arXiv:1708.07335},
  year={2017},
}
```


