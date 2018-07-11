# Practicum-I
<p align="center">
  <img width="450" height="250" src="https://gdb.voanews.com/A13A1BE6-4C25-48DF-B44F-253481BD4333_w1023_r1_s.jpg">
</p>

## Classification of Medical image files
## Company interest project

**Description of data:**
The images were obtained from video of medical procedures.  Due to the nature of the images and the current development within the company the images and the specific objective cannot be shared and is protected by an NDA.  There is an interest in specific features that occur at a specific time point.  The images collected are roughly collected from the same time point and have 5 different classifications present.  The classifications will be labeled simply 0,1,2,3,4+ or labeled with a PN prefix.  This is a limited data set as there is only 3732 samples present.  

**Objective:**
The objective of this project is to accurately identify the 5 classifications with good accuracy.  This data set is a good candidate for use with transfer learning.  There are not enough samples for a convolution neural network to properly learn the defining features and will benefit from the use of models that have been trained on much larger data sets.
## Results
Using a smaller data set has proven challenging and has shown the limitations of some of the models and methods avaible for use.  There is more work to be done with this data set as the imbalance of the classes has proven the main issue when looking for the objective of high and correct calssification accuracy.  

<p align="center">
  <div class="row">
  <div class="column">
    <img src="https://github.com/jgarci135/Practicum-I/blob/master/Figures/train%20class%20distribution.JPG" style="width:80%">
  </div>
  <div class="column">
    <img height="290" src="https://github.com/jgarci135/Practicum-I/blob/master/Figures/validation%20class%20distribution.JPG" style="width:80%">
  </div>
</div>
</p>

Best recoreded accuacry was about 70%. Due to the imbalance of the data the predictions were heavliy influenced to only predict one class.  This can be seen in the confusion matrix or the predicted values.  And depending on the predicted class the accuracy was only expressed as the percentage of the class in the total data set.
<p align="center">
  <img width="450" height="500" src="https://github.com/jgarci135/Practicum-I/blob/master/Figures/Confusion%20matrix.JPG">
</p>

The methods used in this project had marginal positive results.  One such result that is to be further explored is the loss when using SGD.  It continued reducing over each epoch which suggests that the model was learning even though the validation accuracy was not improving.

<p align="center">
  <img width="450" height="375" src="https://github.com/jgarci135/Practicum-I/blob/master/Figures/extracted%20acc-loss%20mobilenetF.JPG">
</p>

This project and its issues requrie some other methods to be used to better address the imbalanced data, such as:

* Use different models
* Augment the minoirty data samples to balance the classes of the data
* The use of oversampling/undersampling may be useful
* Generate synthetic data

I believe there can be success in correctly classifying this data set with transfer learning. Some modifications are needed.
