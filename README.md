# Practicum-I
## Classification of Medical image files
## Company interest project

**Description of data:**
The images were obtained from video of medical procedures.  Due to the nature of the images and the current development within the company the images and the specific objective cannot be shared and is protected by an NDA.  There is an interest in specific features that occur at a specific time point.  The images collected are roughly collected from the same time point and have 5 different classifications present.  The classifications will be labeled simply 0,1,2,3,4+ or labeled with a PN prefix.  This is a limited data set as there is only 3732 samples present.  

**Objective:**
The objective of this project is to accurately identify the 5 classifications with good accuracy.  This data set is a good candidate for use with transfer learning.  There are not enough samples for a convolution neural network to properly learn the defining features and will benefit from the use of models that have been trained on much larger data sets.
## Results
Using a smaller data set has proven to be challenging and has shown the limitations of some of the models and methods avaible for use.  There is more work to be done with this data set as the imbalance of the classes has proven the main issue when looking for the objective of high and correct calssification accuracy.  Best recoreded accuacry was about 70% .The methods used in this project had marginal positive results but exposed some other methods to be used to betteraddress the imbalanced data.

* Use different models
* Augment the minoirty data samples to balance the classes of the data
* The use of oversampling/undersampling may be useful

I believe there can be success in correctly classifying this data set with transfer learning. Some modifications are needed.
