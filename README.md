# Survival-Predicition-using-Autoencoder-and-AFT-
Objective: Create a feature set for cancer patients using clinical features and gene expression data and use the features to predict overall survival.

The goal of this project is to predict survival times of patients using gene expression features and clinical features. We will see if the gene expression features has any significance in predicting overall survival. For the 1st part of our projcet, we are aiming to reduce the dimensionality of our gene-expression dataset which has been extracted from http://www.linkedomics.org/. The RNA-Seq data has 528 samples and 4571 features. For dimensionality reduction we will be using standard, denoisng and variational Autoencoder, which are a special type of neural network. After the feature reduction, we will use reduced feature in our clinical dataset and evaluate the effectiveness of predictions.

Autoencoder is an unsupervised artificial neural network that learns how to efficiently compress and encode data then learns how to reconstruct the data back from the reduced encoded representation to a representation that is as close to the original input as possible. For higher dimensional data, autoencoders are capable of learning a complex representation of the data (manifold) which can be used to describe observations in a lower dimensionality and correspondingly decoded into the original input space.

Denoised Autoencoder.py has the implementation of standard and denoised autoencoder where user will be able to select the number of layers, layer size and activations for the autoencoder. The variational autoenocoder ipython file implements variational autoencoder.

The reduced output has been used in weibull.py file, where these features were merged with clinicla features of patients. After preprocessing, the final data has 500*10 dimensional data. We used weibull AFT model to predict expectation of survival. Finally, a c-index has been used to evaluate our prediction.

Currently I have been experimenting with different output dimensions and their impact on the overall accuracy. In future, we will add gridseach method to tune our network for further improvements.

To get a complete idea, please look at the project report (pdf) or the presentation (pptx) file and look at the ipython notebook files. Happy Coding !

© 2021 GitHub, Inc.
