# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:56:37 2021

@author: Salehin
"""

#Load required packages
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Input, GaussianNoise
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import GaussianNoise
import tensorflow_probability as tfp

#Build Denoised Autoencoder

class Autoencoder:
    
    def __init__(self,a,activations1,activations2,noise):
        #The constructor specifies
        #The number of layer represents size of the hidden layers
        #Layer_size is a list representing size of each hidden units
        #Activations1 and Activation2 represents activation functions for encoder and decoder, each passed as a list
        #If the noise variable is set as true, user can set amount of noise, the autoencoder will act as denoised autoencoder in this case
    
        self.layer_size=a # a list that includes hidden layer sizes
        self.activations1=activations1 # a list that includes successive activation functions for encoder
        self.activations2=activations2 # Same concept, but for decoder
        self.noise=noise # Setting true will make the autoencoder a denoised one
        
    
    def set_noise(self,noiseamount): # this function sets amount of Gaussian noise for encoder
        if self.noise==True:
            self.noiseamount=noiseamount # set amount of noise
        else:
            self.noiseamount=0
        return self.noiseamount    
    
              
    
    def encoder(self):
        self.encoder=Sequential()
        if self.noise:
            self.encoder.add(GaussianNoise(self.noiseamount))
        self.encoder.add(Dense(units=self.layer_size[1],activation=self.activations1[0],input_shape=(self.layer_size[0],)))
        for i in range(1,len(self.layer_size)-1):
            self.encoder.add(Dense(units=self.layer_size[i+1],activation=self.activations1[i-1]))

        return self.encoder

    def decoder(self):  
        self.decoder=Sequential()
        self.layer_size.reverse()
        self.decoder.add(Dense(units=self.layer_size[1],activation=self.activations2[0],input_shape=(self.layer_size[0],)))
        for i in range(1,len(self.layer_size)-1):
            self.decoder.add(Dense(units=self.layer_size[i+1],activation=self.activations2[i]))
                
        #encoder.add(BatchNormalization())
        #encoder.add(Dropout(rate=0.5))
        return self.decoder
    
    def create_model(self,loss_function,optimizer,lr):
        autoencoder = Sequential([self.encoder, self.decoder])      
        autoencoder.compile(loss=loss_function,optimizer=optimizer(lr=lr),metrics=['accuracy'])
        return autoencoder
    
        


