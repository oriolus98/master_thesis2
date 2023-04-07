"""
    Definition of the class ClimateClassifier:
        An object of this class will receive a dataset with a different number of climate variables (eg. temperature, radiation etc) 
        temporal series, for each pixel of earth. It will be able to classify Earth pixels in different climate groups. The number of
        groups is parametrizable. The temporal series clustering is implemented as follows:
        
        *   First we condensate the information on different variables on a unique temporal series per pixel using an autoencoder
        *   Then we use the algorithm k-means with Dynamic Time Warping as a measure of similarity

        The best autoencoder model weights will be saved. We also save an average temporal serie for each variable and each climate
        cluster. It can also plot the results.
"""

# libraries used
import pandas as pd
import numpy as np
import os
import datetime
import netCDF4
import logging

import geopandas as gpd
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from tslearn.clustering import TimeSeriesKMeans
from tslearn.barycenters import dtw_barycenter_averaging as dtw_avg



class ClimateClassifier:
    """
    Definition of the class ClimateClassifier:
        An object of this class will receive a dataset with a different number of climate variables (eg. temperature, radiation etc) 
        temporal series, for each pixel of earth. It will be able to classify Earth pixels in different climate groups. The number of
        groups is parametrizable. The temporal series clustering is implemented as follows:
        
        *   First we condensate the information on different variables on a unique temporal series per pixel using an autoencoder
        *   Then we use the algorithm k-means with Dynamic Time Warping as a measure of similarity

        The best autoencoder model weights will be saved. We also save an average temporal serie for each variable and each climate
        cluster. It can also plot the results.
    """

    def __init__(self, data, var_names, n_clusters = 5):
        self.df = data.df
        self.var_names = var_names
        self.n_clusters = n_clusters


    def classify(self):
        """
            Main method:
                It calls the different methods depending on input parameters
        """
        self.reduce_dimentionality()

        self.pixels_clustering


    def first_autoencoder(self):
        """
            First step to classify pixels is to reduce dimensionality from different climate variables to a unique temporal series
            for each pixel, as the k-means algorithm does not support yet multi-dimensional time series clustering.

            We saw that training the autoencoder on the hole dataset was very computationally and time expensive. We also saw that time
            series from the same cluster are quite similar. Thus we can choose a small random sample of data in order to pre-train our 
            autoencoder. We can save the weights that achieve a better validation loss. This is implemented on this method.
        """
        sample_size = self.df.shape[0]
        ind = np.random.permutation(self.df.shape[0])
        sample = self.df[ind[:1000], :, :]


class DataLoader:
    """
        Definition of the class DataLoader:
            An object of this class will receive a path and a format type and will load a complex climate dataframe. It calls different
            methods depending on the original format of the data. The final structure of the data loaded is always the same: a
            3-dimensional array where dimension 1 is pixel index, dimension 2 is time, and dimension 3 is climate variables. The final
            dataframe is saved on df attribute, as well as pixel coordinates are saved at longitud and latitud attributes. 

            This class allows to preprocess data in order to perform a pixels climatic classification using an object of the class 
            ClimateClassifier.

            It supports netCDF4 and csv original data formats. The methods implemented here are solutions ad hoc for the problems 
            proposed at my master thesis. However, this class can be easily modified to include new data formats and original structures.
    """

    def __init__(self, df_path, var_names, data_format = 'csv'):
        self.df_path = df_path
        self.var_names = var_names
        self.data_format = data_format

        # We charge the data depending on the original format. The final format is always the same: 3-d array where dimension 1 is 
        # pixel number, dimension 2 is time, and dimension 3 is climate variables
        logging.info('DataLoader object initialised')

        if self.data_format == 'csv':
            self.load_csv()
            logging.info('Loading data with csv format')

        elif self.data_format == 'nc':
            self.load_nc()
            logging.info('Loading data with netCDF4 format')


    def load_nc(self):
        """
            In case data was saved in CDF4 format
        """

        # first we load one variable as its structure is different from the others.
        # It also allows us to extract the pixels coordinates
        logging.info('Charging radiation data')
        f = netCDF4.Dataset(os.path.join(self.df_path, '2011_radiation.nc'))
        a = f.variables['Rg']

        df = np.empty((a.shape[1]*a.shape[2], a.shape[0], len(self.var_names) + 1))

        pixels = []
        k = 0
        lat = f.variables['lat']
        lon = f.variables['lon']
        dim = len(self.var_names)

        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                df[k,:,dim] = np.array(a[:, i, j])
                pixels.append([lat[i], lon[j]])
                k += 1

        self.pixels = pixels 

        l = 0

        # now we load all the other variables
        for var in self.var_names:
            f = netCDF4.Dataset(os.path.join('./ccm_dataset/', var, '2011_' + var + '.nc'))
            a = f.variables[var]
            k = 0

            for i in range(a.shape[1]):
                for j in range(a.shape[2]):
                    df[k,:,l] = np.array(a[:, i, j])
                    k += 1  

            l += 1

        self.var_names.append('radiation')
        self.df = df
        self.separate_pixels_nc()



    def load_csv(self):
        """
            In case data was saved in csv format
        """

        df = np.empty((len(os.listdir(self.df_path)), 385, len(self.var_names)))
        pixels = []
        i = 0

        for file in os.listdir(self.df_path):
            df[i,:,:] = pd.read_csv(os.path.join(self.df_path, file), header=None)
            i += 1
            pixels.append(file)
            
        self.df = df[:,1:,:]
        self.pixels = pixels
        self.separate_pixels_csv()



    def separate_pixels_nc(self):
        """
            In this case, the coordinates of each pixel where saved as a list of lists
        """
        
        self.latitud = np.array(self.pixels)[:,0]
        self.longitud = np.array(self.pixels)[:,1]



    def separate_pixels_csv(self):
        """
            In this case, the coordinates of each pixel where saved as a string, coming from each file name
        """

        coord_x = []
        coord_y = []

        for i in range(len(self.pixels)):
            self.pixels[i] = self.pixels[i].replace('.csv', '')
            coord_x.append(float(self.pixels[i].split(',')[1]))
            coord_y.append(float(self.pixels[i].split(',')[0]))

        self.longitud = np.array(coord_x)
        self.latitud = np.array(coord_y)


    
