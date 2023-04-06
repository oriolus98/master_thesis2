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

import geopandas as gpd
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from tslearn.clustering import TimeSeriesKMeans
from tslearn.barycenters import dtw_barycenter_averaging as dtw_avg

class ClimateClassifier:
    def __init__(self, df_path, var_names, n_clusters = 5, data_format = 'csv'):
        self.df_path = df_path
        self.var_names = var_names
        self.n_clusters = n_clusters
        self.data_format = data_format

    def classify(self):
        """
            Main method:
                It calls the different methods depending on input parameters
        """
        # We charge the data depending on the original format. The final format is the same: 3-d array where dimension 1 is pixel number,
        # dimension 2 is time, and dimension 3 is climate variables
        if self.data_format == 'csv':
            self.load_csv()
        elif self.data_format == 'netCDF4':
            self.load_nc()


    def load_nc(self):
        """
            In case data was saved in CDF4 format
        """

        # first we load one variable as its structure is different from the others
        # it also allows us to extract the pixels index
        f = netCDF4.Dataset(os.path.join(self.df_path, '2011_radiation.nc'))
        a = f.variables['Rg']

        df = np.empty((a.shape[1]*a.shape[2], a.shape[0], len(self.var_names) + 1))

        pixels = []
        k = 0
        lat = f.variables['lat']
        lon = f.variables['lon']

        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                df[k,:,5] = np.array(a[:, i, j])
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
