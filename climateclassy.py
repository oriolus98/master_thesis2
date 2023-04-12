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

    def __init__(self, data, results_path, n_clusters = 5, sample_size = 200, path_to_model = None, epochs_step1 = 100000, epochs_final = 100):
        self.df = data.df
        self.var_names = data.var_names
        self.results_path = results_path
        self.n_clusters = n_clusters
        self.sample_size = sample_size
        self.path_to_model = path_to_model
        self.epochs_step1 = epochs_step1
        self.epochs_final = epochs_final
        self.longitud = data.longitud
        self.latitud = data.latitud


    def classify(self):
        """
            Main method:
                It calls the different methods depending on input parameters
        """
        logging.info('Step 1')
        self.first_autoencoder()

        logging.info('Final autoencoder')
        self.final_autoencoder()

        logging.info('We keep only the encoder part')
        self.encoder()

        logging.info('Time series clustering')
        self.ts_clustering()

        self.plot_results()

        logging.info('Computing average series for each variable on each climate group')
        self.average_series()



    def first_autoencoder(self):
        """
            First step to classify pixels is to reduce dimensionality from different climate variables to a unique temporal series
            for each pixel, as the k-means algorithm does not support yet multi-dimensional time series clustering.

            We saw that training the autoencoder on the hole dataset was very computationally and time expensive. We also saw that time
            series from the same cluster are quite similar. Thus we can choose a small random sample of data in order to pre-train our 
            autoencoder. We can save the weights that achieve a better validation loss. This is implemented on this method.
        """
        ind = np.random.permutation(self.df.shape[0])
        sample = self.df[ind[:self.sample_size], :, :]

        ind = np.random.permutation(sample.shape[0])
        training_idx, test_idx = ind[:int(self.sample_size*0.7)], ind[int(self.sample_size*0.7):] # 70% training, 30% validation
        x_train, x_test = sample[training_idx,:,:], sample[test_idx,:,:]

        if self.path_to_model:
            # in case we already saved a model and just want to retrain it
            autoencoder = tf.keras.models.load_model(self.path_to_model)

        else:
            # define our model for the first time, in case we don't have a saved model:
            inp = layers.Input(shape=(self.df.shape[1], self.df.shape[2]))
            encoder = layers.TimeDistributed(layers.Dense(50, activation='tanh'))(inp)
            encoder = layers.TimeDistributed(layers.Dense(10, activation='tanh'))(encoder)
            latent = layers.TimeDistributed(layers.Dense(1, activation='tanh'))(encoder)
            decoder = layers.TimeDistributed(layers.Dense(10, activation='tanh'))(latent)
            decoder = layers.TimeDistributed(layers.Dense(50, activation='tanh'))(decoder)
            out = layers.TimeDistributed(layers.Dense(self.df.shape[2]))(decoder)

            autoencoder = Model(inputs=inp, outputs=out)
    
        autoencoder.compile(optimizer='adam', loss='mse')
        logging.info('Training autoencoder with a small sample of {} pixels.'.format(self.sample_size))
        model_checkpoint = ModelCheckpoint(os.path.join(self.results_path, 'best_model_step1.h5'), monitor='val_loss', mode='min', save_best_only=True)
        autoencoder.fit(x_train, x_train, epochs = self.epochs_step1, validation_data= (x_test, x_test), callbacks=[model_checkpoint])

        logging.info('Saving best model with a validation loss of {}'.format(model_checkpoint.best))
        self.sample = sample


    def final_autoencoder(self):
        """
            Once the model is pre-trained with a small sample, we can train it with the hole dataset with a small number of epochs
        """     
        ind = np.random.permutation(self.df.shape[0])
        training_idx, test_idx = ind[:int(self.df.shape[0]*0.7)], ind[int(self.df.shape[0]*0.7):] # 70% training, 30% validation
        x_train, x_test = self.df[training_idx,:,:], self.df[test_idx,:,:]

        autoencoder = tf.keras.models.load_model(self.path_to_model)
        autoencoder.compile(optimizer='adam', loss='mse')
        logging.info('Training autoencoder with the hole dataset.')
        model_checkpoint = ModelCheckpoint(os.path.join(self.results_path, 'best_model_final.h5'), monitor='val_loss', mode='min', save_best_only=True)
        autoencoder.fit(x_train, x_train, epochs = self.epochs_final, validation_data= (x_test, x_test), callbacks=[model_checkpoint])

        logging.info('Saving best model with a validation loss of {}'.format(model_checkpoint.best))

    
    def encoder(self):
        """ 
            We only want the encoder part, to reduce dimensionality
        """
        model = tf.keras.models.load_model(os.path.join(self.results_path, 'best_model_final.h5'))

        inp = layers.Input(shape=(self.df.shape[1], self.df.shape[2]))

        encoder = layers.TimeDistributed(layers.Dense(50, activation='tanh'))(inp)
        encoder1 = layers.TimeDistributed(layers.Dense(10, activation='tanh'))(encoder)
        latent = layers.TimeDistributed(layers.Dense(1, activation='tanh'))(encoder1)

        encoder_model = Model(inp, latent)
        encoder_model.get_layer('time_distributed').set_weights(model.get_layer('time_distributed').get_weights())
        encoder_model.get_layer('time_distributed_1').set_weights(model.get_layer('time_distributed_1').get_weights())
        encoder_model.get_layer('time_distributed_2').set_weights(model.get_layer('time_distributed_2').get_weights())

        self.encoder_model = encoder_model


    def ts_clustering(self):
        """
            We fit the model on the small dataset and predict on the hole dataset
        """
        logging.info('Reducing dimensionality of subset')
        data = self.encoder_model.predict(self.sample)
        data = data.reshape((self.sample_size, self.df.shape[1]))

        data = pd.DataFrame(data)

        logging.info('Fitting k-means')
        model = TimeSeriesKMeans(n_clusters= self.n_clusters, metric="dtw", max_iter=100)
        model.fit(data)

        res = self.encoder_model.predict(self.df)
        res = res.reshape((self.df.shape[0], self.df.shape[1]))

        logging.info('Predicting hole dataset')
        results = pd.DataFrame()
        results['group'] = model.predict(res)
        results['coord_x'] = self.longitud
        results['coord_y'] = self.latitud

        results.to_csv(os.path.join(self.results_path, 'clustering_results.csv'))
        self.results = results



    def plot_results(self):
        """
            Plot climate zones computed on a world map. To execute this method is necessary execute before the method ts_clustering
        """
        results = pd.read_csv(os.path.join(self.results_path, 'clustering_results.csv'))

        # From GeoPandas, our world map data
        worldmap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

        # Creating axes and plotting world map
        fig, ax = plt.subplots(figsize=(12, 6))
        worldmap.plot(color="lightgrey", ax=ax)

        for group, color in zip(results['group'].unique(), ['blue', 'red', 'green', 'orange', 'purple']):
            plt.scatter(x = results.coord_x[results['group'] == group], y = results.coord_y[results['group'] == group], s = 10, color=color, label=f"Group {group}")

        # Creating axis limits and title
        plt.xlim([-180, 180])
        plt.ylim([-90, 90])

        plt.title('Pixels classification on climate type')
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.savefig(os.path.join(self.results_path, 'groups_map.png'))
        plt.show()



    def average_series(self):
        """
            Compute average temporal serie for each variable on each climatic group using DTW formalism.
        """
        mean_series = np.empty((self.n_clusters, self.df.shape[1], self.df.shape[2]))

        for i in range(self.n_clusters):
            for j in range(self.df.shape[2]):
                mean_series[i,:,j] = dtw_avg(self.df[self.results['group'] == i,:,j], max_iter = 100).reshape([self.df.shape[1],])

        logging.info('Saving average series')
        for i in range(mean_series.shape[0]):
            np.savetxt(os.path.join(self.results_path,f"{'mean_series/'}{i}{'.txt'}"), mean_series[i,:,:])




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
            logging.info('Loading data with csv format')
            self.load_csv()
            

        elif self.data_format == 'nc':
            logging.info('Loading data with netCDF4 format')
            self.load_nc()
            


    def load_nc(self):
        """
            In case data was saved in CDF4 format
        """

        # first we load one variable as its structure is different from the others.
        # It also allows us to extract the pixels coordinates
        logging.info('Charging radiation data')
        f = netCDF4.Dataset(os.path.join(self.df_path, 'radiation', '2011_radiation.nc'))
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
            f = netCDF4.Dataset(os.path.join(self.df_path, var, '2011_' + var + '.nc'))
            a = f.variables[var]
            k = 0
            logging.info('Charging {} data'.format(var))

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
        logging.info('Saving pixels index')
        self.latitud = np.array(self.pixels)[:,0]
        self.longitud = np.array(self.pixels)[:,1]
        logging.info('{} pixels saved'.format(self.latitud.shape[0]))



    def separate_pixels_csv(self):
        """
            In this case, the coordinates of each pixel where saved as a string, coming from each file name
        """
        logging.info('Saving pixels index')
        coord_x = []
        coord_y = []

        for i in range(len(self.pixels)):
            self.pixels[i] = self.pixels[i].replace('.csv', '')
            coord_x.append(float(self.pixels[i].split(',')[1]))
            coord_y.append(float(self.pixels[i].split(',')[0]))

        self.longitud = np.array(coord_x)
        self.latitud = np.array(coord_y)
        logging.info('{} pixels saved'.format(self.latitud.shape[0]))