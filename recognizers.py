#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 16:16:08 2018

@author: schan
"""


import errno
import os
import sys

from hmmlearn import hmm
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Activation, BatchNormalization, Conv1D, Dense, Dropout
from keras.layers import GlobalAveragePooling1D, Input, LSTM, Masking, Permute, Reshape
from keras.layers import concatenate, multiply
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score

import preprocessing

RNN_MODEL_DIRECTORY = os.path.join("rnn_models")
HMM_MODEL_DIRECTORY = os.path.join("hmm_models")
LEFT_MODEL_DIR = os.path.join(HMM_MODEL_DIRECTORY, "left")
RIGHT_MODEL_DIR = os.path.join(HMM_MODEL_DIRECTORY, "right")
LR_MODEL_DIR = os.path.join(HMM_MODEL_DIRECTORY, "LR")


#==============================================================================
# HMM RECOGNIZER
#==============================================================================

class HmmRecognizer(BaseEstimator, ClassifierMixin):
    """
    Hidden Markov Model (HMM) Recognizer extends Scikit learn BaseEstimator and becomes 
    compatible with other scikit learn utilities. This class provides means to train, evaluate, and 
    predict gesture class, and save trained models into preset directories.
    
    Attributes
    ----------
    hand: str
        this parameter determines which models to use during prediction (necessary for GridSearch 
        where class methods are not directly accessible) Options: 'left', 'right', 'double'
    num_states: int
        the number of hidden states the gestures will be assumed to have during training.
    topology: str
        the topology of the model to be trained. 
        Options: 
            'full': fully-connected states
            'left-to-right-full': left-to-right connected, can transition to any of the states ahead
            'left-to-right-1': left-to-right connected, can transition to only the next state or itself
            'left-to-right-2' left-to-right connected, can transition to up to 2 states ahead or itself
    standardize: bool
        if set to true, training and prediction will rely on internal standardization parameters. 
        Default: False. Only set to true when doing GridSearch.
    """
    
    def __init__(self, hand=None, num_states=None, topology=None, standardize=False):
        super().__init__()
        self.models = None
        self.load_models()
        self.hand = hand
        self.num_states = num_states
        self.topology = topology
        self.standardize = standardize
        
    
    def fit(self, X, y, save=False):
        """
        Creates and trains a Gaussian HMM for each class of gesture in the input data. 
        
        Parameters
        ----------
        X: list
            a list of pd.DataFrame, each DataFrame being the data of a single gesture instance
        Y: list
            a list of gesture labels (strings), where each label corresponds to the data in X 
            of the same index.
        save: boolean, optional
            Set to true if desired to save trained models to local directory. Default: false
        standardize: boolean, optional
            If set to True, the data will be standardized before training. 
        """
        X_train = {}
        
        # If incoming data is already standardized, self.standardize should be set to False. 
        # If running GridSearch, then this should be set to True.
        if self.standardize:
            self._accel_range, self._euler_range = preprocessing.get_standardize_params(X)
            X = preprocessing.standardize_data(X, self._accel_range, self._euler_range)
        
        # Map gesture data to their gesture names
        for instance, gesture_name in zip(X, y):
            if gesture_name not in X_train:
                X_train[gesture_name] = []
            X_train[gesture_name].append(instance)
        
        # Train and save each gesture model with their respective gesture data
        for gesture_name, instances in X_train.items():
            self.__train_gesture(gesture_name,
                                 instances,
                                 save=save)
    
    def predict(self, X):
        """ 
        Classifies input gesture instance X using the models from the sepcified hand.
        
        Parameters
        ----------
        X: pd.DataFrame
            DataFrame that contains instance data
        """
        prev_p = -sys.float_info.max
        
        # Load models if models have not been loaded yet
        if not self.models:
            self.load_models()
        
        # Standardize data based on training data standardization parameters
        if self.standardize:
            X = preprocessing.standardize_data(X, self._accel_range, self._euler_range)
        
        # Compute the ln-probability of the instance being generated by each gesture model
        # and save the gesture name that returns the highest value.
        for label, model in self.models[HmmRecognizer.__get_gesture_type(self.hand)].items():
            ln_prob = model.score(X)
            if ln_prob > prev_p:
                gesture_pred = label
                prev_p = ln_prob
        
        if gesture_pred is None:
            print(f"An error has occured during recognition of gesture {gesture_pred}")
        else:
            return gesture_pred
        
        
    def score(self, X, y_true, score_type='accuracy'):
        """
        Evaluates performance of the loaded models.
        
        ----------
        X: list
            a list of pd.DataFrame that each contain gesture data for each gesture instance.
        y_true: list
            a list of gesture labels (strings), where each label corresponds to the data in X 
            of the same index.
        score_type: str, optional
            a parameter to determine what metric the models will be evaluated with. 
            Options include: 'accuracy' (default) and 'f1'
        
        Returns
        -------
        float:
            an accuracy or f1 score value between 0 and 1. 
            
        """
        y_predict = []
        
        for instance in X:
            y_predict.append(self.predict(instance))
        
        if score_type == 'accuracy':
            return accuracy_score(y_true, y_predict)
        elif score_type == 'f1':
            return f1_score(y_true, y_predict, average='micro')
        
        
    def load_models(self):
        """
        Loads trained models that have been saved in the preset directory (set as global variables)
        
        Returns
        -------
        dict:
            dictionary containing trained and saved HMM models with keys 'L', 'R', and 'LR' to 
            indicate 'left', 'right', and 'double' handed gesture models.
        """
        self.models = {}
        
        self.models['L'] = HmmRecognizer.load_hmm_models_for_gesture_type('L')
        self.models['R'] = HmmRecognizer.load_hmm_models_for_gesture_type('R')
        self.models['LR'] = HmmRecognizer.load_hmm_models_for_gesture_type('LR')
        
        return self.models
    
    
    def __train_gesture(self, gesture_name, instances, save=False):
        # Trains and saves a single Gaussian HMM model.
        print(f"Training {gesture_name}...")
        model = self.__create_gaussian_model(instances)
        
        if save:
            HmmRecognizer.save_model(gesture_name, model)
        return model
    
        
    def __create_gaussian_model(self, X_train):
        # Trains a single Gaussian HMM model.
        lengths = [len(x) for x in X_train]
        X_train = pd.concat(X_train).reset_index(drop=True).values
        
        transmat = self.__create_transition_matrix()
        pi = self.__create_start_probabilities()
        means, covars = self.__estimate_normal_distribution_params(X_train)
        
        # Set diagonal covariance matrix
        covars_ = np.array([covar.diagonal() for covar in covars])
        model = hmm.GaussianHMM(n_components=self.num_states, covariance_type='diag', params='mct', 
                                init_params='', n_iter=500, verbose=True, tol=1e-2)
        
        model.transmat_ = np.array(transmat)
        model.startprob_ = np.array(pi)
        
        model.means_ = np.array(means)
        model.covars_ = covars_
        
        model.fit(X_train, lengths)
        
        return model
    
    @staticmethod
    def __get_gesture_type(hand):
        # Returns the gesture handedness of the current recognizer.
        if hand == 'left':
            return 'L'
        elif hand == 'right':
            return 'R'
        else:
            return 'LR'
    
    
    def __create_start_probabilities(self):
        # Creates different start probabilities (pi) for each state depending on the topology chosen.
        pi = np.zeros(self.num_states)
        
        if self.topology.startswith('left-to-right'):
            # i.e. sequence always begins with the first state
            pi[0] = 1.0
        elif self.topology.startswith('full'):
            # i.e. fully-connected states, sequence have equal probability to start from any state
            pi[:] = 1.0 / float(self.num_states)
        else:
            raise ValueError(f"Unknown topology {self.topology}")
            
        # Make sure probabilities always sum to 1.0
        assert np.isclose(np.sum(pi), 1.0)
        
        return pi
    
    
    def __create_transition_matrix(self):
        # Creates transition matrix of n x n size, n being number of states. This matrix 
        # characterizes the probabilities of each state transitioning to another.
        transmat = np.zeros((self.num_states, self.num_states))
        
        if self.topology.startswith('left-to-right'):
            if self.topology.endswith('-full'):
                # In the full 'left-to-right' topology, each state is connected to itself and all
                # states to the right of it. The transition matrix will therefore be upper-right
                # -triangular in shape
                for i in range(self.num_states):
                    transmat[i,i:] = 1.0 / float(self.num_states - i)
                    
            elif self.topology.endswith('-1'):
                # delta = 1, i.e. self transition and transition to next state is allowed
                for i in range(self.num_states):
                    if i == self.num_states-1:
                        transmat[i,i] = 1.0
                    else:
                        transmat[i,i] = 0.5
                        transmat[i,i+1] = 0.5
            elif self.topology.endswith('-2'):
                # delta = 2, i.e. self transition and transition to any of the next two states 
                # are allowed.
                for i in range(self.num_states):
                    if i == self.num_states-1:
                        transmat[i,i] = 1.0
                    elif i == self.num_states-2:
                        transmat[i,i] = 0.5
                        transmat[i,i+1] = 0.5
                    else:
                        transmat[i,i] = 1.0 / 3.0
                        transmat[i,i+1] = 1.0 / 3.0
                        transmat[i,i+2] = 1.0 / 3.0
                        
        elif self.topology.startswith('full'):
            # Fully-connected topology, i.e. each state can transition to any other state.
            for i in range(self.num_states):
                transmat[:,:] = 1.0 / float(self.num_states)
        
        else:
            raise ValueError(f"Unknown topology {self.topology}")
        
        assert np.allclose(np.sum(transmat, axis=1), np.ones(self.num_states))
        
        return transmat
    
    
    def __estimate_normal_distribution_params(self, observations):
        # Creates the initial emissions distribution. Data is clustered into n distinct gaussian 
        # distributions, n being the number of states. Return a list of mean values, representing 
        # the center of each Gaussian distribution, and the covariance matrix of the states.
        if isinstance(observations, list):
            if len(observations) == 0:
                raise ValueError("Observations must contain at least one sequence.")
            all_obs = np.concatenate(observations)
        elif isinstance(observations, np.ndarray):
            all_obs = observations
        else:
            raise ValueError("Observations mst be a list of sequences or a numpy concatenated "
                             "array of sequences.")
        
        n_features = all_obs.shape[1]
        
        # Estimate means of each state cluster
        kmeans = KMeans(n_clusters=self.num_states)
        kmeans.fit(all_obs)
        means = kmeans.cluster_centers_
        assert means.shape == (self.num_states, n_features)
        
        # Predict clusters and calcualte covariances
        predict = kmeans.predict(all_obs)
        covars = []
        for state in range(self.num_states):
            indices = np.where(predict == state)[0]
            state_obs = [all_obs[i] for i in indices]
            
            assert len(state_obs) > 0
            covar_estimator = EmpiricalCovariance().fit(state_obs)
            covar = covar_estimator.covariance_
            covars.append(covar)
        covars = np.array(covars)
        
        # Set covariance matrix to be diagonal
        # (i.e. assume each state to be independent from another)
        for state in range(self.num_states):
            covar = covars[state]
            diag = covar.diagonal().copy()
            covar[:,:] = 0.0
            diag[diag < 1e-4] = 1e-4
            np.fill_diagonal(covar, diag)
            
        assert means.shape == (self.num_states, n_features)
        assert covars.shape == (self.num_states, n_features, n_features)
        
        return means, covars
    
        
    @staticmethod
    def load_hmm_models_for_gesture_type(gesture_type):
        # Returns a model object of gesture_type from directory.
        models = {}
        
        if gesture_type == 'L':
            directory = LEFT_MODEL_DIR
        elif gesture_type == 'R':
            directory = RIGHT_MODEL_DIR
        elif gesture_type == 'LR':
            directory = LR_MODEL_DIR
        else:
            directory = HMM_MODEL_DIRECTORY
        
        for f in os.listdir(directory):
            if f.endswith(".pkl"):
                gesture_name = f.replace(".pkl", "")
                models[gesture_name] = joblib.load(os.path.join(directory, gesture_name + ".pkl"))
        
        return models
    
    
    @staticmethod
    def save_model(gesture_name, model):
        
        if 'left' in gesture_name:
            directory = LEFT_MODEL_DIR
        elif 'right' in gesture_name:
            directory = RIGHT_MODEL_DIR
        else:
            directory = LR_MODEL_DIR
        
        if not os.path.exists(directory):
            os.mkdir(directory)
        
        joblib.dump(model, os.path.join(directory, gesture_name + ".pkl"))
        print(f"{gesture_name} gesture model has been saved to '{directory}'")
    
    
#==============================================================================
# LSTM RECOGNIZER
#==============================================================================
class LstmRecognizer(BaseEstimator, ClassifierMixin):
    """
    LSTM Recognizer extends Scikit learn BaseEstimator and becomes compatible with other 
    scikit learn utilities. This class provides means to train, evaluate, and predict gesture 
    class, and save trained models into preset directories.
    
    Attributes
    ----------
    epochs: int
        number of epochs training to be used for training
    """
    def __init__(self, epochs=None, standardize=False):
        self.label_index_dict = {}
        self.epochs = epochs
        self.standardize = standardize
        
        
    def fit(self, X, y, trial_prefix=None, save=True):
        """
        Trains a Neural Network model with the input data and corresponding labels.
        
        Parameters
        ----------
        instances: list or ndarray
            list of training data as pd.DataFrame with shape (num_features, time_steps),
            or numpy array containing all training data with shape (number of instances, 
            num_features, time_steps)
        y: list
            a list of gesture labels (strings)
        trial_prefix: str, optional
            given string identifier for the trained model
        save: bool, optional
            if True, the trained model will be saved to the directory set by the global variable 
            in the beginning of this script.
        """
        if isinstance(X, list):
            if isinstance(X[0], pd.DataFrame):
                # If incoming data is already standardized, self.standardize should be set to False. 
                # If running GridSearch, then this should be set to True.
                if self.standardize:
                    self._accel_range, self._euler_range = preprocessing.get_standardize_params(X)
                    X = self.__standardize_data(X)
            else:
                raise ValueError("Input data must be list of pandas DataFrame")
            
            # Shuffle list before training
            X, y = preprocessing.shuffle_lists(X, y, seed=45)
            
            # padding and transposing will return a numpy array, X
            X = self.__pad_and_transpose_keras_lstm_data(X)
        
        time_steps = X.shape[2]
        num_features = X.shape[1]
        num_labels = len(set(y))
        
        # Stores the training max length as a class variable to be used during prediction
        if not hasattr(self, '_pad_length'):
            self._pad_length = time_steps
        
        # Create the cnn lstm model
        self.model = self.__create_cnn_lstm_model(time_steps, num_features, num_labels)
        
        # Map each label to an integer using one-hot-encoding
        label_categories = self.__categorize_labels(y)
        one_hot_labels = to_categorical(np.array(label_categories), num_classes=num_labels)
        
        # Create callbacks to adaptively adjust learning rate and create model checkpoints
        callbacks = self.__create_callbacks(trial_prefix=trial_prefix)
        
        # trains a neural network model for the given hand
        self.history = self.model.fit(X, one_hot_labels,
                                      validation_split=0.3,
                                      batch_size=128,
                                      epochs=self.epochs,
                                      verbose=2,
                                      callbacks=callbacks)
        if save:
            LstmRecognizer.save_model(self.model, trial_prefix)
        
        return self.model
    
    
    def predict(self, X):
        """
        Classifies input gesture instance X using the loaded trained model.
        
        Parameters
        ----------
        X: ndarray
            a numpy array that contains the data for a single gesture instance
            
        """
        if not self.model:
            raise ValueError("Model is not available. Please train or load model.")
        
        # If a single pd.DataFrame gesture instance is the input
        if isinstance(X, pd.DataFrame) and self.standardize:
            X = preprocessing.standardize_data([X])[0]
        
        # pad data to be identical in length to the data used to train the model.
        X_padded = pad_sequences(X, dtype='float64',
                                 maxlen=self._pad_length,
                                 padding='post',
                                 truncating='post')
        
        return self.model.predict_classes(X_padded)
    
    
    def score(self, X, y, score_type='accuracy'):
        """
        Evaluates performance of the loaded model.
        
        Parameters
        ----------
        X: list or ndarray
            a list of pd.DataFrame that each contain gestyure data for each gesture instance, 
            or a single numpy array containing all test data with shape (number of instances, 
            num_features, time_Steps)
        y: list
            a list of gesture labels (strings)
        score_type: str, optional
            a parameter to determine what metric the models will be evaluated with.
            Options include: 'accuracy' (default) and 'f1'
        """
        if score_type == 'accuracy' and hasattr(self, '_accuracy'):
            return self._accuracy
        if score_type == 'f1' and hasattr(self, '_f1_score'):
            return self._f1_score
        
        if isinstance(X, list):
            if isinstance(X[0], pd.DataFrame):
                if self.standardize:
                    X = self.__standardize_data(X)
            else:
                raise ValueError("Input data must be list of pandas DataFrame")
                
            X = self.__pad_and_transpose_keras_lstm_data(X, max_length=self._pad_length)
        
        if not self.label_index_dict:
            raise ValueError("Label index dictionary has not yet been loaded. Model must be "
                             + "trained prior to calling the score method.")
        
        num_labels = len(self.label_index_dict)
        
        label_categories = []
        
        # Convert label list to one-hot-encoded list
        for label in y:
            label_categories.append(self.label_index_dict[label])
        one_hot_labels = to_categorical(np.array(label_categories), num_classes=num_labels)
        
        self._loss, self._accuracy, self._f1_score = self.model.evaluate(X, 
                                                                         one_hot_labels,
                                                                         verbose=2)
        
        if score_type == 'accuracy':
            return self._accuracy
        if score_type == 'f1':
            return self._f1_score
    
    
    def load_model(self, model_name):
        """
        Loads model by the model_name into the class object for prediction.
        
        Parameters
        ----------
        model_name: str
            the name of the model to be loaded
        
        Returns
        -------
        keras.models.Model:
            The model loaded from local directory indicated by global RNN_MODEL_DIRECTORY
        """
        model_path = os.path.join(RNN_MODEL_DIRECTORY, model_name, "model.h5")
        self.model = load_model(model_path)
        
        #TODO: Find way to load self_tag_length and self.label_index_dict into class as well
        
        return self.model
    
    
    def __create_cnn_lstm_model(self, time_steps, num_features, num_labels):
        # Generates the neural network architecture for gesture recognition
        input_layer = Input(shape=(num_features, time_steps))
        
        # LSTM Network
        lstm = Masking()(input_layer)
        lstm = LSTM(8)(lstm)
        lstm = Dropout(0.8)(lstm)
        
        # Temporal Convolution Network, total of 3 layers
        cnn = Permute((2,1))(input_layer)
        cnn = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(input_layer)
        cnn = BatchNormalization()(cnn)
        cnn = Activation('relu')(cnn)
        cnn = self.__squeeze_excite_block(cnn)
        
        cnn = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Activation('relu')(cnn)
        cnn = self.__squeeze_excite_block(cnn)
        
        cnn = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Activation('relu')(cnn)
        
        cnn = GlobalAveragePooling1D()(cnn)
        
        # Combine LSTM and CNN, then add softmax layer to create model
        cnn_lstm = concatenate([cnn, lstm])
        output = Dense(num_labels, activation='softmax')(cnn_lstm)
        model = Model(input_layer, output)
        
        # Compile with starting learning rate set to 0.01, and add customized f1 score metrics
        adam_optimizer = Adam(lr=0.01)
        model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy',
                      metrics=['accuracy', self.__f1_score])
        
        print(model.summary())
        
        return model
    
    def __pad_and_transpose_keras_lstm_data(self, data, max_length=None):
        # Pads input data with zeros to the max length, and transposes it to fit keras lstm model 
        # input configuration
        if max_length is None:
            max_length = 0
            for instance in data:
                length = len(instance)
                if length > max_length:
                    max_length = length
            self._pad_length = max_length
        
        num_features = data[0].shape[1]
        
        result = np.zeros((len(data), num_features, max_length))
        for i in range(len(data)):
            result[i,:,:len(data[i])] = data[i].T.values
        
        return result
    
    def __create_callbacks(self, fpath=None, trial_prefix=None):
        # Creates callbacks to optimize model learning.
        if fpath is None:
            if trial_prefix:
                fpath = os.path.join("rnn_models", "checkpoints", f"{trial_prefix}_weights.h5")
            else:
                fpath = os.path.join("rnn_models", "checkpoints", "weights.h5")
        
        # Model checkpoints is for the user's reference only
        model_checkpoint = ModelCheckpoint(fpath, verbose=2, mode='auto', monitor='loss', 
                                           save_best_only=True, save_weights_only=True)
        # Trainer automatically reduces learning rate when loss plateaus
        factor = 1/np.cbrt(2) # factor by which the learning rate will be reduced
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto', factor=factor,
                                      cooldown=0, min_lr=1e-4, verbose=2)
        
        return [model_checkpoint, reduce_lr]
        
    
    def __f1_score(self, y_true, y_pred):
        # Customized metrics to be added to the model compiler. The f1 score will be computed at 
        # every epoch, and will be added to the model training history for reference later.
        def recall(y_true, y_pred):
            """Recall metric.
    
            Only computes a batch-wise average of recall.
    
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall
    
        def precision(y_true, y_pred):
            """Precision metric.
    
            Only computes a batch-wise average of precision.
    
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    
    def __squeeze_excite_block(self, input):
        # Create a squeeze-excite block where input is the input tensor, and returns a keras tensor
        filters = input._keras_shape[-1] # channel_axis = -1 for TF
    
        se = GlobalAveragePooling1D()(input)
        se = Reshape((1, filters))(se)
        se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = multiply([input, se])
            
        return se

    
    def __categorize_labels(self, labels):
        # Categorize unique labels into numerical values and stores their relationship in a class 
        # dictionary object for one-hot-encoding and decoding.
        self.label_index_dict = {}
        label_categories = []
        
        index = 0
        
        for label in labels:
            if label not in self.label_index_dict:
                self.label_index_dict[label] = index
                index += 1
                
            label_categories.append(self.label_index_dict[label])
                
        return label_categories
    
    
    @staticmethod
    def save_model(model, model_name):
        """
        Saves the input keras model to the local RNN_MODEL_DIRECTORY (global)
        
        Parameter
        ---------
        model: keras.models.Model
            a trained keras model
        model_name: str
            the file name name the model will be saved as
        """
        model_dir = os.path.join(RNN_MODEL_DIRECTORY, model_name)
        
        if not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        
        model.save(os.path.join(model_dir, "model.h5"))
        
        with open(os.path.join(model_dir, 'arch.json'), 'w') as file:
            file.write(model.to_json())
            
        model.save_weights(os.path.join(model_dir, 'weights.h5'))
        
        
    


    
    