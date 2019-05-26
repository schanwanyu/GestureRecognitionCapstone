# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:00:50 2018

@author: Samantha
"""

import numpy as np
import pandas as pd
import random

from gestures import Gesture

class CapstoneDataLoader:
    """
    Loads and preprocesses data prior to training.
    
    Attributes
    ----------
    left_data_set: dict
        contains all left hand data organized by gesture name as keys
    right_data_set: dict
        contains all right hand data organized by gesture name as keys
    double_data_set: dict
        contains all double hand data organized by gesture name as keys        
    seed: int, optional
        randomize seed used for all random methods in this object
    """
    def __init__(self, seed=45):
        self.left_data_set = {}
        self.right_data_set = {}
        self.double_data_set = {}
        self.seed = seed
        
        
    def load_gestures(self, remove_outliers=False):
        """
        Loads all gesture data from local directory.
        
        Parameters
        ----------
        remove_outliers: bool, optional
            a flag to indicate whether outlier removal is desired during data loading process
        """
        # Gets all different gesure names available in the data set
        gesture_name_list = Gesture.get_all_gesture_names()
        
        # Load all the data for each type of gesture
        for gesture_name in gesture_name_list:
            print(f"collecting data for {gesture_name}...")
            gesture_instances = Gesture.get_all_instance_for_gesture(gesture_name)
            print(f"total instances: {len(gesture_instances)}")
            
            if remove_outliers:
                # Remove outliers by IQR method
                print("removing outliers...")
                gesture_instances = self.__remove_outliers(gesture_instances, method='iqr')
                print(f"total instances after outlier removal: {len(gesture_instances)}\n")
            
            # Convert all data formats into DataFrames and store them in a list
            gesture_data_set = [instance.get_training_data() for instance in gesture_instances]
            
            # Store the list of DataFrame gestures into their respective gesture dictionaries
            if 'left' in gesture_name:
                self.left_data_set[gesture_name] = gesture_data_set
            elif 'right' in gesture_name:
                self.right_data_set[gesture_name] = gesture_data_set
            else:
                self.double_data_set[gesture_name] = gesture_data_set
    
    
    def get_hmm_data(self, hand, train_split=0.8):
        """
        Returns preprocessed and preformatted data for HMM training and testing.
        
        Parameters
        ----------
        hand: str
            this parameters indicates which dataset to return.
        train_split: float, optional
            a float between 0 and 1, indicates the proportion of the dataset that will be set 
            as training data. (1 - train_split) will be set as testing data
            
        Returns
        -------
        (list, list, list, list):
            X_train, y_train, X_test, y_test
        """
        X_train = {}
        X_test = []
        y_test = []
        y_train = []
        
        _temp_X_test = {}
        _all_train = []
        
        data_set = self.__get_hand_data(hand)
        
        # Split data into train and test sets
        for gesture_name, instances in data_set.items():
            X_train[gesture_name], _temp_X_test[gesture_name] = self.__train_test_split(instances, train_split)
            _all_train += X_train[gesture_name]
        
        # Compute the standardize parameters
        self._accel_range, self._euler_range = get_standardize_params(_all_train)
           
        # Standardize both training and testing sets
        for gesture_name, instances in X_train.items():
            X_train[gesture_name] = standardize_data(X_train[gesture_name],
                                                     self._accel_range,
                                                     self._euler_range)
        for gesture_name, instances in _temp_X_test.items():
            _temp_test = _temp_X_test[gesture_name]
            X_test += standardize_data(_temp_test,
                                       self._accel_range,
                                       self._euler_range)
            y_test += [gesture_name for _ in range(len(_temp_test))]
        
        # Shuffle data and labels in the same order
        X_test, y_test = shuffle_lists(X_test, y_test)
            
        X_list = []
        for gesture_name, instances in X_train.items():
            X_list += instances
            y_train += [gesture_name for _ in range(len(instances))]
    
        return (X_list, y_train, X_test, y_test)
                
    
    def get_lstm_data(self, hand):
        """
        Prepares training and testing data sets for LSTM Recognizer.
        
        Parameters
        ----------
        hand: str
            this parameters indicates which dataset to return.
        
        Returns
        -------
        (ndarray, list, ndarray, list):
            X_train, y_train, X_test, y_test
        """
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        data_set = self.__get_hand_data(hand)
        
        # Split data into train and test sets
        _train, _test = self.__train_test_split_lstm(data_set)
        for gesture_name, instances in _train.items():
            X_train += instances
            y_train += [gesture_name for _ in range(len(instances))]
        for gesture_name, instances in _test.items():
            X_test += instances
            y_test += [gesture_name for _ in range(len(instances))]
        
        # Shuffle the train and test data and their labels in the same order
        X_train, y_train = shuffle_lists(X_train, y_train)
        X_test, y_test = shuffle_lists(X_test, y_test)
        
        # Standardize both train and test data with the standardization parameters generated 
        # by the training set
        self._accel_range, self._euler_range = get_standardize_params(X_train)
        X_train = standardize_data(X_train, self._accel_range, self._euler_range)
        X_test = standardize_data(X_test, self._accel_range, self._euler_range)
        
        # Pad, transpose, and convert the data into numpy arrays
        X_train, X_test = CapstoneDataLoader.pad_and_transpose_keras_lstm_train_and_test(X_train, 
                                                                                         X_test)
        return (X_train, y_train, X_test, y_test)
            
    def get_all_data_unprocessed(self, hand):
        """
        Returns all data in a single list, left unprocessed
        
        Parameters
        ----------
        hand:
            this parameters determines which dataset to return.
        
        Returns
        -------
        (list, list):
            list of all data, list of corresponding labels
        """
        data_set = self.__get_hand_data(hand)
        
        # Put all data in a list
        data = []
        labels = []
        for gesture_name, instances in data_set.items():
            for instance in instances:
                data += instances
                labels += [gesture_name for _ in range(len(instances))]
        
        return (data, labels)
    
    
    def __get_hand_data(self, hand):
        # Returns the data set that belongs to the input hand type
        if hand is 'left':
            return self.left_data_set
        elif hand is 'right':
            return self.right_data_set
        else:
            return self.double_data_set
            
    
    def __remove_outliers(self, X, method='iqr'):
        # Returns a set of data with outliers removed
        length_list = []
        instances_passed = []
        
        for instance in X:
            length_list.append(instance.get_duration())
        
        # Finds the upper and lower bounds or thresholds for outliers
        if method == 'stdev':
            # sets upper and lower bounds as 2 standard deviations away from the mean in either direction
            mean_duration = np.mean(length_list)
            stdev = np.std(length_list)
            upper_bound = mean_duration + 2*stdev
            lower_bound = mean_duration - 2*stdev
            
        elif method == 'iqr':
            # sets upper and lower bounds as 1.5 times the interquartile range from the second and third quadrant
            q75, q25 = np.percentile(length_list, [75, 25])
            iqr = q75 - q25
            upper_bound = q75 + 1.5 * iqr
            lower_bound = q25 - 1.5 * iqr
        
        else:
            raise ValueError("Invalid `method` argument: must be either 'stdev' or 'iqr'.")
        
        # Remove the instances that have durations beyond the upper and lower bounds
        for instance, duration in zip(X, length_list):
            if duration <= upper_bound and duration >= lower_bound:
                instances_passed.append(instance)
            
        return instances_passed
            
    
    def __train_test_split(self, X, ratio):
        # Split input X data into train and test set according to the split ratio provided
        if self.seed is not None:
            random.seed(self.seed)
        
        random.sample(X, len(X))
    
        n_train = int(len(X) * ratio)
        
        train_set = X[:n_train]
        test_set = X[n_train:]
        
        return (train_set, test_set)
    
    
    def __train_test_split_lstm(self, data_set, num_train=400, num_test=50):
        # Split data for LSTM by making sure there is equal number of training data per gesture to 
        # avoid data imbalance and cause bias during training
        X_train = {}
        X_test = {}
        
        num_total = num_train + num_test # The total number of instances to be used from the set
        
        for gesture_name, instances in data_set.items():
            
            train_set = []
            test_set = []
            
            if self.seed:
                random.Random(self.seed).shuffle(instances)
            else:
                random.shuffle(instances)
            
            if len(instances) >= num_total:
                X_train[gesture_name] = instances[:num_train]
                X_test[gesture_name] = instances[-num_test:]
            else:
                _train_set, _test_set = self.__train_test_split(instances, 0.8)
                
                num_train_samples = len(_train_set)
                num_test_samples = len(_test_set)
                train_diff = num_train - num_train_samples
                test_diff = num_test - num_test_samples
                
                if train_diff > num_train_samples:
                    while (train_diff > num_train_samples):
                        train_set += _train_set
                        num_train_samples = len(train_set)
                        train_diff = num_train - num_train_samples
                
                if test_diff > num_test_samples:
                    while (test_diff > num_test_samples):
                        test_set += _test_set
                        num_test_samples = len(test_set)
                        test_diff = num_test - num_test_samples
                
                train_set += _train_set[:train_diff]
                test_set += _test_set[:test_diff]
                
                X_train[gesture_name] = train_set
                X_test[gesture_name] = test_set
                
        return (X_train, X_test)
    
    
    @staticmethod
    def pad_and_transpose_keras_lstm_train_and_test(X_train, X_test):
        """
        Pads and reformats the training and testing data to comply with keras model input configuration.
        
        Parameter
        ---------
        X_train: list
            a list of training data where each element is pd.DataFrame
        X_test: list
            a list of test data where each element is pd.DataFrame
        
        Returns
        -------
        (ndarray, ndarray):
            train_array, test_array
        """
        max_length = 0
        for instance in X_train:
            length = len(instance)
            if length > max_length:
                max_length = length
        num_features = X_train[0].shape[1]
        
        train_result = np.zeros((len(X_train), num_features, max_length))
        test_result = np.zeros((len(X_test), num_features, max_length))
        
        for i in range(len(X_train)):
            train_result[i,:,:len(X_train[i])] = X_train[i].T.values
        for i in range(len(X_test)):
            if len(X_test[i]) > max_length:
                test_result[i,:,:] = X_test[i].T.values[:,:max_length]
            else:
                test_result[i,:,:len(X_test[i])] = X_test[i].T.values
            
        return (train_result, test_result)


def get_standardize_params(training_data):
    """
    Computes standardization parameters accelerometer and orientation data separately. 
    This should only be run on the training data set.
    
    Parameters
    ----------
    training_data: list
        list of pd.DataFrame each containing data for a single gesture, each containing 6 columns
        first 3 columns are accelerometer data
        last 3 columns are orientation data in form of euler angles
        
    Returns
    -------
    float, float
        accel_range, euler_range
    """
    # Search for the maximum and minimum values of the data to determine the range of the data.
    # (assuming outliers have already been removed.)
    max_accel = max([instance.iloc[:,:3].max().max() for instance in training_data])
    min_accel = min([instance.iloc[:,:3].min().min() for instance in training_data])
    
    max_euler = max([instance.iloc[:,3:].max().max() for instance in training_data])
    min_euler = min([instance.iloc[:,3:].min().min() for instance in training_data])
    
    accel_range = max(abs(max_accel), abs(min_accel))
    euler_range = max(abs(max_euler), abs(min_euler))
    
    return (accel_range, euler_range)
    

def standardize_data(X, accel_range, euler_range):
    """
    Standardizes input data instances by applying the standarization parameters computed in 
    __get_standarize_params on the training data. Returns the standardized data set.
    
    Parameters
    ----------
    X: list
        a list of pd.DataFrames containing data from a single gesture instance
    accel_range: float
        the +/- range of accel data from the training set
    euler_range: list
        the +/- range of euler data from the training set
        
    Results
    -------
    list:
        list of normalized instances
    """
    norm_instances = []
    
    # Place single instance in a list for the proceeding processing step that requires 
    # instances to be iterable
    if isinstance(X, pd.DataFrame):
        X = [X]
    
    # Normalizes the data for all values to be between -1 and 1. The different data is 
    # processed in parts, and joined together before being placed in a list.
    for instance in X:
        _instance_accel = instance.iloc[:,:3].divide(accel_range)
        _instance_euler = instance.iloc[:,3:].divide(euler_range)
        
        _instance = _instance_accel.join(_instance_euler)
        _instance[_instance > 1.0] = 1.0
        
        norm_instances.append(_instance)
        
    return norm_instances


def shuffle_lists(list1, list2, seed=None):
    """
    Shuffles two lists in the same order.
    
    Parameters
    ----------
    list1: list
        the first list to be shuffled
    list2: list
        the second list to be shuffled in the same order as list1
    seed: int, optional
        the randomized seed value
    
    Returns
    -------
    (list, list):
        a tuple of the pair of shuffled lists
    """
    if len(list1) != len(list2):
        raise ValueError(f"Lists must have same length! list1 length = {len(list1)}, list2 length = {len(list2)}")
    
    combined_list = list(zip(list1, list2))
    
    if seed:
        random.Random(seed).shuffle(combined_list)
    else:
        random.shuffle(combined_list)
    
    tup1, tup2 = zip(*combined_list)
    
    return list(tup1), list(tup2)

