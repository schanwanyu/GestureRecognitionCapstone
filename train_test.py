#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 17:05:57 2018

@author: schan
"""

from recognizers import LstmRecognizer, HmmRecognizer
from preprocessing import CapstoneDataLoader
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Initialize and Train LSTM Recongizer
    
    data_loader = CapstoneDataLoader(seed=45)
    data_loader.load_gestures(remove_outliers=True)
    
    hand_set = ['left','right','double']
    
    
    #%% Single Optimized LSTM Model
    lstm_accuracy = {}
    lstm_f1 = {}
    lstm_history = {}
    
    epochs = {'left':100, 'right':100, 'double':100}
    for hand in hand_set:
        X_train, y_train, X_test, y_test = data_loader.get_lstm_data(hand)
        model_name = hand + "_cnn_lstm"
        lstm_recognizer = LstmRecognizer(epochs[hand])
        model = lstm_recognizer.fit(X_train, y_train, trial_prefix=model_name, save=True)
        
        lstm_accuracy[hand] = lstm_recognizer.score(X_test, y_test)
        lstm_f1[hand] = lstm_recognizer._f1_score
        lstm_history[hand] = lstm_recognizer.history
    
    # Plot training curve
    for hand in hand_set:
        plt.plot(lstm_history[hand].history['acc'])
        plt.plot(lstm_history[hand].history['val_acc'])
        plt.title('Double Gesture Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='lower right')
        plt.show()
    
    #%% Single Optimized HMM
    hmm_accuracy = {}
    hmm_f1 = {}
    
    for hand in hand_set:
        
        X_train, y_train, X_test, y_test = data_loader.get_hmm_data(hand)
        hmm_recognizer = HmmRecognizer(hand=hand, num_states=5, topology='left-to-right-1')
        hmm_recognizer.fit(X_train, y_train, save=True)
        hmm_recognizer.load_models()
        hmm_accuracy[hand] = hmm_recognizer.score(X_test, y_test, score_type='accuracy')
        hmm_f1[hand] = hmm_recognizer.score(X_test, y_test, score_type='f1')
        
    #%% Load data for grid search
    X, y = data_loader.get_all_data_unprocessed(hand)
    cv = KFold(n_splits=3, shuffle=True)
    
    lstm_grid = {}
    lstm_grid_results = {}
    lstm_grid_highest_score = {}
    lstm_grid_params = {}
    
    hmm_grid = {}
    hmm_grid_results = {}
    hmm_grid_highest_score = {}
    hmm_best_params = {}
    
    #%%
    # LSTM Grid Search
    param_grid = {'epochs':[50, 100, 300, 500, 1000]}
    
    for hand in hand_set: 
        model_name = hand + "_cnn_lstm"
        grid = GridSearchCV(estimator=LstmRecognizer(), param_grid=param_grid, cv=cv, verbose=2)
        grid.fit(X, y)
        lstm_grid[hand] = grid
        lstm_grid_results[hand] = grid.cv_results_
        lstm_grid_highest_score[hand] = grid.best_score_
        lstm_grid_params[hand] = grid.best_params_
    
    #%%
    # HMM Grid Search
    param_grid = {'num_states': [4, 5, 6, 7, 8],
                  'topology': ['full', 'left-to-right-full', 'left-to-right-1']}
    
    for hand in hand_set:
        grid = GridSearchCV(estimator=HmmRecognizer(hand=hand), param_grid=param_grid, cv=cv, verbose=10)
        grid.fit(X, y)
        hmm_grid[hand] = grid
        hmm_grid_results[hand] = grid.cv_results_
        hmm_grid_highest_score[hand] = grid.best_score_
        hmm_best_params[hand] = grid.best_params_
    