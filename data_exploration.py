# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 21:29:30 2018

@author: Samantha
"""

#%matplotlib qt

import sys
import numpy as np
import matplotlib.pyplot as plt

from gestures import Gesture


def get_outlier_indices(metric_list):
    
    q75, q25 = np.percentile(metric_list, [75, 25])
    iqr = q75 - q25
    
    upper_bound = q75 + 1.5 * iqr
    lower_bound = q25 - 1.5 * iqr
    
    outlier_index = []
    
    print(f"upper bound: {upper_bound}, lower bound: {lower_bound}")
    
    for i in range(len(metric_list)):
        if metric_list[i] < lower_bound or metric_list[i] > upper_bound:
            outlier_index.append(i)
    
    return outlier_index

#%% Load all gesture instances

gesture_names = Gesture.get_all_gesture_names()

gesture_instances = {}
gesture_durations = []
gesture_labels = []

for gesture_name in gesture_names:
    
    print(f"\ncollecting data for {gesture_name}...")
    
    instances = Gesture.get_all_instance_for_gesture(gesture_name)
    gesture_instances[gesture_name] = instances

#%% Survey gestures
    
x_left_gesture_labels = []
y_left_num_instances = []
x_right_gesture_labels = []
y_right_num_instances = []
x_double_gesture_labels = []
y_double_num_instances = []

for gesture_name, instances in gesture_instances.items():
    
    
    if 'left' in gesture_name:
        x_left_gesture_labels.append(gesture_name)
        y_left_num_instances.append(len(instances))
    elif 'right' in gesture_name:
        x_right_gesture_labels.append(gesture_name)
        y_right_num_instances.append(len(instances))
    else:
        x_double_gesture_labels.append(gesture_name)
        y_double_num_instances.append(len(instances))

fig, ax = plt.subplots()
plt.title('Gesture Instance Count')
plt.xlabel('Gesture Name')
plt.ylabel('Count')
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.3)
plt.bar(x_left_gesture_labels, y_left_num_instances)
plt.bar(x_right_gesture_labels, y_right_num_instances)
plt.bar(x_double_gesture_labels, y_double_num_instances)
plt.legend(['left', 'right', 'double-handed'])

plt.show()


#%% Plot data samples out

left_hook_sample = gesture_instances['left_hooks'][0].get_training_data()
right_hook_sample = gesture_instances['right_hooks'][0].get_training_data()

left_acc = left_hook_sample.filter(regex='acc')
left_rpy = left_hook_sample.filter(regex='roll|pitch|yaw')
right_acc = right_hook_sample.filter(regex='acc')
right_rpy = right_hook_sample.filter(regex='roll|pitch|yaw')

fig, ax = plt.subplots(2, 2, sharex=True)
ax[0,0].plot(left_acc)
ax[0,0].set_ylabel('acc values')
ax[0,0].set_title('LEFT')
ax[0,0].legend(['x','y','z'])

ax[1,0].plot(left_rpy)
ax[1,0].legend(['roll','pitch','yaw'])
ax[1,0].set_ylabel('degrees')

ax[0,1].plot(right_acc)
ax[0,1].set_title('RIGHT')
ax[0,1].legend(['x','y','z'])

ax[1,1].plot(right_rpy)
ax[1,1].legend(['roll','pitch','yaw'])

fig.suptitle('Hook Gesture Sample Data')
fig.text(0.5, 0.05, 'Index #', ha='center', va='center')


#%% Survey gesture durations    

filtered_durations = []
filtered_labels = []

durations = []
labels = []

outliers = []
outlier_labels = []    

for gesture_name, instances in gesture_instances.items():
    _durations = []
    for instance in instances:
        _durations.append(instance.get_duration())
        labels.append(gesture_name)
    
    # remove outliers
    _durations = np.array(_durations)
    outlier_indices = get_outlier_indices(_durations)
    print(f"{gesture_name} total instances: {len(instances)}, num outliers: {len(outlier_indices)}")
    mask = np.ones(len(_durations), dtype=bool)
    mask[outlier_indices] = False
    _filtered_durations = _durations[mask]
    
    mask = np.zeros(len(_durations), dtype=bool) 
    mask[outlier_indices] = True
    _outliers = _durations[mask]
    
    durations += _durations.tolist()
    outliers += _outliers.tolist()
    filtered_durations += _filtered_durations.tolist()
    filtered_labels += [gesture_name for _ in range(len(_filtered_durations))]
    outlier_labels += [gesture_name for _ in range(len(_outliers))]
    print("collection complete!")

plt.scatter(outliers, outlier_labels, s=1)
plt.scatter(filtered_durations, filtered_labels, s=1)

plt.title('Outlier Removal by Duration Using IQR')
plt.legend(['Outliers', 'To keep'])
plt.xlabel('Gesture Instance Duration (seconds)')
plt.ylabel('Gesture Name')

#%% Finding max and min ranges of data

acc_means = []
rpy_mean = []
acc_max = 0
acc_min = sys.float_info.max
rpy_max = 0
rpy_min = sys.float_info.max

for gesture_name, instances in gesture_instances.items():
    
    for instance in instances:
        training_data = instance.get_training_data()
        acc = training_data.filter(regex='acc')
        rpy = training_data.filter(regex='(roll|pitch|yaw)')
        
        acc_means.append(np.mean(np.mean(np.abs(acc))))
        rpy_mean.append(np.mean(np.mean(np.abs(rpy))))
        
        _acc_max = np.max(np.max(np.abs(acc)))
        _acc_min = np.min(np.min(np.abs(acc)))
        
        _rpy_max = np.max(np.max(np.abs(rpy)))
        _rpy_min = np.min(np.min(np.abs(rpy)))
        
        if _acc_max > acc_max:
            acc_max = _acc_max
        if _acc_min < acc_min:
            acc_min = _acc_min
            
        if _rpy_max > rpy_max:
            rpy_max = _rpy_max
        if _rpy_min < rpy_min:
            rpy_min = _rpy_min
        
        
        
        
    
