import os
import pandas as pd

DATA_DIRECTORY = os.path.join("data", "gesture", "ssf_gestures_2013")


class Gesture():
    """
    A getsure class object that is responsible for selectively reading data from left, right, or 
    double handed gesture folders, containing and organizing the data before it is preprocessed. 
    Each class object will hold data for only one single gesture instance.
    """
    def __init__(self, gesture_name=None, gesture_id=None, time_stamps=None, left_data=None, right_data=None):
        self.__check_data_valid(time_stamps, left_data, right_data)
        
        self.gesture_name = gesture_name
        self.gesture_id = gesture_id
        
        self.time = time_stamps
        self.left_data = left_data
        self.right_data = right_data
        
        if gesture_name is None:
            
            if self.left_data is None and self.right_data is not None:
                self.gesture_type = 'R'
            elif self.left_data is not None and self.right_data is None:
                self.gesture_type = 'L'
            elif self.left_data is not None and self.right_data is not None:
                self.gesture_type = 'LR'
            else:
                raise Exception("Unable to identify gesture type. Please provide valid data.")
        else:
            self.gesture_type = Gesture.get_gesture_type_by_gesture_name(gesture_name)
    
    
    def get_training_data(self):
        """
        Training data is returned, and depending on the whether or not left, right or data for 
        both hands are present, a join operation may be applied.
        
        Returns
        -------
        pd.Dataframe:
            dataframe that contains all sequence data for this instance
            (row: time, columns: features)
        """
        train_data = None
        
        if self.left_data is not None:
            train_data = self.left_data
            
        if self.right_data is not None:
            if train_data is not None:
                train_data = train_data.join(self.right_data)
            else:
                train_data = self.right_data
                
        return train_data
        
    
    def get_gesture_type(self):
        """
        Determines whether this gesture is a left, right, or double handed gesture judging from 
        the amount of data read from file.
        
        Returns
        -------
        str:
            symbol for type of data
            'L': left handed
            'R': right handed
            'LR': double handed
        """
        if self.left_data is not None and self.right_data is None:
            return 'L'
        elif self.left_data is None and self.right_data is not None:
            return 'R'
        elif self.left_data is not None and self.right_data is not None:
            return 'LR'
        else:
            raise ValueError("Gesture object is empty.")

    
    def get_duration(self):
        """
        Returns the total duration of the instance.
        
        Returns
        -------
        float:
            instance duration in seconds
        """
        return float(self.time.iloc[-1] - self.time.iloc[0])

    def __check_data_valid(self, time_stamps, left_data=None, right_data=None):
        # Checks that the data instance object contains valid data (i.e. data without missing
        # values or unequal number of data points)
        valid = False
        
        if left_data is not None and right_data is not None:
            valid = len(time_stamps) == len(right_data) \
                    and len(time_stamps) == len(left_data)
                    
        elif left_data is not None:
            valid = len(time_stamps) == len(left_data)
            
        elif right_data is not None:
            valid = len(time_stamps) == len(right_data)
        
        if not valid:
            raise Exception("Invalid data detected. All data must have same lengths.")
            
            
    @staticmethod
    def get_all_gesture_names():
        """
        Returns all available gesture names from the data directory.
        
        Returns
        -------
        list:
            list of strings that are gesture names, which are also the folder name that contains 
            that gesture set
        """
        return [name for name in os.listdir(DATA_DIRECTORY) 
                if os.path.isdir(os.path.join(DATA_DIRECTORY, name))]
    
    
    @staticmethod
    def get_gesture_type_by_gesture_name(gesture_name):
        """
        Determine gesture type by the gesture name
        
        Returns
        -------
        str:
            symbol for type of data
            'L': left handed
            'R': right handed
            'LR': double handed
        """
        if "left" in gesture_name:
            return 'L'
        elif "right" in gesture_name:
            return 'R'
        else:
            return 'LR'
        
        
    @staticmethod
    def list_files_for_gesture(gesture_name):
        # Returns all filenames that are contained in the gesture folder.
        return [fname for fname in os.listdir(os.path.join(DATA_DIRECTORY, gesture_name)) 
                 if fname.endswith('.ins')]
    
    @staticmethod
    def get_all_instance_for_gesture(gesture_name):
        """
        Returns all instance of the gesture in this Gesture class object by walking through the 
        gesture directory and reading each existing data file.
        
        Returns
        -------
        list:
            list of gesture instances
        """
        files = Gesture.list_files_for_gesture(gesture_name)
        
        gesture_list = []
        
        for fname in files:
            
            fpath = os.path.join(DATA_DIRECTORY, gesture_name, fname)
            
            left_data = None
            right_data = None
            time_data = None
            
            __is_first_line = True
            
            with open(fpath, 'r') as f:
                
                __line_data = []
                
                for line in f:
                    
                    if __is_first_line:
                        glove, gesture_id = line.split(' ')
                        __is_first_line = False
                        
                    else:
                        
                        if 'LEFT' in line:
                            left_data = []
                            __line_data = left_data
                        elif 'RIGHT' in line:
                            right_data = []
                            __line_data = right_data
                        elif 'TIME' in line:
                            time_data = []
                            __line_data = time_data
                        else:
                            line = line.rstrip()
                            __line_data.append(map(float, line.split(',')))
            
            if left_data is not None:
                __col_labels = ['', '', '',
                              'left_acc_x','left_acc_y','left_acc_z',
                              'left_yaw','left_pitch','left_roll']
                left_data = pd.DataFrame(left_data, columns=__col_labels).drop([''], axis=1)
                
            if right_data is not None:
                __col_labels = ['', '', '',
                              'right_acc_z','right_acc_y','right_acc_z',
                              'right_yaw','right_pitch','right_roll']
                right_data = pd.DataFrame(right_data, columns=__col_labels).drop([''], axis=1)
                
            time_data = pd.DataFrame(time_data, columns=['time'])
            gesture_list.append(Gesture(gesture_name, gesture_id,
                                        time_data, left_data, right_data))
            
        return gesture_list
        
        
    
