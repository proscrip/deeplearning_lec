import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import skew
import os
import csv
import pandas as pd

features = pd.read_csv("C:/Users/alsgu/Downloads/train_set.csv", index_col=0)

indexes = features[(features['activity'] == 'crossTrainer')].index
features.drop(indexes , inplace=True)
indexes = features[(features['activity'] == 'stepper')].index
features.drop(indexes , inplace=True)
indexes = features[(features['activity'] == 'cyclingVertical')].index
features.drop(indexes , inplace=True)
indexes = features[(features['activity'] == 'standingInElevatorStill')].index
features.drop(indexes , inplace=True)
indexes = features[(features['activity'] == 'cyclingHorizontal')].index
features.drop(indexes , inplace=True)
indexes = features[(features['activity'] == 'basketBall')].index
features.drop(indexes , inplace=True)
indexes = features[(features['activity'] == 'rowing')].index
features.drop(indexes , inplace=True)
indexes = features[(features['activity'] == 'movingInElevator')].index
features.drop(indexes , inplace=True)
features['activity'].value_counts().plot(kind='bar', title='Training examples by activity type');

features.to_csv("C:/Users/alsgu/Downloads/features.csv", mode='w')

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import skew
import os
import csv
import pandas as pd
features_RA = features.filter(regex='RA_')
features_LA = features.filter(regex='LA_')
features_RA.insert(3, 'activity', features['activity'])
features_LA.insert(3, 'activity', features['activity'])
features_RA['activity'].value_counts()

features_LA['activity'].value_counts()


