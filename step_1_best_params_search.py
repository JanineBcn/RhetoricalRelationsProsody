'''documentation: http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html'''
from __future__ import print_function
import pickle
import os
import subprocess
import pandas as pd
import numpy as np
import pydotplus 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# get data from absolute path
def get_data():
    df = pd.read_csv('data/small_limited_matriz.csv', index_col=0)
    return df

df = get_data()

# delete unnecessary columns with strings and other irrelevant information
del df['spk_EDU1']

del df['conv_EDU1']
del df['parent_rel_EDU2']
del df['spk_EDU2']
del df['conv_EDU2']
del df['starttime_EDU2']
del df['parent.id_EDU2']
del df['endtime_EDU2']
del df['nwords_EDU1']
del df['nwords_EDU2']

# Preprocessing: to pass data into scikit-learn, encode relation_types to integers
# write function and return modified data frame and list of class names
# maps target names to numbers according to the order they appear in df
def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
             new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)} #???
    df_mod['Target'] = df_mod[target_column].replace(map_to_int)
    
    return (df_mod, targets)

# show name and target column
df2, targets = encode_target(df, 'parent_rel_EDU1')

# get the names of the feature columns
features = list(df2.columns[7:])
features.remove('Target')

# Preparo datos para clasificar. X = features, y = class
y = df2['Target'] #target
X = df2[features] #data

total = y.shape[0]
counts = y.value_counts()
counts.sort_index(0, inplace=True)

freq = y.value_counts(normalize=True)
freq.sort_index(0, inplace=True)

print(pd.DataFrame({'counts': counts, 'frequency': freq}))


#array(['contrast', 'elaboration_LeftToRight', 'attribution_RightToLeft',
#      'explanation_LeftToRight', 'attribution_LeftToRight',
#      'enablement_LeftToRight', 'background_RightToLeft',
#      'background_LeftToRight', 'contrast_RightToLeft',
#      'condition_RightToLeft', 'manner-means_LeftToRight',
#      'condition_LeftToRight'], dtype=object)


# Binarize the output
y = label_binarize(y, classes=[0,1,2,3,4,5,6,7,8,9,10,11])
n_classes = y.shape[1]

# Split into training and test set (e.g., 80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

param_grid = {
   "estimator__n_estimators": [100, 500], 
   "estimator__max_features": [75, 100, 125], 
   "estimator__criterion": ["entropy"], 
   "estimator__n_jobs": [-1]
}


# Chose model

grid = GridSearchCV(OneVsRestClassifier(RandomForestClassifier()), param_grid=param_grid, scoring='roc_auc', cv=10, verbose = 3)
grid.fit(X_train, y_train)

best_params = grid.best_params_

# save the best params 
with open("best_params_by_grid_search.p", "w") as f:
    pickle.dump(best_params, f)






















