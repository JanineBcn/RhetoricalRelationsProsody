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
   "estimator__n_estimators": [50, 100], 
   "estimator__max_features": [15, 30, 45, 60, 75], 
   "estimator__criterion": ["entropy"], 
   "estimator__n_jobs": [-1]
}


# Chose model

#clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=500, max_features="sqrt", criterion="entropy", n_jobs=-1, verbose=3))
clf = GridSearchCV(OneVsRestClassifier(RandomForestClassifier()), param_grid=param_grid, scoring='roc_auc', cv=10, verbose = 3)
clf.fit(X_train, y_train)

with open("best_model_of_grid_search.p", "w") as f:
    pickle.dump(clf.best_estimator_, f)
# Para abrir esto:
# with open("best_model_of_grid_search.p", "r") as f:
#    best_classifier = pickle.load(f)
# Para mirar uno de los clasificadores dentro del OvsRest classifier:
#   best_classifier.estimators_[0]








# score = clf.score(X_test, y_test)

#scores = []
#for i in range(len(clf.estimators_)):
#   
#    random_forest = clf.estimators_[i]
#    y_pred = random_forest.predict(X_test) # tomar x_test y comparar y_pred con y_test
#    scores.append(roc_auc_score(y_test[:,i], y_pred))

#print(scores)

#cv_score = cross_val_score(clf, X, y, cv=10, scoring="roc_auc")
#print(cv_score)

# with open("final_models.p", "w") as f:
    # pickle.dump(models, f)
