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


# get data from absolute path
def get_data():
	df = pd.read_csv('data/final_matriz.csv', index_col=0)
	return df

df = get_data()

# head and tail
print('* df.head()', df.head(), sep='\n', end='\n\n')
print('* df.head()', df.head(), sep='\n', end='\n\n')

# use pandas to show the relation types
print('* relation types:', df['parent_rel_EDU1'].unique(), sep='\n')

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
print('* df2.head()', df2[['Target', 'parent_rel_EDU1']].head(),
      sep="\n", end='\n\n')
print('* df2.tail()', df2[['Target', 'parent_rel_EDU1']].tail(),
      sep='\n', end='\n\n')
print('* targets', targets, sep='\n', end='\n\n')

# get the names of the feature columns
features = list(df2.columns[7:])
features.remove('Target')

# Preparo datos para clasificar. X = features, y = class
y = df2['Target'] #target
X = df2[features] #data

# take y, create a new variable y_elab, when value in y equals elaboration, put 1 in elab, when not, 0
df_elab = df2.copy()
df_elab['Elaboration'] = (y == 0)
y_elab = df_elab['Elaboration'] # y_elab = vector with binary labels (1 for elab., 0 for other)

# Split into training and test set (e.g., 80/20)
# X_train, X_test, y_train, y_test = train_test_split(X, y_elab, test_size=0.2, random_state=0)

# param_grid = {"max_depth": [20,25], "max_features": [60,70,80], "min_samples_split": [3,4], "criterion": ["gini", "entropy"]}

# param_grid = {"n_estimators": [500], "max_features": [20, 30], "criterion": ["entropy"], "n_jobs": [-1]}

# Chose model 
# clf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=3, scoring='roc_auc', verbose = 3)
# clf.fit(X,y_elab)

# Calculate AUC score
# fp_rate, tp_rate, thresholds = roc_curve(y_elab, clf.predict_proba(X)[:,1])
# print(auc(fp_rate, tp_rate))
# 0.569628587484
# print(roc_auc_score(y_elab, clf.predict_proba(X)[:,1]))
# 0.569628587484

models = {}
for rel in range(6):
	print(rel)
	# take y, create a new variable y_elab, when value in y equals elaboration, put 1 in elab, when not, 0
	df_elab = df2.copy()
	y_target_binary = (y == rel)
	
	# Split into training and test set (e.g., 80/20)
	X_train, X_test, y_train, y_test = train_test_split(X, y_target_binary, test_size=0.2, random_state=0)

	param_grid = {"n_estimators":[500], "max_features": [30, 50], "criterion": ["entropy"], "n_jobs": [-1]}

	# Chose model 
	models[rel] = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=10, scoring='roc_auc', verbose = 3)
	models[rel].fit(X, y_target_binary)

with open("final_models.p", "w") as f:
	pickle.dump(models, f)

# models = pickle.load(open("final_models.p"))
