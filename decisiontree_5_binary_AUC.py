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
	df = pd.read_csv('data/limited_rel_matriz.csv', index_col=0)
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
#	0: 'elaboration_LeftToRight', 63.107
#	1: 'attribution_RightToLeft', 20.563
#	2: 'attribution_LeftToRight',  4.745
#	3: 'condition_RightToLeft',    3.974
#	4: 'background_LeftToRight',   3.548
#	5: 'condition_LeftToRight',    2.457
#	6: 'explanation_LeftToRight',  1.601

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

total = y.shape[0]
print(y.value_counts())
for i in y.value_counts():
	print(100*(float(i)/float(total)))

# take y, create a new variable y_elab, when value in y equals elaboration, put 1 in elab, when not, 0
df_elab = df2.copy()
df_elab['Elaboration'] = (y == 0)
y_elab = df_elab['Elaboration'] # y_elab = vector with binary labels (1 for elab., 0 for other)

models = {}
for rel in range(6):
	print(rel)
	# take y, create a new variable y_elab, when value in y equals elaboration, put 1 in elab, when not, 0
	df_elab = df2.copy()
	y_target_binary = (y == rel)
	
	# Split into training and test set (e.g., 80/20)
	X_train, X_test, y_train, y_test = train_test_split(X, y_target_binary, test_size=0.2, random_state=0)

	param_grid = {"n_estimators":[500], "max_features": [40], "criterion": ["entropy"], "n_jobs": [-1]}

	# Chose model 
	models[rel] = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=10, scoring='roc_auc', verbose = 3)
	models[rel].fit(X, y_target_binary)

with open("final_models.p", "w") as f:
	pickle.dump(models, f)

# models = pickle.load(open("final_models1.p"))
