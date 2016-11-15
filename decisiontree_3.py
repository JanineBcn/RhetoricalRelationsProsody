'''documentation: http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html'''

from __future__ import print_function

import os
import subprocess
import pandas as pd
import numpy as np
import pydotplus 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.model_selection import train_test_split


# get data from absolute path
def get_data():
	df = pd.read_csv('/home/janine/Documents/phd/ted/limited_relations/out/limited_rel_matriz.csv', 	index_col=0)
	return df

df = get_data()

# head and tail
print('* df.head()', df.head(), sep='\n', end='\n\n')
print('* df.head()', df.head(), sep='\n', end='\n\n')

# use pandas to show the relation types
print('* relation types:', df['parent_rel_EDU1'].unique(), sep='\n')

# delete unnecessary columns with strings
del df['spk_EDU1']
del df['conv_EDU1']
del df['parent_rel_EDU2']
del df['spk_EDU2']
del df['conv_EDU2']


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

# fitting decision tree with scikit-learn
y = df2['Target']
X = df2[features]

# Split into training and test set (e.g., 80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = tree.DecisionTreeClassifier(max_features=60, min_samples_split=2, criterion='entropy', max_depth=5)

clf.fit(X_train,y_train)
print(clf.score(X_test, y_test))


# visualize the tree
def visualize_tree(tree, feature_names, class_names):
	"""Create tree png using graphviz.

	Args
	----
	tree -- scikit-learn DecsisionTree.
	feature_names -- list of feature names.
	"""
	with open('clf.dot', 'w') as f:
		export_graphviz(tree, out_file=f,
				feature_names=feature_names,
				filled=True, rounded=True,
				class_names = class_names)
	command = ['dot', '-Tpng', 'clf.dot', '-o', 'clf.png'] #???
	try:
		subprocess.check_call(command)
	except:
		exit('Could not run dot, ie graphviz, to produce visualization')


visualize_tree(clf, features, targets)








	
