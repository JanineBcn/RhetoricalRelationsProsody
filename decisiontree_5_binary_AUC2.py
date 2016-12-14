'''documentation: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
'''
from __future__ import print_function
import pickle
import os
import subprocess
import pandas as pd
import numpy as np
import pydotplus 
import matplotlib.pyplot as plt
from sklearn import tree
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Learn to predict each class against the other
clf = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, verbose = 3))

y_score = clf.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class 
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
	fpr[i], tpr[i], _ = roc_curve(y_test[:, 1], y_score[:, 1])
	roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



# Plot ROC curves

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
lw = 2 #?????????????

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()






with open("final_models_svm.p", "w") as f:
	pickle.dump(models, f)

# models = pickle.load(open("final_models1.p"))
