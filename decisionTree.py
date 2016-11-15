import pandas as pd
import numpy as np
import csv as csv
import random
from operator import itemgetter
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV

# Load File and store in data array
csv_file_object = csv.reader(open('C://Users//Januna//Desktop//phd//Ted//pros_test//out//5.csv', 'rb'))
header = csv_file_object.next()
data = []
for row in csv_file_object:
	data.append(row)
data = np.array(data)

# Me quedo con solo x instancias para usar los ultimos como validacion
random.seed(10)
indexes = random.sample(xrange(182),172)
txt = itemgetter(*indexes)(data) #?

# Armar dataset de pandas
df = pd.DataFrame(data)

# Delete unrequired columns and those which contain strings
del df[117]
del df[118]
del df[119]
del df[120]
del df[121]
del df[122]

# Skip the first 6 features (contain string and unrequired information)
features = df.columns[7:]

# Preparo datos para clasificar. X = features, y = class
X = df[features].values
y = df[1]

# Chose Decision Tree
clf = DecisionTreeClassifier()

param_grid = {"max_depth": [10, 5, 3, None], "max_features": [1, 3, 10, 20, None], "min_samples_split": [1, 2, 3, 5, 10], "criterion": ["gini", "entropy"]}

grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
grid_search.fit(X,y)

print grid_search.best_score_
print grid_search.best_params_
# Results
# 0.462121212121
# {'max_features': 1, 'min_samples_split': 5, 'criterion': 'gini', 'max_depth': 3}