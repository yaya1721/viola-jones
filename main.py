from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd 
import os
import random
from features import *

'''
Viola-Jones Algorithms
1. Selecting Haar-like features
2. Creating an integral image
3. Running AdaBoost training
4. Creating classifier cascade

'''


path = os.getcwd()
print(path)
path_face = path + "/face.test/face/"
path_noface = path + "/face.test/non-face/"

# Convert to integral images
# Compute Haar-like features

# [Haar] 1 rectagles 
# Vertical (white-black)

# [Haar] 2 rectagles 
# Horizontal (w-b)


# [Haar] 3 rectagles 
# Horizontal (w-b-w)

# [Haar] 4 rectagles 
# Matrix (w-b, b-w)


X = []
y = []


try:
    for img in os.listdir(path_face):
        img_path = f"{path_face}{img}"
        X.append(cv.imread(img_path))
        y.append(1)
        break
        

    for img in os.listdir(path_noface):
        img_path = f"{path_noface}{img}"
        X.append(cv.imread(img_path))
        y.append(-1)
        break
    
except FileNotFoundError:
    print("Unable to locate files...")

#image.shape = (19, 19, 3)
print(X)

# Split the pictures in faces.train.tar.gz further into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)



# Create AdaBoost classifer object
abc = AdaBoostClassifier(n_estimators = 10, algorithm = 'SAMME')

abc.fit(X_train, y_train)


# Train classfier cascade

for i in range(len(abc.estimators_)):
    print(f"Tree {i}: ")
    print(f"Weight: {abc.estimator_weights_[i]}")
    print(tree.export_text(abc.estimators_[i]))


f = abc.feature_importances_
print([i for i in range(len(f)) if f[i]!=0])
