from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import cv2 as cv
import numpy as np
import pandas as pd 
import os
import random

'''
1. Selecting Haar-like features
2. Creating an integral image
3. Running AdaBoost training
4. Creating classifier cascade

'''


path = os.getcwd()
print(path)
path_face = path + "/face.test/face/"
path_noface = path + "/face.test/non-face/"

#load data
X = []

try:
    for img in os.listdir(path_face):
        img_path = f"{path_face}{img}"
        X.append(cv.imread(img_path))

    for img in os.listdir(path_noface):
        img_path = f"{path_noface}{img}"
        X.append(cv.imread(img_path))
    
except FileNotFoundError:
    print("Unable to locate files...")

# Split the pictures in faces.train.tar.gz further into training and validation sets
X_train, X_val = train_test_split(X, test_size=0.33)



# Compute Haar-like features



# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators = 10, algorithm = {'SAMME'})


for i in range(len(abc.estimators_)):
  print(f"Tree {i}: ")
  print(f"Weight: {abc.estimator_weights_[i]}")
  print(tree.export_text(abc.estimators_[i]))