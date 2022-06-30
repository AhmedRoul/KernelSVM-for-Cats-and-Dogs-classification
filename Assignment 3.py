#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm 
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier

img = imread(r'C:\Users\Almodather\Desktop\Ass ML\dogs-vs-cats\train\cat.1.jpg')



address=r'C:\Users\Almodather\Desktop\Ass ML\dogs-vs-cats\train\{name}.{number}.jpg'
data=[]


for i in range(1100):
    cat=imread(address.format(number=i,name='cat'))
    dog=imread(address.format(number=i,name='dog'))
    
    resize_cat=resize(cat,(128,64))
    resize_dog=resize(dog,(128,64))
    
    
    
    fd_cat, hog_image_cat = hog(resize_cat, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=True, multichannel=True)
    
    fd_dog, hog_image_dog = hog(resize_dog, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=True, multichannel=True)   
   
    
    data.append(np.append(fd_cat,[1]))
    data.append(np.append(fd_dog,[-1]))
    
    
   
   




# In[ ]:



Y=np.array(data)[:,-1]
X=np.array(data)[:,:-1]
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=10)
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=1).fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=5, C=3).fit(X_train, y_train)
sigmoid=svm.SVC(kernel='sigmoid',C=3).fit(X_train, y_train)

sigmoid_pred=sigmoid.predict(X_test)
sigmoid_accuracy_train=sigmoid.score(X_train, y_train)

poly_pred = poly.predict(X_test)
poly_accuracy_train=poly.score(X_train, y_train)
                    
rbf_pred = rbf.predict(X_test)
rbf_accuracy_train=rbf.score(X_train, y_train)
                    
sigmoid_pred_accuracy_train=accuracy_score(y_test,sigmoid_pred)
                    
poly_accuracy = accuracy_score(y_test, poly_pred)
rbf_accuracy =accuracy_score(y_test,rbf_pred)
                    
print('Accuracy (Polynomial Kernel):Test ', "%.2f" % (poly_accuracy*100))
print('Accuracy (Polynomial Kernel):train  ', "%.2f" % (poly_accuracy_train*100)) 
                    
print('Accuracy (rbf Kernel):train ', "%.2f" % (rbf_accuracy_train*100))            
print('Accuracy (rbf Kernel): Test', "%.2f" % (rbf_accuracy*100))
                    
print('Accuracy (sigmoid Kernel): ', "%.2f Test" % (sigmoid_pred_accuracy_train*100))
print('Accuracy (sigmoid Kernel): ', "%.2f Train" % (sigmoid_pred_accuracy*100))


# In[ ]:




