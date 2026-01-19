from tensorflow import keras
(train_input,train_target),(test_input,test_target) =\
keras.datasets.fashion_mnist.load_data()
print(train_input.shape,train_target.shape)
print(test_input.shape,test_target.shape)

import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,10,figsize=(10,10))
for i in range(10):
  axs[i].imshow(train_input[i],cmap='gray_r')
  axs[i].axis('off')
plt.show()

print([train_target[i] for i in range(10)])

import numpy as np
print(np.unique(train_target,return_counts=True))

train_scaled=train_input/255.0
train_scaled=train_scaled.reshape(-1,28*28)
print(train_scaled.shape)

from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
SC=SGDClassifier(loss='log_loss',max_iter=5,random_state=42)
scores=cross_validate(SC,train_scaled,train_target,n_jobs=-1)
print(np.mean(scores['test_score']))

#ANN

import tensorflow as tf
from sklearn.model_selection import train_test_split
train_scaled,val_scaled,train_target,val_target= train_test_split(train_scaled,train_target,test_size=0.2,random_state=42)
print(train_scaled.shape,train_target.shape)
print(val_scaled.shape,val_target.shape)

dense=keras.layers.Dense(10,activation='softmax')
model=keras.Sequential([keras.layers.Input(shape=(784,)),dense])
model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print(train_target[:10])
model.fit(train_scaled,train_target,epochs=5)
model.evaluate(val_scaled,val_target)
