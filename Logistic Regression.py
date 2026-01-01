import pandas as pd
fish=pd.read_csv('http://bit.ly/fish_csv_data')
fish.head()
print(pd.unique(fish['Species']))
fish_input=fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target=fish['Species'].to_numpy()
from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target=train_test_split(fish_input,fish_target,random_state=42)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled=ss.transform(train_input)
test_scaled=ss.transform(test_input)

from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled,train_target)
print(kn.score(train_input,train_target))
print(kn.score(test_input,test_target))

print(kn.classes_) #check target
print(kn.predict(test_scaled[:5]))

import numpy as np
proba=kn.predict_proba(test_scaled[:5])
print(np.round(proba,decimals=4))

distances,indexes= kn.neighbors(test_scaled[3:4])
print(train_target[indexes])

import matplotlib.pyplot as plt

z=np.arange=(-5,5,0.1)
phi= 1/(1+np.exp(-z))
plt.plot(z,phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

#Boolean Indexing 
bream_smelt_indexes=(train_target =='Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt=train_target[bream_smelt_indexes]

#Logistic Regression

from sklearn.linear_model import LogisticRegression
lr= LogisticRegression(C=20,max_iter=1000)
lr.fit(train_scaled,train_target)
print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))

print(lr.predict(test_scaled[:5]))

proba=lr.predict(test_scaled[:5])
print(np.round(proba,decimals=3))
