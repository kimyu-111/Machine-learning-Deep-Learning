import pandas as pd

wine=pd.read_csv('http://bit.ly/wine_csv_data')
data=wine[['alcohol','sugar','pH']].to_numpy()
target=wine['class'].to_numpy()

from sklearn.model_selection import train_test_split

train_input,test_input,train_target,tesst_target=train_test_split(wine,target,train_size=0.2,random_state=42)

sub_input,val_input,sub_target,val_target=train_test_split(train_input,train_target,test_size=0.2,random_state=42)

print(sub_input.shape,val_input.shape)

from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier(random_state=42)
dt.fit(sub_input,sub_target)
print(dt.score(sub_input,sub_target))
print(dt.score(val_input,val_target))

#Cross Validation
from sklearn.model_selection import cross_validate
scores= cross_validate(dt,train_input,train_target)
print(scores)
