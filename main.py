import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report ,accuracy_score
import numpy as np


data=pd.read_csv('data.csv')

#Data Exploration

print(data.describe())
print(data.info())
print(data.isnull().sum())
print(data['diagnosis'].value_counts(normalize=True))
data.drop(['id','Unnamed: 32'],axis=1,inplace=True)
print(data.head())

fields=data.drop(['diagnosis'],axis=1)
y = (data['diagnosis'] == 'M').astype(int)
correlations = fields.corrwith(y)
correlations.plot(kind='bar')
plt.show()


drop_list=correlations[correlations<0.2].index
Lable=LabelBinarizer()
data['diagnosis']=Lable.fit_transform(data['diagnosis'])
print(data['diagnosis'].value_counts())
data.drop(drop_list,axis=1,inplace=True)
print(data)

#Visulizing by Heatmap
predictors=data.columns[1:]
plt.figure(figsize=(18,18))
sns.heatmap(data[predictors].corr(),linewidths = 1, annot = True, fmt = ".2f")
plt.show()
for col in predictors:
    sns.scatterplot(x=data[col],y=data['diagnosis'])
    plt.show()


#Modeling
X=data.iloc[:,1:]
Y=data['diagnosis']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,random_state=42)

#trying k in range(1:15)
error1=[]
error2=[]
for k in range(1,15):
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred1= knn.predict(x_train)
    error1.append(np.mean(y_train!= y_pred1))
    y_pred2= knn.predict(x_test)
    error2.append(np.mean(y_test!= y_pred2))
# plt.figure(figsize(10,5))
plt.figure('test K')
plt.plot(range(1,15),error1,label="train")
plt.plot(range(1,15),error2,label="test")
plt.xlabel('k Value')
plt.ylabel('Error')
plt.legend()
plt.show()

#Final Model with K = 11 is the optimal

Knn=KNeighborsClassifier(n_neighbors=11)
Knn.fit(x_train,y_train)
y_pred=Knn.predict(x_test)
print(classification_report(y_test, y_pred))
#print('Accuracy={:.2%}'.format(accuracy_score(y_test,y_pred)))

