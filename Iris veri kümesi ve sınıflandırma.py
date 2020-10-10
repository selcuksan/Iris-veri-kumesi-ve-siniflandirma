import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score

#VERİ YÜKLEME
df = pd.read_csv("Iris.csv")

df = df.drop(["Id"],axis=1)
x = df.values[:,0:4]
y = df.values[:,4:]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)


print("veri setinin içeriği")
print(df.head(10))

print("veri setinin sayısal istatistikleri")
print(df.describe().T)

print("verilerin tür değişkenine göre dağılımı")
print(df.groupby("Species").size())

models = [("lr",LogisticRegression(random_state=0,max_iter=1200000)),
          ("knn",KNeighborsClassifier(n_neighbors=5,metric="minkowski")),
          ("dt",DecisionTreeClassifier(criterion="gini")),
          ("nb",GaussianNB()),
          ("svm",SVC(kernel="rbf")),
          ("rf",RandomForestClassifier(random_state=0,n_estimators=10))]


sonuclar = []
isimler = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_sonuc = cross_val_score(model,x_train,y_train.ravel(),cv=kfold,scoring="accuracy")
    sonuclar.append(cv_sonuc)
    isimler.append(name)
    print(f"{name}: {np.mean(sonuclar)}")


knn = KNeighborsClassifier(n_neighbors=5,metric="minkowski")
knn.fit(x_train,y_train.ravel())
tahmin = knn.predict(x_test)

from sklearn.metrics import accuracy_score
print("accuracy degeri: ",accuracy_score(y_test,tahmin))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,tahmin))

from sklearn.metrics import classification_report
print(classification_report(y_test,tahmin))

