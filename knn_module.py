import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def min_max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())


breast_cancer = pd.read_csv("breast-cancer.csv")
del breast_cancer["id"]  # 由于id属于无关变量将id这一行删除
breast_cancer.drop("Unnamed: 32", axis=1, inplace=True)
dignosis_dict = {"B": 0, "M": 1}
breast_cancer["diagnosis"] = breast_cancer["diagnosis"].map(dignosis_dict)

for col in breast_cancer.columns[1:31]:
    breast_cancer[col] = min_max_normalize(breast_cancer[col])

y = breast_cancer['diagnosis']
del breast_cancer['diagnosis']
X = breast_cancer
feature_name = pd.DataFrame(X).columns
[breast_cancer_train, breast_cancer_test, breast_cancer_train_labels,
 breast_cancer_test_labels] = train_test_split(X, y, test_size=0.3, random_state=8)
"""max_score=0 #若想测试最佳K值可将此段注释取消
min_M=1
for i in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(breast_cancer_train, breast_cancer_train_labels)
    knn_breast_cancer_test_pre = knn_model.predict(breast_cancer_test)
    score=metrics.accuracy_score(breast_cancer_test_labels, knn_breast_cancer_test_pre)
    print(score)
    M=metrics.confusion_matrix(breast_cancer_test_labels, knn_breast_cancer_test_pre)
    print(M[1][0]/M[1][1])
    print("k="+str(i)+"\n")
    if max_score<score:
        max_score=score
        x=i
    if min_M>M[1][0]/M[1][1]:
        min_M=M[1][0]/M[1][1]
        y=i
print(x, y)"""
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(breast_cancer_train, breast_cancer_train_labels)
knn_breast_cancer_test_pre = knn_model.predict(breast_cancer_test)
