import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

breast_cancer = pd.read_csv("breast-cancer.csv")
del breast_cancer["id"]  # 由于id属于无关变量将id这一行删除
breast_cancer.drop("Unnamed: 32", axis=1, inplace=True)
dignosis_dict = {"B": 0, "M": 1}
breast_cancer["diagnosis"] = breast_cancer["diagnosis"].map(dignosis_dict)
y = breast_cancer['diagnosis']
del breast_cancer['diagnosis']
X = breast_cancer
feature_name = pd.DataFrame(X).columns
[breast_cancer_train, breast_cancer_test, breast_cancer_train_labels,
 breast_cancer_test_labels] = train_test_split(X, y, test_size=0.3, random_state=8)
model = DecisionTreeClassifier(criterion='entropy')  # 模型
model.fit(breast_cancer_train, breast_cancer_train_labels)  # 拟合模型
dtree_breast_cancer_test_pre = model.predict(breast_cancer_test)  # 对测试数据进行预测
export_graphviz(model, out_file='tree.dot', feature_names=feature_name)