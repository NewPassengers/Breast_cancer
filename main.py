from sklearn import metrics
from sklearn.tree import export_graphviz

from Dtree_module import breast_cancer_test_labels, dtree_breast_cancer_test_pre, model ,feature_name
from knn_module import knn_breast_cancer_test_pre
#决策树结果显示
export_graphviz(model, out_file='tree.dot', feature_names=feature_name)
print(metrics.classification_report(breast_cancer_test_labels, dtree_breast_cancer_test_pre))
print(metrics.confusion_matrix(breast_cancer_test_labels, dtree_breast_cancer_test_pre))
print(metrics.accuracy_score(breast_cancer_test_labels, dtree_breast_cancer_test_pre))
M=metrics.confusion_matrix(breast_cancer_test_labels, dtree_breast_cancer_test_pre)
print(M[1][0]/M[1][1])
print(metrics.classification_report(breast_cancer_test_labels, knn_breast_cancer_test_pre))
print(metrics.confusion_matrix(breast_cancer_test_labels, knn_breast_cancer_test_pre))
print(metrics.accuracy_score(breast_cancer_test_labels, knn_breast_cancer_test_pre))
M=metrics.confusion_matrix(breast_cancer_test_labels, knn_breast_cancer_test_pre)
print(M[1][0]/M[1][1])