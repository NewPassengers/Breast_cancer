在项目中main.py为主程序，它调用knn_module和Dtree_module两个模块中经过处理过的数据，来进行显示


knn_module.py是KNN模块，其中使用max_min和One-Hot方法将数据标准化，并进行了一定的数据预处理
经过循环测试K值后，确定最佳K值为5，最终准确率在97%左右


Dtree_module.py是决策树算法模块，其数据预处理与未进行max_min前的KNN模块内的数据相同，就不加以赘述
可以在程序中更改criterion属性的取值，我们经过测试选择了信息熵，此时在各种决策树算法中分类效果最好
我们还生成了dot文件以转化为png文件 我们通过graphviz.exe在PowerShell窗口中使用命令行  dot -Tpng tree.dot -o tree.png
将tree.dot转化为tree.png文件


breast-cancer.csv 是一份来自于UCI上的公开数据集