#!/c/Apps/Anaconda3/python
"""
[Title] Decision Tree Calculator
[Code Name] dt-calcu.py
[Author] Yibeck Lee(Yibeck.Lee@gmail.com)
[Comment]
	- 어쩌구 ~~~~
[History]
	- 2019-06-02 : 최초 작성
"""
print(__doc__)
from sklearn import tree
X = [[0, 0], [0,1],[1, 1]]
Y = [0, 1, 2]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, Y)
import graphviz 


with open("model_regression.txt", "w") as f:
	f = tree.export_graphviz(
		clf
	, 	out_file=f
	, 	feature_names=['V1','V1']
	)
	graph = graphviz.Source(f) 
	# graph.render('model') 

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("dtr") 

graph = graphviz.Source(dot_data)
graph.render("decition-tree") 

output = clf.predict([[2., 2.]])
print(output)
