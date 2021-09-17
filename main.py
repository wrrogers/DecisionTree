import pandas as pd
import numpy as np
import titanic
import pprint
from DecisionTree import *
from sklearn.metrics import accuracy_score

X, y, a = titanic.get_data()

dt = DecisionTree()

tree = dt.fit(X, y, a)

records = X

preds = dt.predict(records, a, tree)

#pprint(tree)
#print("\n", preds, "\n")

print("THE ACCURACY IS:")
print(accuracy_score(y, preds))