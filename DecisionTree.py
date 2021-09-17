import pandas as pd
import numpy as np
from pprint import pprint

class DecisionTree(object):
    def __init__(self):
        print("Modeling...\n")
        self.preds = np.array([])
    
    def partition(self, a):
        return {c: (a==c).nonzero()[0] for c in np.unique(a)}
    
    def entropy(self, s):
        res = 0
        val, counts = np.unique(s, return_counts=True)
        freqs = counts.astype('float')/len(s)
        for p in freqs:
            if p != 0.0:
                res -= p * np.log2(p)
        return res
    
    def mutual_information(self, y, x):
        res = self.entropy(y)
    
        # We partition x, according to attribute values x_i
        val, counts = np.unique(x, return_counts=True)
        freqs = counts.astype('float')/len(x)
    
        # We calculate a weighted average of the entropy
        for p, v in zip(freqs, val):
            res -= p * self.entropy(y[x == v])
    
        return res
    
    def impure(self, d):
        return len(set(d)) == 1
    
    def empty(self, d):
        return len(d) == 0
    
    def fit(self, x, y, a):
        #print("\n")
        # If there could be no split, just return the original set
        if self.impure(y) or self.empty(y):
            #print("IT\'S PURE")
            return y
        
        # We get attribute that gives the highest mutual information
        gain = np.array([self.mutual_information(y, x_attr) for x_attr in x.T])
        #print("Gain:", gain)
        selected_attr = np.argmax(gain)
        attr = a[selected_attr]
        #print("Selected attribute:",attr)
    
        # If there's no gain at all, nothing has to be done, just return the original set
        if np.all(gain < 1e-6):
            #print("NO GAIN")
            return y
    
        # We split using the selected attribute
        #print("Partitioning:", x[:, selected_attr])
        sets = self.partition(x[:, selected_attr])
        #print("Sets:", sets)
    
        tree = {}
        #Loop through all the partitions.  k is the attribute class v is the location
        for k, v in sets.items():
            #print("loop... partition:", k,", locations",v)
            y_subset = y.take(v, axis=0)
            #print("subset y:", y_subset)
            x_subset = x.take(v, axis=0)
            #print("subset x:\n", x_subset)
            #res["If {} == {}".format(attr, k)] = recursive_split(x_subset, y_subset, a)
            tree[(attr, k.item())] = np.max(self.fit(x_subset, y_subset, a))
            
        return tree
    
    def find_node(self, x, attrs, tree):
        #print(x)
        #print("First", x['first'])
        #print("Second", x['second'], "\n")
        
        for branch, pred in tree.items():
            if branch[1] == x[branch[0]]:
                if type(pred) == type(dict()):
                    #print("---> Pass branch")
                    self.find_node(x, attrs, pred)
                else:
                    #print("----->", branch, pred)
                    self.preds = np.append(self.preds, pred)
                    break
            #else:
                #print("--> Neither")
        return self.preds
    
    def predict(self, x, attrs, tree):
        df = pd.DataFrame(x, index = None, columns = attrs)
        for idx, row in df.iterrows():
            self.find_node(row, attrs, tree)
        
        return self.preds