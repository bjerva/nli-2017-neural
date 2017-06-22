import json, sys, pickle
import numpy as np
from collections import Counter
from operator import itemgetter
#from sklearn.linear_model import LogisticRegression
from sklearn import linear_model, datasets
from sklearn.preprocessing import Normalizer


matrix = []
labels = []

with open("matrix_X.file", "r") as xfile:
    matrix = json.load(xfile)
xfile.close()

with open("document_label_Y.file", "r") as yfile:
    labels = json.load(yfile)
yfile.close()

X = np.array(matrix,dtype=np.float64)

scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)


Y = np.array(labels)

lr = linear_model.LogisticRegressionCV()
lr_train = lr.fit(normalizedX, Y)

vector = []
ivector = {}

with open("feature_names.file", "r") as ffile:
    data = json.load(ffile)
    for el in data:
        vector.append("$".join(el))
        ivector = dict.fromkeys(vector,0)
ffile.close()

## load data for predicting



#print(matrix)
#print(X)
print(X.shape)
#print(normalizedX)
print(normalizedX.shape)
#print(labels)
#print(Y)
#print(test_data)

test_batch = {}
for n in sys.argv[1:]:
    with open(n, "r") as testfile:
        test_vector = json.load(testfile)
        vec = []
        for el in test_vector:
            vec.append("$".join(el))
        ivec = Counter(vec)
        ivec0=dict.fromkeys(vector,0)
        for k in ivec.keys():
            if k in ivec0:
                ivec0[k]=ivec[k]
        #ivec0.update(ivec)
        fv=[]
        for k, v in sorted(ivec0.items(), key=itemgetter(0)):
            fv.append(v)
        #print(fv)
        test_batch[n[:-9][-21:]] = fv #features from cor.feat file gets represented in full vector and appended to batch
        #fit feature vector to current test_data size
    testfile.close()
        

# output format:
#{ '2007': { 'ARA': 0.123, 'CHN': 0.321, ... },
# '1935': { 'ARA': 0.111, ... },
# ...
#}

#test_vector = [0]*len(test_data)
#a=[["e", "0"], ["oe", "o0"], ["oep", "o0p"], ["0", "e"], ["p0", "pe"], ["p0o", "peo"], ["p", "0"], ["ep", "e0"], ["epl", "e0l"], ["e", "0"], ["le", "l0"], ["le_", "l0_"], ["0", "p"], ["x0", "xp"], ["x0e", "xpe"], ["o", "i"], ["do", "di"], ["don", "din"], ["0", " "], ["f0", "f "], ["f0c", "f c"], ["t", "0"], ["ht", "h0"], ["hte", "h0e"], ["0", "t"], ["i0", "it"], ["i0h", "ith"], ["0", "s"], ["s0", "ss"], ["s0f", "ssf"], ["0", " "], ["f0", "f "], ["f0e", "f e"], ["r", "0"], ["rr", "r0"], ["rra", "r0a"], ["0", "o"], ["h0", "ho"], ["h0o", "hoo"], ["r", "y"], ["or", "oy"], ["or_", "oy_"]]
#a.sort()

results={}
test_data = np.random.randint(2, size=71)

for k, elem in test_batch.items():
    result = {}
    result = dict(zip(lr_train.classes_.tolist(), lr_train.predict_proba(elem).tolist()[0]))
    #outinfo = lr_train.predict(elem)
    #outinfo_prob = lr_train.predict_proba(elem)
    results[k] = result
    print(k, result)

with open("test_run_spell_errors.results", "wb") as outfile:
    pickle.dump(results, outfile)
outfile.close()
