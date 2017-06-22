from operator import itemgetter
import sys
import difflib
import json
from collections import Counter

feat = {}
result_feat = {}

labelfile = "/Users/gintare/Documents/SU2017/labels.dev.csv"

# loop through all .cor files for instance python3 extracting_spellfeatures.py *.cor
for n in sys.argv[2:]:
    #print("processing file: ", n)
    features = []
    inputfile = open(n, "r")
    for line in inputfile:
        if line.startswith("&"):
            elem = line.split(":")
            #take 1st correction only
            #print (elem[0].split()[1], elem[1].split(",")[0].strip())
            err = elem[0].split()[1]
            cor = elem[1].split(",")[0].strip()
            diff= [i for i in difflib.ndiff(err,cor)]
            #print(diff)
            #4-grams for classical 1 replacement
            #3-grams for classical 1 deletions or insertion
            #6-grams for 2 char replacement?
            
            #adding padding for word boundaries
            diff.insert(0, "  _")
            diff.append("  _")
            diff.append("  _")

            ins = [] #insertions
            det = [] #deletions
            rep = [] #replacements
            feature = [] #all together
            
            D3={}
            for i in range(len(diff)-3): # 3-grams
                try:
                    D3[tuple(diff[i:i+3])] += 1
                except:
                    D3[tuple(diff[i:i+3])] = 1
            
            diff.append("  _")

            D4={}
            for i in range(len(diff)-4): # 4-grams
                try:
                    D4[tuple(diff[i:i+4])] += 1
                except:
                    D4[tuple(diff[i:i+4])] = 1



            #insertion 1
            for e in D3.keys():
                if "+" in e[1] and not (("+" in e[0] or "-" in e[0]) or ("+" in e[2] or "-" in e[2])):
                    ins.append(tuple([str(0),e[1][-1:]]))
                    ins.append(tuple([e[0][-1:]+str(0),e[0][-1:]+e[1][-1:]]))
                    ins.append(tuple([e[0][-1:]+str(0)+e[2][-1:],e[0][-1:]+e[1][-1:]+e[2][-1:]]))
                    #print("insert: ", ins)
            #deletion 1
            for e in D3.keys():
                if "-" in e[1] and not (("+" in e[0] or "-" in e[0]) or ("+" in e[2] or "-" in e[2])):
                    det.append(tuple([e[1][-1:],str(0)]))
                    det.append(tuple([e[0][-1:]+e[1][-1:],e[0][-1:]+str(0)]))
                    det.append(tuple([e[0][-1:]+e[1][-1:]+e[2][-1:],e[0][-1:]+str(0)+e[2][-1:]]))
                    #print("delete: ", det)

            #replacment 1
            for e in D4.keys():
                if "-" in e[1] and "+" in e[2] and not (("+" in e[0] or "-" in e[0]) or ("+" in e[3] or "-" in e[3])):
                    rep.append(tuple([e[1][-1:],e[2][-1:]]))
                    rep.append(tuple([e[0][-1:]+e[1][-1:],e[0][-1:]+e[2][-1:]]))
                    rep.append(tuple([e[0][-1:]+e[1][-1:]+e[3][-1:],e[0][-1:]+e[2][-1:]+e[3][-1:]]))
                    #print("replace: ", rep)

            #replacment 1
            for e in D4.keys():
                if "+" in e[1] and "-" in e[2] and not (("+" in e[0] or "-" in e[0]) or ("+" in e[3] or "-" in e[3])):
                    rep.append(tuple([e[1][-1:],e[2][-1:]]))
                    rep.append(tuple([e[0][-1:]+e[1][-1:],e[0][-1:]+e[2][-1:]]))
                    rep.append(tuple([e[0][-1:]+e[1][-1:]+e[3][-1:],e[0][-1:]+e[2][-1:]+e[3][-1:]]))
                    #print("replace2: ", rep)

            ##features= [i for i in diff if ("+" in i or "-" in i)]
            feature=det+rep+ins
            if feature:
                features+=feature
            #print(feature)
            with open(n + ".feat", "w") as outfile:
                json.dump(features, outfile)
    #print(features)
    feat[n]=features
    inputfile.close()
    outfile.close()

row = Counter([item for sublist in feat.values() for item in sublist])
row0 = dict.fromkeys(row,0)

for k, li in feat.items():
    small_row = Counter(li) # make a dict from list of features, count frequency
    small_row.update(row0)  # fit it to the actual size of row, copy small_row values
    result_feat[k[:-4]]=small_row
print(len(feat), len(row))

Y=[] # documents
X=[]  # feature matrix
feature_names=[]
doc_labels=[]
Dlabel = {}

for k,v in result_feat.items():
    #print(sorted(v.items(), key=itemgetter(0)))
    Y.append(k)
    X_small = []
    feature_names=[]
    for t1, t2 in sorted(v.items(), key=itemgetter(0)):
        X_small.append(t2)
        feature_names.append(t1)
    X.append(X_small)

print(len(Y))
print(len(X))
print(len(feature_names))

with open(sys.argv[1], "r") as lf:
    for line in lf.readlines():
        el = line.split(",")
        if el[0] in Y:
            Dlabel[el[0]]=el[3].rstrip()
lf.close()

for d in Y:
    doc_labels.append(Dlabel[d])

print(len(doc_labels))


with open("matrix_X.file", "w") as outfile:
    json.dump(X, outfile)
outfile.close()

with open("document_Y.file", "w") as outfile:
    json.dump(Y, outfile)
outfile.close()

with open("document_label_Y.file", "w") as outfile:
    json.dump(doc_labels, outfile)
outfile.close()

with open("feature_names.file", "w") as outfile:
    json.dump(feature_names, outfile)
outfile.close()

#sum([i[0] != ' '  for i in difflib.ndiff(a, b)]) / 2
#[i for i in difflib.ndiff(a,b)]
