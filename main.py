# -*- coding: utf-8 -*-

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np
import random
import csv

train='train_data.csv'
test='test_data.csv'
y_frags='test_frags.csv'

X_train=np.genfromtxt(train, delimiter=",")
y_train=[0]*7000+[1]*10000
X_test=np.genfromtxt(test, delimiter=",")


m=LogisticRegression(solver='liblinear',penalty='l2',C=0.1)
m.fit(X_train, y_train)
predicted=m.predict(X_test)

with open(y_frags, 'r') as y_frag_file:
  y_frag_data=csv.reader(y_frag_file, delimiter=',')
  y_frag_data=[y for y in y_frag_data]

count_pos=0
result_frags=[]

for p in predicted:
  if p==0:
    count_pos+=1
  else:
    result_frags.append(y_frag_data[count_pos])
    count_pos+=1

result=''
for rf in result_frags:
  if rf[1]!=rf[2] and int(rf[1])!=int(rf[2])+1:
    result+=rf[0]+'\t'+rf[1]+'\t'+str(int(rf[2])+1)+'\n'

with open('result.txt', 'w') as outfile:
  outfile.write(result)
