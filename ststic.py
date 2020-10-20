import os
import re
import string
import nltk
from nltk.tokenize import word_tokenize
import tqdm
import json
import sys
import pandas
import numpy

def s_ubuntu(root):
    par = tqdm.tqdm()
    with open(root+'/count.txt','w') as f:
        for dirName, subdirList, fileList in os.walk(root):
            par.update(1)
            for fname in fileList:
                if ".tsv" in fname:
                    file = os.path.join(dirName, fname)
                    tcount = single_ubuntu(file)
                    f.write(str(tcount)+'\n')
    f.close()
    

def single_ubuntu(file):
    with open(file, 'r') as f:
        count = 0
        for line in f:
            count+=1
    return count

root = '/Users/chaofeng/Documents/GitHub/dialog_system/dialogs'
#s_ubuntu(root)

def s_twiter(file):
    par = tqdm.tqdm()
    count = 0
    with open(root+'/t_count.txt','w') as out, open(file, 'r') as ins:
        for line in ins:
            if len(line)==1:
                out.write(str(count)+'\n')
                count = 0
            else:
                count+= 1

file = '/Users/chaofeng/Documents/GitHub/dialog_system/conversations.out'
#s_twiter(file)

with open('/Users/chaofeng/Documents/GitHub/dialog_system/count.txt', 'r') as f:
    data = []
    for line in f:
        a =int(line.rstrip())
        data.append(a)
df = pandas.DataFrame(data)
n = numpy.array(data)
average = n.mean()
print(average)
cov = n.std()
print(cov)
'''import plotly.express as px

fig = px.histogram(df, log_y=True)
fig.show()'''