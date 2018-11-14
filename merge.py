import pickle as pk
from tqdm import tqdm
from IPython import embed

def o(f):
    return pk.load(open(f, 'rb'))

a = 'data1'
b = 'data2'

d1 = o(a)
d2 = o(b)

ks = list(d1.keys())
for k in ks:
    d1[k] =( d2[k] + d1[k])

import numpy as np

output_file = open('merge1.csv', "w")
output_file.write("Filename,ClassId\n")
for f in tqdm(ks):
    if True:

        pred = int(np.argmax(d2[f]))
        #embed()
        file_id = f
        output_file.write("%s,%d\n" % (file_id, pred))

output_file.close()
