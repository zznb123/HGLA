import numpy as np

f1 = open('dtifold1.csv','rb')
dtifold_t = np.loadtxt(f1, delimiter=' ',skiprows=1)
f1.close()
dtifold_m = np.array(dtifold_t)
