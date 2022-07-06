import numpy as np
import matplotlib
import matplotlib.pyplot as plt

O0 = np.load('O0.npy')
O1 = np.load('O1.npy')
O2 = np.load('O2.npy')
N = 1000
#plt.plot(O0[:N],label="obt_lvl='O0'")
#plt.plot(O1[:N],label="obt_lvl='O1'")
#plt.plot(O2[:N],label="obt_lvl='O2'")
plt.plot(O0[:N],label="obt_lvl='O0'")
plt.plot(O1[:N],label="obt_lvl='O1'")
plt.plot(O2[:N],label="obt_lvl='O2'")

plt.legend()
plt.ylim(0, 0.6)
plt.title('loss values of a CAE trained on single and mixed precisions' )
plt.savefig('cae.png')
#plt.show()
