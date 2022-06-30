import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

mixed = np.load('mixed.npy')
fp32 = np.load('fp32.npy')
N = 1000
plt.plot(mixed[:N],label='Mixed Precision')
plt.plot(fp32[:N],label='FP32')
plt.legend()
plt.title('loss values of a VAE trained on single and mixed precisions' )
plt.savefig('vae.png')
plt.show()
