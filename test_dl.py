import deep_learning as dl
import numpy as np
import deci2binArr as d2b
import matplotlib.pyplot as plt
X = np.array([[1,0,1],
            ])
y = np.array([[0]
            ])
syn0 = 2*np.random.random((3,1)) - 1
syn1 = 2*np.random.random((1,1)) - 1
times=0
for times in xrange(10):
    plt.plot(dl.deeplearning(X,y,syn0,syn1,times).lista,dl.deeplearning(X,y,syn0,syn1,times).listb)
plt.xlim(0,1000)

plt.show()
