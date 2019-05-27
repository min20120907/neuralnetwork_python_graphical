import deep_learning as dl
import numpy as np
import deci2binArr as d2b
import matplotlib.pyplot as plt
X = np.array([[1,0,1],
              [0,0,1],
              [0,1,1],
              [1,1,1]
            ])
y = np.array([[0.5],
              [0.4],
              [0.3],
              [0.2]
            ])
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
times=5

plt.plot(dl.deeplearning(X,y,syn0,syn1,times).lista,dl.deeplearning(X,y,syn0,syn1,times).listb,label="l2_a")
plt.plot(dl.deeplearning(X,y,syn0,syn1,times).lista,dl.deeplearning(X,y,syn0,syn1,times).listc,label="l2_b")
plt.plot(dl.deeplearning(X,y,syn0,syn1,times).lista,dl.deeplearning(X,y,syn0,syn1,times).listd,label="l2_c")
plt.plot(dl.deeplearning(X,y,syn0,syn1,times).lista,dl.deeplearning(X,y,syn0,syn1,times).liste,label="l2_d")

plt.legend()

plt.show()
