import numpy as np
import matplotlib.pyplot as plt
import deep_learning
lista = [] 
listb = []
listc = []
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))
def bin2deci(listA):
        a=0.0
        for i in xrange(len(listA)):
                a += listA[i]*2**i
        return a

#input data
X = np.array([[0,0,1],
            [0,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [0,1,0],
            [0,0,0],
            [0,1,1],
            [1,1,1],
            [0,0,0],
            [0,1,0],
            [0,1,1],
            [1,1,1],
            [0,0,1],
            [1,1,0],
            [0,0,1]
            ])

#output data
y = np.array([[0],
            [1],
            [1],
            [0],
            [1],
            [1],
            [0],
            [0],
            [1],
            [1],
            [0],
            [0],
            [0],
            [0],
            [1],
            [1]])
XA = np.array([[0,0,1],
            [0,1,0],
            [1,1,0],
            [0,1,1],
            [0,1,0],
            [0,1,1],
            [0,1,0],
            [0,1,1],
            [1,1,1],
            [0,1,0],
            [0,1,0],
            [0,1,1],
            [1,1,1],
            [0,0,1],
            [1,1,1],
            [0,1,1]
            ])

#output data
yA = np.array([[0],
            [1],
            [1],
            [1],
            [1],
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [1],
            [0],
            [0]])

np.random.seed(1)

#synapses
syn0 = 2*np.random.random((3,16)) - 1
syn1 = 2*np.random.random((16,1)) - 1
syn0A = 2*np.random.random((3,16)) - 1
syn1A = 2*np.random.random((16,1)) - 1

times = 25

for j in xrange(times*10000):
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    l2_error = y - l2

    l0A = X
    l1A = nonlin(np.dot(l0A,syn0A))
    l2A = nonlin(np.dot(l1A,syn1A))

    l2A_error = yA - l2A

    l2_delta = l2_error*nonlin(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error*nonlin(l1, deriv=True)

    l2A_delta = l2A_error*nonlin(l2A, deriv=True)

    l1A_error = l2A_delta.dot(syn1A.T)

    l1A_delta = l1A_error*nonlin(l1A, deriv=True)
    #update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    syn1A += l1A.T.dot(l2A_delta)
    syn0A += l0A.T.dot(l1A_delta)
    if(j%100) == 0:
            listb.append(float(np.mean(l2)))
            lista.append(float(np.mean(l2A)))
            listc.append(j)
    if(j%10000) == 0:
        plt.plot(listc,listb,color="red",linewidth=1,label="l2")
        plt.plot(listc,lista,color="blue",linewidth=1,label="l2A",linestyle="--")
plt.ylim(0,1)
plt.xlim(0,times*10000)
plt.show()



print "Output after training"

print l2
