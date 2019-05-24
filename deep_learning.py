import numpy as np
class deeplearning():
    def __init__(self,X,y,syn0,syn1,times=6,interval_dur=10000):
        #Proclaim the Sigmoid function and Derivation function
        def nonlin(x,deriv=False):
            if(deriv==True):
                return x*(1-x)
            return 1/(1+np.exp(-x))
        self.lista = []
        self.listb = []
        #Traing Steps
        for j in xrange(times*10000):
            #Put input data inside
            l0 = X
            #Training first layer
            l1=nonlin(np.dot(l0,syn0))
            #Training second layer
            l2=nonlin(np.dot(l1,syn1))
            #Get the error rate of trained values
            l2_error=y-l2
            #Backpropagation
            l2_delta=l2_error*nonlin(l2,deriv=True)
            l1_error=l2_delta.dot(syn1.T)
            l1_delta=l1_error*nonlin(l1,deriv=True)
            #Upgrade the weights
            syn1 += l1.T.dot(l2_delta)
            syn0 += l0.T.dot(l1_delta)
            if(j%100) == 0:
                #Updating the data lists(time,trained output)
                self.lista.append(np.mean(j))
                self.listb.append(np.mean(l2))