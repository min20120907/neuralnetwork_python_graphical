import deci2binArr as d2b
import numpy as np
a = d2b.deci2bin(2)
c = a.tolist()
c.extend([[0]]*(8-len(c)))
print np.asarray(c)