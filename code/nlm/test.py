import numpy as np
from numpy.random import random

a = []

for i in range(3):
    a.append(random(2))

print np.shape(a)

print np.shape(random(100))
