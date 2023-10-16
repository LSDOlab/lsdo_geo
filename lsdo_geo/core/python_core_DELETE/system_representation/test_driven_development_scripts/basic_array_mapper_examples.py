import numpy as np
import array_mapper as am


input = np.arange(3)
map = np.arange(9).reshape((3,3))
a = am.dot(map, input)
print(a)
print(type(a))
