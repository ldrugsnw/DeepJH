import weakref
import numpy as np

a = np.array([1,2,3])
b = a

print(b)

a = None
print(b)

print('--------------')

a = np.array([1,2,3])
b = weakref.ref(a)

a = None
print(b)