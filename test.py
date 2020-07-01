import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a)
il1 = np.tril_indices(3, -1)
print(a[il1])
