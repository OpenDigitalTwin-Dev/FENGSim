import numpy as np

p1 = np.array([0, 0, 0])
n1 = np.array([1, 2, 3])
p2 = np.array([1, 0, 0])
n2 = np.array([4, 5, 6])
u = n1
dp2p1 = p2 - p1
print("***f4: ")
#print(dp2p1)
dist = np.linalg.norm(p2-p1)
print(dist)
print("***f3: ")
phi = np.inner(n1, (p2-p1)/dist)
print(phi)

print("***f2: ")
_v = np.cross(u, dp2p1/dist)
v = _v / np.linalg.norm(_v)
#print(v)
alpha = np.inner(v, n2)
print(alpha)

w = np.cross(u, v)
print("***f1: ")
theta = np.arctan2(np.inner(w,n2),np.inner(u,n2))
print(theta)
