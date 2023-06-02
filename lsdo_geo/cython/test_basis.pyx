import numpy as np
import matplotlib.pyplot as plt


cdef int order, ncp, npt
cdef int i, k, l, index

order = 4
ncp = 10
npt = 100
CP = np.zeros((ncp, 2))
CP[:, 0] = np.linspace(0., 1., ncp)
CP[:, 1] = np.linspace(0., 1., ncp)

cdef double knot_vector[14]
cdef double basis[4]

get_open_uniform(order, ncp, knot_vector)

pts = np.zeros((npt, 2))
u_list = np.linspace(0, 1, npt)

for i in range(npt):
    index = get_basis0(order, ncp, u_list[i], knot_vector, basis)

    for k in range(2):
        pts[i, k] = 0
        for l in range(order):
            pts[i, k] += CP[l+index, k] * basis[l]


CP = np.random.random((ncp, 2))

k = 0
u = 0.3
h = 1e-5
pt_ref = 0
pt_plus = 0
pt_minus = 0
deriv1 = 0
deriv2 = 0

index = get_basis0(order, ncp, u, knot_vector, basis)
for l in range(order):
    pt_ref += CP[l+index, k] * basis[l]

index = get_basis0(order, ncp, u+h, knot_vector, basis)
for l in range(order):
    pt_plus += CP[l+index, k] * basis[l]

index = get_basis0(order, ncp, u-h, knot_vector, basis)
for l in range(order):
    pt_minus += CP[l+index, k] * basis[l]

index = get_basis1(order, ncp, u, knot_vector, basis)
for l in range(order):
    deriv1 += CP[l+index, k] * basis[l]

index = get_basis2(order, ncp, u, knot_vector, basis)
for l in range(order):
    deriv2 += CP[l+index, k] * basis[l]

print((pt_plus-pt_minus)/2./h)
print(deriv1)

print((pt_plus-2*pt_ref+pt_minus)/h**2)
print(deriv2)

plt.plot(pts[:, 0], pts[:, 1], 'o-')
plt.show()