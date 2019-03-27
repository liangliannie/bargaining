import numpy as np
import matplotlib.pylab as plt

USD = [0.5621, 0.3770, 0.4878, 0.4583, 0.4346, 0.4652, 0.3212, 0.2144, 0.2117, 0.2341, 0.4166, 0.4922, 0.5672, 0.2321, 0.1692, 0.1933, 0.4363, 0.4087, 0.5987, 0.1841]
USF = USD

A = np.zeros([20, 20])
p = range(20)
for i in range(20):
    if i>0:
        p = [p[-1]] + p[:-1]
    for j in range(20):
        A[i][j] = USD[j]**(p[j])
        # print(i,j,p[j])

A_inv = np.linalg.inv(A)
b = np.ones([20, 1])
print(A_inv.shape, b.shape)

s=(np.dot(A_inv, b))
share=(s*A)
# print(np.multiply(A_inv, b))

fig = plt.figure()
#
ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
#

# print(np.array(p1))
plt.plot(range(20), share, '.-', linewidth=.3, alpha=0.5)
plt.plot(range(20), share.mean(axis=1), '*-', label='Average', linewidth=2, markersize=10)

# plt.plot(range(20), np.array(USD)/np.array(USD).sum(), 'o-', label='USD')
# plt.plot(range(20), share.mean(axis=1), '*-', label='Multilateral Bargaining')
# plt.plot(range(20), np.ones([20,1])*0.05, 'o-', label='Nash Bargaining')

for i in range(20):
    ymax = share[i].max()
    plt.annotate(str(i), xy=(i, ymax), xytext=(i, ymax+.005), arrowprops=dict(facecolor='red', shrink=0.05, width=3, headwidth=3, headlength=3))
# plt.plot(range(20), portion_propose1, 'o-', label='USF ')
# plt.plot(range(20), portion_propose, '--', label='USD')
# # plt.xlabel('# of SU')
# plt.ylabel('Portion(Proposer Exits)')
# plt.legend()
# # plt.suptitle(''
# ax2 = plt.subplot2grid((6, 1), (3, 0), rowspan=3, colspan=1, sharex=ax1)
# plt.plot(range(20), portion_accept1, '.-', label='USF')
# plt.plot(range(20), portion_accept, '*-', label='USD')
# # plt.suptitle('')
plt.xticks(np.arange(20),[str(i+1) for i in range(20)])
plt.xlabel('# of AP')
plt.ylabel('Portion')
plt.legend()
plt.show()