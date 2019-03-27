import matplotlib.pylab as plt
import numpy as np

USF = [0.8243, 0.5039, 0.9256, 0.6966, 0.6193, 0.6304, 0.3925, 0.2287, 0.1733, 0.2682, 0.6083, 0.6843, 0.4345, 0.2143, 0.1385, 0.2866, 0.5727, 0.5974, 0.9475, 0.2682]
# USF = np.ones([20])*0.9
a_propose = []
a_accept = []
for i in range(len(USF)-1):
    delta1 = USF[i]
    delta2 = USF[i+1]
    a1 =  (1 - delta2) / (1 - delta1 * delta2)
    delta1 = USF[0]
    delta2 = USF[i + 1]
    a2 =  (1 - delta2) / (1 - delta1 * delta2)
    a_propose.append(a1)
    a_accept.append(a2)


A_propose = np.zeros([19, 19])
A_accept = np.zeros([19, 19])
for i in range(18):
    A_propose[i][i] = 1 - a_propose[i]
    A_propose[i][i+1] = - a_propose[i+1]

    A_accept[i][i] = a_accept[i]
    A_accept[i][i+1] = - a_accept[i+1]

last_line_propose = [1]
last_line_accept = [1]

for i in range(18):
    last_line_propose.append(1-a_propose[i+1])
    last_line_accept.append(1-a_accept[i+1])

A_propose[-1] = np.array(last_line_propose)
A_accept[-1]=np.array(last_line_accept)

A_propose_inv, A_accept_inv = np.linalg.inv(A_propose), np.linalg.inv(A_accept)

X_P, X_A = (A_propose_inv[:,-1], A_accept_inv[:,-1])

portion_propose1 = [X_P[0]*(a_propose[0])]
portion_accept1 =[X_A[0]*(a_propose[0])]
for i in range(19):
    portion_propose1.append(X_P[i]*(1-a_propose[i]))
    portion_accept1.append(X_A[i]*(1-a_accept[i]))



USD = [0.5621, 0.3770, 0.4878, 0.4583, 0.4346, 0.4652, 0.3212, 0.2144, 0.2117, 0.2341, 0.4166, 0.4922, 0.5672, 0.2321, 0.1692, 0.1933, 0.4363, 0.4087, 0.5987, 0.1841]
USF = USD
a_propose = []
a_accept = []
for i in range(len(USF)-1):
    delta1 = USF[i]
    delta2 = USF[i+1]
    a1 =  (1 - delta2) / (1 - delta1 * delta2)
    delta1 = USF[0]
    delta2 = USF[i + 1]
    a2 =  (1 - delta2) / (1 - delta1 * delta2)
    a_propose.append(a1)
    a_accept.append(a2)


A_propose = np.zeros([19, 19])
A_accept = np.zeros([19, 19])
for i in range(18):
    A_propose[i][i] = 1 - a_propose[i]
    A_propose[i][i+1] = - a_propose[i+1]

    A_accept[i][i] = a_accept[i]
    A_accept[i][i+1] = - a_accept[i+1]

last_line_propose = [1]
last_line_accept = [1]

for i in range(18):
    last_line_propose.append(1-a_propose[i+1])
    last_line_accept.append(1-a_accept[i+1])

A_propose[-1] = np.array(last_line_propose)
A_accept[-1]=np.array(last_line_accept)

A_propose_inv, A_accept_inv = np.linalg.inv(A_propose), np.linalg.inv(A_accept)

X_P, X_A = (A_propose_inv[:,-1], A_accept_inv[:,-1])

portion_propose = [X_P[0]*(a_propose[0])]
portion_accept =[X_A[0]*(a_propose[0])]
for i in range(19):
    portion_propose.append(X_P[i]*(1-a_propose[i]))
    portion_accept.append(X_A[i]*(1-a_accept[i]))

print(portion_propose)
print(portion_accept)

fig = plt.figure()

ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=3, colspan=1)

plt.plot(range(20), portion_propose1, 'o-', label='USF ')
plt.plot(range(20), portion_propose, '--', label='USD')
# plt.xlabel('# of SU')
plt.ylabel('Portion(Proposer Exits)')
plt.legend()
# plt.suptitle(''
ax2 = plt.subplot2grid((6, 1), (3, 0), rowspan=3, colspan=1, sharex=ax1)
plt.plot(range(20), portion_accept1, '.-', label='USF')
plt.plot(range(20), portion_accept, '*-', label='USD')
# plt.suptitle('')
plt.xticks(np.arange(20),[str(i+1) for i in range(20)])
plt.xlabel('# of SU')
plt.ylabel('Portion(Accepter Exits)')
plt.legend()
plt.show()

