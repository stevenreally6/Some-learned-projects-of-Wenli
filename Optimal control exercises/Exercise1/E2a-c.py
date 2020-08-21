import numpy as np
import matplotlib.pyplot as plt

"First define the state xn and the control un"
xn=[-4,-3,-2,-1,0,1,2,3,4]
un=[-2,-1,0,1,2]

"the dynamic system"
"xn_plus_one=xn+un"
            
"the cost function"   

"stage 15"
J15=[]
X15=[]
for i in range (len(xn)):
    if xn[i]==0:      
        J=0
    else:
        J=float("inf")
    x15=xn[i]
    J15.append(J)      
    X15.append(x15)
print("J15=",J15)
print("X15=",X15)

"stage 14-0"

def DP(XN,JN):
    JN_minus_1=[]
    XN_minus_1=[]
    UN_minus_1=[]
    for i in range (len(xn)):
        for j in range (len(un)):
            if xn[i]+un[j]<=4 and xn[i]+un[j]>=-4:
                xn_1 = xn[i]
                un_1 = un[j]
                a=xn_1+un_1
                num = [h for h in range(len(XN)) if XN[h] == a]
                jn_list = []
                for k in range(len(num)):
                    j_list = JN[num[k]]
                    jn_list.append(j_list)
                jn = min(jn_list)                
                J = (xn_1+4)**2+un_1**2+jn
                JN_minus_1.append(J)      
                XN_minus_1.append(xn_1)       
                UN_minus_1.append(un_1)
    return JN_minus_1, XN_minus_1, UN_minus_1

for i in range (14,-1,-1):
    locals()['J'+str(i)],locals()['X'+str(i)],locals()['U'+str(i)]=DP(locals()['X'+str(i+1)],locals()['J'+str(i+1)])
    print('J'+str(i) , locals()['J'+str(i)])
    print('X'+str(i) , locals()['X'+str(i)])
    print('U'+str(i) , locals()['U'+str(i)])
X=[]
J=[]
U=[]
for i in range (15):
    X.append(locals()['X'+str(i)])
    J.append(locals()['J'+str(i)])
    U.append(locals()['U'+str(i)])
X=np.array(X)
J=np.array(J)
U=np.array(U)



"Search for the optimal control /stage sequence"
num = J0.index(min(J0))
x0 = X0[num]
u0 = U0[num]
j0 = J0[num]
X_opt=[x0]
U_opt=[u0]
J_opt=[j0]
for l in range (1,15):
    locals()['x'+str(l)] = locals()['x'+str(l-1)] + locals()['u'+str(l-1)]
    xn = locals()['x'+str(l)]
    num = [m for m in range(len(X[l])) if X[l,m] == xn]
    jn_list = []
    for k in range(len(num)):
        j = J[l,num[k]]
        jn_list.append(j)
    jn = min(jn_list)             
    num2 = jn_list.index(jn)
    un= locals()['U'+str(l)][num[num2]]
    locals()['u'+str(l)] = un
    X_opt.append(xn)
    U_opt.append(un)
    J_opt.append(jn)
print("J_opt=", J_opt)
print("X_opt=", X_opt)
print("U_opt=", U_opt)

def opt (initial_state,J,U):
    x0 = initial_state
    num = [p for p in range(len(X0)) if X0[p] == initial_state]
    j0_list = []
    for k in range(len(num)):
        j = J[0,num[k]]
        j0_list.append(j)
    j0 = min(j0_list)                 
    num2 = J0.index(j0)
    u0= U0[num2]   
    X_opt=[x0]
    U_opt=[u0]
    J_opt=[j0]
    for l in range (1,15):
        locals()['x'+str(l)] = locals()['x'+str(l-1)] + locals()['u'+str(l-1)]
        xn = locals()['x'+str(l)]
        num = [m for m in range(len(X[l])) if X[l,m] == xn]
        jn_list = []
        for k in range(len(num)):
            j = J[l,num[k]]
            jn_list.append(j)
        jn = min(jn_list)  
        J_list=J[l].tolist()
        num2 = jn_list.index(jn)
        U_list=U[l].tolist()
        un= U_list[num[num2]]
        locals()['u'+str(l)] = un
        X_opt.append(xn)
        U_opt.append(un)
        J_opt.append(jn)
    return X_opt, U_opt, J_opt
X_opt_0,U_opt_0,J_opt_0 = opt(0,J,U)
print("J_opt_0=", J_opt_0)
print("X_opt_0=", X_opt_0)
print("U_opt_0=", U_opt_0)

X_opt_1,U_opt_1,J_opt_1 = opt(1,J,U)
print("J_opt_1=", J_opt_1)
print("X_opt_1=", X_opt_1)
print("U_opt_1=", U_opt_1)

X_opt_4,U_opt_4,J_opt_4 = opt(-4,J,U)
print("J_opt_-4: =", J_opt_4)
print("X_opt_-4：=", X_opt_4)
print("U_opt_-4：=", U_opt_4)
axe=[]
for m in range (15) :
    x='stage' + str(m)
    axe.append(x)
axe=np.array(axe)
J_opt_0=np.array(J_opt_0)
J_opt_1=np.array(J_opt_1)
J_opt_4=np.array(J_opt_4)
fig1, ax1 = plt.subplots(figsize=(15,10))
ax1.plot(axe, J_opt_0, 'r--', marker='o', label='x0=0')
ax1.plot(axe, J_opt_1, 'b:', marker='^', label='x0=1')
ax1.plot(axe, J_opt_4, 'g-.',marker='*', label='x0=-4')
legend = ax1.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.title('Optimal Cost')
plt.savefig("./optimal cost_qe")
plt.show()

fig2, ax2 = plt.subplots(figsize=(15,10))
ax2.plot(axe, U_opt_0, 'r--',marker='o', label='x0=0')
ax2.plot(axe, U_opt_1, 'b:',marker='^', label='x0=1')
ax2.plot(axe, U_opt_4, 'g-.',marker='*', label='x0=-4')
legend = ax2.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.title('Optimal Control')
plt.savefig("./optimal control")
plt.show()

fig3, ax3 = plt.subplots(figsize=(15,10))
ax3.plot(axe, X_opt_0, 'r--',marker='o' ,label='x0=0')
ax3.plot(axe, X_opt_1, 'b:',marker='^', label='x0=1')
ax3.plot(axe, X_opt_4, 'g-.',marker='*', label='x0=-4')
legend = ax3.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.title('State Sequence')
plt.savefig("./state_sequence")
plt.show()

