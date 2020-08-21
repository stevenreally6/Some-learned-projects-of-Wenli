from numpy import *
import numpy as np
import math
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

a=20
Q=np.mat([[100,0,0],[0,100,0],[0,0,100]])
R=np.mat(np.eye(2))
P=np.mat([[100,0,0],[0,100,0],[0,0,100]]) 
A=np.mat([[0.5,0,0.5],[0,0,-0.5],[0.5,0.5,0.5]])
B=np.mat([[0,0],[1,0],[0,1]])


K=np.mat([[0,0,0],[0,0,0]])

x=np.mat([[10],[10],[10]])
u=np.mat([[0],[0]])
J=np.mat([[0]])
n=0
t=[0]

for n in range(a):    
    if n==0:
        u=K*x
        J=u.T*1*u+x.T*100*x
    else:  
        x_ns=A*x[0:3,n-1]+B*u[0:2,n-1]
        x=np.append(x,x_ns,axis=1)  
        #3x3matr * 3x1matr+3x2matr * 2x1matr =3x1 matr'''
        k_ns=-np.linalg.inv(R+B.T*P[0:3,(n-1)*3:n*3]*B)*B.T*P[0:3,(n-1)*3:n*3]*A
        K=np.append(K,k_ns,axis=1)
        #(2x2matr + 2x3matr * 3x3matr * 3x2matr) * 2x3matr * 3x3matr * 3x3matr = 2x3matr'''
        u_ns=K[0:2,n*3:(n+1)*3]*x[:,n]
        u=np.append(u,u_ns,axis=1)  
        #2x3matr *  3x1matr = 2x1matr'''
        j_ns=x[0:3,n].T*Q*x[0:3,n]+u[0:2,n].T*R*u[0:2,n]+J[0,n-1]
        J=np.append(J,j_ns,axis=1) 
         #1x3matr * 3x3matr * 3x1matr + 1x2matr * 2x2matr * 2x1matr + 1x1matr'''
        p_ns=Q+A.T*P[0:3,(n-1)*3:n*3]*A+A.T*P[0:3,(n-1)*3:n*3]*B*k_ns
        P=np.append(P,p_ns,axis=1)
        #3x3matr + 3x3matr * 3x3matr * 3x3matr + 3x3matr * 3x3matr * 3*2matr * 2x3matr= 3x3matr'''
        t.append(n)

        

print('x',x)
print('u',u)
print('J',J)
print('K',K)
print('P',P)
xf0=[]
xf1=[]
xf2=[]
uf0=[]
uf1=[]
Jf=[]
for j in range(len(t)):
    xf0.append(x[0,j])
    xf1.append(x[1,j])
    xf2.append(x[2,j])
    uf0.append(u[0,j])
    uf1.append(u[1,j])
    Jf.append(J[0,j])

plt.plot(t,xf0, "b-", lw=2, label="X0")
plt.plot(t,xf1, "g-", lw=2, label="X1")
plt.plot(t,xf2, "r-", lw=2, label="X2")
plt.title('Control')
plt.legend()
plt.show()
plt.plot(t,uf0, "b-", lw=2, label="U0")
plt.plot(t,uf1, "r-", lw=2, label="U1")
plt.title('Control')
plt.legend()
plt.show()
plt.plot(Jf, ls="-", lw=2, label="J")
plt.title('Control')
plt.legend()
plt.show()

#%%

a=20
Q=np.mat([[100,0,0],[0,100,0],[0,0,100]])
R=np.mat(np.eye(2))
P=np.mat([[100,0,0],[0,100,0],[0,0,100]]) 
A=np.mat([[0.5,0,0.5],[0,0,-0.5],[0.5,0.5,0.5]])
B=np.mat([[0,0],[0,0],[0,0]])


K=np.mat([[0,0,0],[0,0,0]])

x=np.mat([[10],[10],[10]])
u=np.mat([[0],[0]])
J=np.mat([[0]])
n=0
t=[0]

for n in range(a):    
    if n==0:
        u=K*x
        J=u.T*1*u+x.T*100*x
    else:  
        x_ns=A*x[0:3,n-1]+B*u[0:2,n-1]
        x=np.append(x,x_ns,axis=1)  
        #3x3matr * 3x1matr+3x2matr * 2x1matr =3x1 matr'''
        k_ns=-np.linalg.inv(R+B.T*P[0:3,(n-1)*3:n*3]*B)*B.T*P[0:3,(n-1)*3:n*3]*A
        K=np.append(K,k_ns,axis=1)
        #(2x2matr + 2x3matr * 3x3matr * 3x2matr) * 2x3matr * 3x3matr * 3x3matr = 2x3matr'''
        u_ns=K[0:2,n*3:(n+1)*3]*x[:,n]
        u=np.append(u,u_ns,axis=1)  
        #2x3matr *  3x1matr = 2x1matr'''
        j_ns=x[0:3,n].T*Q*x[0:3,n]+u[0:2,n].T*R*u[0:2,n]+J[0,n-1]
        J=np.append(J,j_ns,axis=1) 
         #1x3matr * 3x3matr * 3x1matr + 1x2matr * 2x2matr * 2x1matr + 1x1matr'''
        p_ns=Q+A.T*P[0:3,(n-1)*3:n*3]*A+A.T*P[0:3,(n-1)*3:n*3]*B*k_ns
        P=np.append(P,p_ns,axis=1)
        #3x3matr + 3x3matr * 3x3matr * 3x3matr + 3x3matr * 3x3matr * 3*2matr * 2x3matr= 3x3matr'''
        t.append(n)

        

print('x',x)
print('u',u)
print('J',J)
print('K',K)
print('P',P)
xf0=[]
xf1=[]
xf2=[]
uf0=[]
uf1=[]
Jf=[]
for j in range(len(t)):
    xf0.append(x[0,j])
    xf1.append(x[1,j])
    xf2.append(x[2,j])
    uf0.append(u[0,j])
    uf1.append(u[1,j])
    Jf.append(J[0,j])
plt.plot(t,xf0, "b-", lw=2, label="X0")
plt.plot(t,xf1, "g-", lw=2, label="X1")
plt.plot(t,xf2, "r-", lw=2, label="X2")
plt.title('Uncontrol')
plt.legend()
plt.show()
plt.plot(t,uf0, "b-", lw=2, label="U0")
plt.plot(t,uf1, "r-", lw=2, label="U1")
plt.title('Uncontrol')
plt.legend()
plt.show()
plt.plot(Jf, ls="-", lw=2, label="J")
plt.title('Uncontrol')
plt.legend()
plt.show()



