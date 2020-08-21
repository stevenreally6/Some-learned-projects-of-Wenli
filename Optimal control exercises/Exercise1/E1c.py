import numpy as np
import matplotlib.pyplot as plt
x=[-2,-1,0,1,2]
u=[-1,0,1]
J=[]
UN=[]
g=[]
N=3
v=[]

while(N>=0):
    if N==3:
        for i in range(5):
            a=x[i]**2
            J.append(a)
        UN=['none']
        
    else:
        for i in range(5):
            for j in range(3):
                if (-x[i]+1+u[j])>=-2 and (-x[i]+1+u[j])<=2:
                    L=-x[i]+1+u[j]                    
                elif (-x[i]+1+u[j])>2:
                    L=2                    
                else:
                    L=-2
                          #w=1
                if (-x[i]+u[j])>=-2 and (-x[i]+u[j])<=2:
                    O=-x[i]+u[j]
                elif (-x[i]+u[j])>2:
                    O=2
                else:
                    O=-2
                          #w=0
                kL=x.index(L)
                kO=x.index(O)
                pL=J[kL+(2-N)*5]
                pO=J[kO+(2-N)*5]
                h=(2*abs(x[i])+abs(u[j])+pL)*0.7+(2*abs(x[i])+abs(u[j])+pO)*0.3
                g.append(h)
                    
            J.append(min(g))
            c=u[g.index(min(g))]
            UN.append(u[c])
            g=[]
        
    v=J[(3-N)*5:(4-N)*5]
    print('X'+str(N),x)
    print('J'+str(N),v)
    print('U'+str(N),UN)
    UN=[]
    v=[]
    N=N-1
