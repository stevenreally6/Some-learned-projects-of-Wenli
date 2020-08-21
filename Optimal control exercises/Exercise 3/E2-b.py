import numpy as np
import matplotlib.pyplot as plt
import matplotlib

G=np.array([[0,0,1,0,-1],[0,20,1,0,10],[0,10,20,1,0],[1,0,0,10,-1]])  #4x5 gridworld
A=np.array([0,1,2,3]) #action


def next_state_data(n,a,G):  #n represent state(x,y)in G, a is action
    if a==0:#up
        if (n[0]==0) and (n[1] in np.arange(0,5,1)):
            return(n,G[n[0]][n[1]])
        else:
            return([n[0]-1,n[1]],G[n[0]-1][n[1]])
    if a==1:#down
        if (n[0]==3) and (n[1] in np.arange(0,5,1)):
            return(n,G[n[0]][n[1]])
        else:
            return([n[0]+1,n[1]],G[n[0]+1][n[1]])
    if a==2:#left
        if (n[1]==0) and (n[0] in np.arange(0,4,1)):
            return(n,G[n[0]][n[1]])
        else:
            return([n[0],n[1]-1],G[n[0]][n[1]-1])
    if a==3:#right
        if (n[1]==4) and (n[0] in np.arange(0,4,1)):
            return(n,G[n[0]][n[1]])
        else:
            return([n[0],n[1]+1],G[n[0]][n[1]+1])
            
    
def compute_value_function(P,alpha,threshold=1e-6):
    J=np.zeros((4,5)) # value table
    J[1][1]=200
    J[2][2]=200
    for i in range(10000):
        J_old=J.copy()  #new value
        for x in range(4):
            for y in range(5):
                if x==1 and y==1: #grey obstable
                    J[x][y]=200
                elif x==2 and y==2: #grey obstable
                    J[x][y]=200
                else:
                    action=P[x][y]
                    state=[x,y]
                    next_state, reward = next_state_data(state,action,G)
                    J[x][y]=reward + alpha*J_old[x][y]   #value of current state      
        if np.sum(np.abs(J_old-J)) <= threshold:
            print(i+1)
            break
    return J

def next_best_policy(J,P,alpha):
    P1=P.copy()
    for x in range(4):
        for y in range(5):
            if x==1 and y==1: #grey obstable
                J[x][y]=200
            elif x==2 and y==2: #grey obstable
                J[x][y]=200
            else:
                a_v=np.zeros(4) #action value memory
                state=[x,y]
                for action in range(4):
                    next_state, reward = next_state_data(state,action,G)
                    a_v[action]=reward + alpha*J[x][y]   #value of current state
                P[x][y]=np.argmin(a_v)
    return P, P1


def policy_iteration(P_old,alpha,ite):
    for i in range(ite):
        J_new=compute_value_function(P_old,alpha)
        P_new, P1=next_best_policy(J_new,P_old,alpha)
        print(P_new)
        if np.all(P1==P_new):
            break
        P_old=P_new.copy()
        
    return P_new, i+1

P_c=np.zeros((4,5)) #random initial policy
P_c[1][1]=5 #grey obstable
P_c[2][2]=5 #grey obstable

best_policy, iteration_time=policy_iteration(P_c,0.99,10000)
print(best_policy)
print(iteration_time)


    
    
    
    
    
    
    