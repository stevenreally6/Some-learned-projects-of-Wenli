import numpy as np
import matplotlib.pyplot as plt
import matplotlib


    #This class describes the gridworld and provides some helper functions to use for value/policy iteration and q-learning with a table
    #There are 5 options to move in every state (up:0,down:1,left:2,right:3)
 
    #the every state cost is different due to different color of the cell.
    #violet=-1  white=0  green=1  red=10  obstacle=-10


G=np.array([[0,0,1,0,-1],[0,20,1,0,10],[0,10,20,1,0],[1,0,0,10,-1]])  #4x5 gridworld reward
A=np.array([0,1,2,3])


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
    
def value_iteration(threshold,alpha,ite):
    P=np.zeros((4,5)) #policy
    P[1][1]=5 #grey obstable
    P[2][2]=5 #grey obstable
    J=np.zeros((4,5)) #old value
    J[1][1]=200
    J[2][2]=200
    for i in range(ite):
        J_new=J.copy()  #new value
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
                        next_state, reward =next_state_data(state,action,G) #get next state date
                        a_v[action]=reward + alpha*J[x][y] #value of current state
                    J[x][y]=min(a_v)
                    P[x][y]=np.argmin(a_v)
        if np.sum(np.abs(J_new-J)) <= threshold:
            print(i+1)
            break
    return J,P,i+1


best_J, best_P, iteration_time = value_iteration(1e-6,0.99,10000)
print(best_J)
print(iteration_time)
print(best_P)


                
                
                
                
        