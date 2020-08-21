# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:32:36 2020

@author: 15945
"""

"Stabilizing the Cart-Pole system"
##In this exercise, we will use LQR to stabilize a cart-pole system 
##and then adapt the LQR controller to get the robot to move along a specified path.

##The difficulty of the cart-pole system is that we can only move the cart back and forth (using u) to move both the cart and the pendulum. 
##Therefore, it is not trivial to find a good controller to get the pendulum to stay balanced on top of the cart while moving the cart around. 
##We will see how we can use our optimal control approach to do such things.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

# note in order to use the animation package to display movies you will need to install ffmpeg
# on Mac with homebrew: brew install ffmpeg
# on ubuntu: apt-get install ffmpeg
# on windows: https://ffmpeg.org/download.html#build-windows
import matplotlib.animation as animation

import IPython

np.set_printoptions(precision=5,linewidth=120,suppress=True)
class CartPole:
    """
    This class describes a cart pole model and provides some helper functions
    """
    
    def __init__(self):
        """
        constructor of the class, takes as input desired discretization number
        for x (angle), v (angular velocity) and u (control) and the maximum control
        """
        #store discretization information
        self.road_length = 3.
        
        #gravity constant
        self.g=9.81

        #integration step
        self.dt = 0.01
        
        #we define lengths and masses
        self.l = 1.0
        self.mc = 5.0
        self.mp = 1.0
            
    def next_state(self,z,u):
        """
        Inputs:
        z: state of the cart pole syste as a numpy array (x,theta,v,omega)
        u: control as a scalar number
        
        Output:
        the new state of the pendulum as a numpy array
        """
        x = z[0]
        th = z[1]
        v = z[2]
        om = z[3]
        x_next = (x + self.dt * v)
        th_next = (th + self.dt * om)
        v_next = v + self.dt*((u + self.mp*np.sin(th)*(self.l*om**2 + self.g * np.cos(th)))/(self.mc+self.mp*np.sin(th)**2))
        w_next = om + self.dt*((-u*np.cos(th)-self.mp*self.l*(om**2)*np.cos(th)*np.sin(th)-(self.mc+self.mp)*self.g*np.sin(th))/(self.mc+self.mp*np.sin(th)**2)/self.l)
        z = np.array([x_next,th_next,v_next,w_next])
        return z
    
    def simulate(self, z0, K, k, controller, horizon_length):
        """
        This function simulates the pendulum of horizon_length steps from initial state x0
        
        Inputs:
        z0: the initial conditions of the pendulum as a numpy array (x,theta,v,omega)
        controller: a function that takes a state z as argument and index i of the time step and returns a control u
        horizon_length: the horizon length
        
        Output:
        z[4xtime_horizon+1] and u[1,time_horizon] containing the time evolution of states and control
        """
        z=np.empty([4, horizon_length+1])
        z[:,0] = z0
        u=np.empty([1,horizon_length])
        for i in range(horizon_length):
            u[:,i] = controller(z[:,i],i,K[i],k[i])
            z[:,i+1] = self.next_state(z[:,i], u[:,i])
        return z, u        
#%%
def animate_cart_pole(x, dt):
    """
    This function makes an animation showing the behavior of the cart-pole
    takes as input the result of a simulation (with dt=0.01s)
    """
    
    min_dt = 0.1
    if(dt < min_dt):
        steps = int(min_dt/dt)
        use_dt = int(min_dt * 1000)
    else:
        steps = 1
        use_dt = int(dt * 1000)
    
    #what we need to plot
    plotx = x[:,::steps]
    
    fig = mp.figure.Figure(figsize=[8.5,2.4])
    mp.backends.backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=[-4.25,4.25], ylim=[-1.,1.4])
    ax.grid()
    
    list_of_lines = []
    
    #create the cart pole
    line, = ax.plot([], [], 'k', lw=2)
    list_of_lines.append(line)
    line, = ax.plot([], [], 'k', lw=2)
    list_of_lines.append(line)
    line, = ax.plot([], [], 'k', lw=2)
    list_of_lines.append(line)
    line, = ax.plot([], [], 'k', lw=2)
    list_of_lines.append(line)
    line, = ax.plot([], [], 'k', lw=2)
    list_of_lines.append(line)
    
    cart_length = 0.5
    cart_height = 0.25
    
    def animate(i):
        for l in list_of_lines: #reset all lines
            l.set_data([],[])
        
        x_back = plotx[0,i] - cart_length
        x_front = plotx[0,i] + cart_length
        y_up = cart_height
        y_down = 0.
        x_pend = plotx[0,i] + np.sin(plotx[1,i])
        y_pend = cart_height - np.cos(plotx[1,i])
        
        list_of_lines[0].set_data([x_back, x_front], [y_down, y_down])
        list_of_lines[1].set_data([x_front, x_front], [y_down, y_up])
        list_of_lines[2].set_data([x_back, x_front], [y_up, y_up])
        list_of_lines[3].set_data([x_back, x_back], [y_down, y_up])
        list_of_lines[4].set_data([plotx[0,i], x_pend], [cart_height, y_pend])
        
        return list_of_lines
    
    def init():
        return animate(0)


    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(plotx[0,:])),
        interval=use_dt, blit=True, init_func=init)
    plt.close(fig)
    plt.close(ani._fig)
    ##IPython.display.display_html(IPython.core.display.HTML(ani.to_html5_video()))
#%%
# we show an example on how to simulate the cartpole and display its behavior

# we create a cart pole
cart = CartPole()

#%%
def check_controllability(A,B):
    """
    This function check  the controllabilitystate for system
    c=[B AB A^2B A^3B]
    """
    c=np.concatenate([B, np.dot(A, B), np.dot(A, A).dot(B),np.dot(A, A.dot(A)).dot(B)], axis=1)
    R=np.linalg.matrix_rank(c)
    print('rank is',R)
    if R< np.linalg.matrix_rank(A):
        print('is not controllable')
    else:print('is controllable')
def solve_ricatti_equations(A,B,Q,R,horizon_length):

    P = [] #will contain the list of Ps from N to 0
    K = [] #will contain the list of Ks from N-1 to 0
    k = []
    p = []
    P.append(Q) #PN
    p.append(-1.0*Q.dot(np.array([0.,np.pi,0.,0.])))
    
    for i in range(horizon_length):
        Knew = -1.0 * np.linalg.inv(B.transpose().dot(P[i]).dot(B) + R).dot(B.transpose()).dot(P[i]).dot(A)
        Pnew = Q + A.transpose().dot(P[i]).dot(A) + A.transpose().dot(P[i]).dot(B).dot(Knew)
        knew = -1.0 * np.linalg.inv(B.transpose().dot(P[i]).dot(B) + R).dot(B.transpose()).dot(p[i])
        pnew = -1.0 * Q.dot(np.array([0.,np.pi,0.,0.])) + A.transpose().dot(p[i]) + A.transpose().dot(P[i]).dot(B).dot(knew)
        K.append(Knew)
        P.append(Pnew)
        k.append(knew)
        p.append(pnew)
    # since we went backward we return reverted lists
    return P[::-1],K[::-1],p[::-1],k[::-1]
def controller(z,i,K,k):
    u = K.dot(z)- K.dot(np.array([0.,np.pi,0.,0.]))

    return u
#%%
z0 = np.array([0.5,np.pi+0.3, 0., 0.])
horizon_length = 1000
A=np.array([[1.,0.,cart.dt,0.],[0.,1.,0.,cart.dt],[0.,cart.dt*cart.mp*cart.g/cart.mc,1.,0],[0.,cart.dt*(cart.mc+cart.mp)*cart.g/(cart.mc*cart.l),0.,1.]])
B=np.array([[0.],[0.],[cart.dt/cart.mc],[cart.dt/cart.mc*cart.l]])
check_controllability(A,B)
Q=np.eye(4)
R=0.001*np.eye(1)
P,K,p,k=solve_ricatti_equations(A, B, Q, R, horizon_length)
z,u = cart.simulate(z0, K, k, controller, horizon_length)
t = np.linspace(0,cart.dt*(horizon_length),horizon_length+1)

J=[]
for i in range(horizon_length):
    if i==0:
        Jnew=z[:,i].transpose().dot(Q).dot(z[:,i])+u[:,i].transpose().dot(R).dot(u[:,i])
    else:
        Jnew=z[:,i].transpose().dot(Q).dot(z[:,i])+u[:,i].transpose().dot(R).dot(u[:,i])+J[i-1]
    J.append(Jnew)
print(J)

plt.plot(J, "r-", lw=2, label="X2")
plt.title('Gains')
plt.legend()
plt.show()

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,z[0,:])
plt.ylabel('Cart position')
plt.subplot(2,1,2)
plt.plot(t,z[1,:])
plt.ylabel('pendulum position')
plt.xlabel('Time')
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,z[2,:])
plt.ylabel('Velocity')
plt.subplot(2,1,2)
plt.plot(t,z[3,:])
plt.ylabel('pendulum angula')
plt.xlabel('Time')
animate_cart_pole(z, cart.dt)
animate_cart_pole(z, cart.dt)
plt.figure()
t_u = np.linspace(cart.dt,cart.dt*(horizon_length),horizon_length)
plt.plot(t_u,u.T)
plt.ylabel('Control')
plt.xlabel('Time')
#%%
z0 = np.array([0.5,0.3, 0., 0.])
horizon_length = 1000
A=np.array([[1.,0.,cart.dt,0.],[0.,1.,0.,cart.dt],[0.,cart.dt*cart.mp*cart.g/cart.mc,1.,0],[0.,cart.dt*(cart.mc+cart.mp)*cart.g/(cart.mc*cart.l),0.,1.]])
B=np.array([[0.],[0.],[cart.dt/cart.mc],[cart.dt/cart.mc*cart.l]])
check_controllability(A,B)
Q=np.eye(4)
R=0.001*np.eye(1)
P,K,p,k=solve_ricatti_equations(A, B, Q, R, horizon_length)
z,u = cart.simulate(z0, K, k, controller, horizon_length)
t = np.linspace(0,cart.dt*(horizon_length),horizon_length+1)
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,z[0,:])
plt.ylabel('Cart position')
plt.subplot(2,1,2)
plt.plot(t,z[1,:])
plt.ylabel('pendulum position')
plt.xlabel('Time')
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,z[2,:])
plt.ylabel('Velocity')
plt.subplot(2,1,2)
plt.plot(t,z[3,:])
plt.ylabel('pendulum angula')
plt.xlabel('Time')
animate_cart_pole(z, cart.dt)
animate_cart_pole(z, cart.dt)
plt.figure()
t_u = np.linspace(cart.dt,cart.dt*(horizon_length),horizon_length)
plt.plot(t_u,u.T)
plt.ylabel('Control')
plt.xlabel('Time')

