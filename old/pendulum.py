'''
Create a simulation environment for a N-pendulum.
Example of use:

env = Pendulum(N)
env.reset()

for i in range(1000):
   env.step(zero(env.nu))
   env.render()

'''

import numpy as np
import pinocchio as pin
from display import Display
from numpy.linalg import inv
import time
import matplotlib.pyplot as plt


class Visual:
    '''
    Class representing one 3D mesh of the robot, to be attached to a joint. The class contains:
    * the name of the 3D objects inside Gepetto viewer.
    * the ID of the joint in the kinematic tree to which the body is attached.
    * the placement of the body with respect to the joint frame.
    This class is only used in the list Robot.visuals (see below).
    '''
    def __init__(self, name, jointParent, placement):
        self.name = name                  # Name in gepetto viewer
        self.jointParent = jointParent    # ID (int) of the joint 
        self.placement = placement        # placement of the body wrt joint, i.e. bodyMjoint
    
    def place(self, display, oMjoint):
        oMbody = oMjoint*self.placement
        display.place(self.name,oMbody,False)

class Pendulum:
    '''
    Define a class Pendulum with nbJoint joints.
    The members of the class are:
    * viewer: a display encapsulating a gepetto viewer client to create 3D objects and place them.
    * model: the kinematic tree of the robot.
    * data: the temporary variables to be used by the kinematic algorithms.
    * visuals: the list of all the 'visual' 3D objects to render the robot, each element of the list being
    an object Visual (see above).    
    '''

    def __init__(self, nbJoint=1, noise_stddev=0.0):
        '''Create a Pinocchio model of a N-pendulum, with N the argument <nbJoint>.'''
        self.viewer     = Display()
        self.visuals    = []
        self.model      = pin.Model()
        self.createPendulum(nbJoint)
        self.data       = self.model.createData()
        self.noise_stddev = noise_stddev

        self.q0         = np.zeros(self.model.nq)

        self.DT         = 5e-2   # Time step length
        self.NDT        = 1      # Number of Euler steps per integration (internal)
        self.Kf         = .10    # Friction coefficient
        self.vmax       = 8.0    # Max velocity (clipped if larger)
        self.umax       = 4.0    # Max torque   (clipped if larger)
        self.DQ = 2*np.pi/self.nq   # discretization resolution for joint angle
        self.DV = 2*self.vmax/self.nv # discretization resolution for joint velocity
        self.withSinCos = False  # If true, state is [cos(q),sin(q),qdot], else [q,qdot]

    def createPendulum(self, nbJoint, rootId=0, prefix='', jointPlacement=None):
        color   = [red,green,blue,transparency] = [1,1,0.78,1.0]
        colorred = [1.0,0.0,0.0,1.0]

        jointId = rootId
        jointPlacement     = jointPlacement if jointPlacement!=None else pin.SE3.Identity()
        length = 1.0
        mass = length
        inertia = pin.Inertia(mass,
                              np.array([0.0,0.0,length/2]).T,
                              mass/5*np.diagflat([ 1e-2,length**2,  1e-2 ]) )

        for i in range(nbJoint):
            istr = str(i)
            name      = prefix+"joint"+istr
            jointName = name+"_joint"
            jointId   = self.model.addJoint(jointId, pin.JointModelRY(), jointPlacement, jointName)
            self.model.appendBodyToJoint(jointId, inertia,pin.SE3.Identity())
            try:self.viewer.viewer.gui.addSphere('world/'+prefix+'sphere'+istr, 0.15,colorred)
            except: pass
            self.visuals.append( Visual('world/'+prefix+'sphere'+istr, jointId, pin.SE3.Identity()) )
            try:self.viewer.viewer.gui.addCapsule('world/'+prefix+'arm'+istr, .1,.8*length, color)
            except:pass
            self.visuals.append( Visual('world/'+prefix+'arm'+istr, jointId,
                                        pin.SE3(np.eye(3), np.array([0.,0.,length/2]))))
            jointPlacement     = pin.SE3(np.eye(3), np.array([0.0,0.0,length]).T)

        self.model.addFrame( pin.Frame('tip', jointId, 0, jointPlacement, pin.FrameType.OP_FRAME) )

    def display(self, q):
        ''' Display the robot in the viewer '''
        pin.forwardKinematics(self.model, self.data,q)
        for visual in self.visuals:
            visual.place( self.viewer, self.data.oMi[visual.jointParent] )
        self.viewer.viewer.gui.refresh()


    ''' Size of the q vector '''
    @property 
    def nq(self): return self.model.nq 
    ''' Size of the v vector '''
    @property
    def nv(self): return self.model.nv
    ''' Size of the x vector '''
    @property
    def nx(self): return self.nq+self.nv
#    @property
#    def nobs(self): return self.nx+self.withSinCos
    ''' Size of the u vector '''
    @property
    def nu(self): return self.nv

    def reset(self, x0=None):
        ''' Reset the state of the environment to x0 '''
        if x0 is None: 
            q0 = np.pi*(np.random.rand(self.nq)*2-1)
            v0 = np.random.rand(self.nv)*2-1
            x0 = np.vstack([q0,v0])
        self.x = x0.copy()
        self.r = 0.0
        return self.obs(self.x)
    

    def step(self, u):
        ''' Simulate one time step '''
        _,self.r = self.dynamics(self.x, u)
        return self.obs(self.x), self.r

    def obs(self, x):
        ''' Compute the observation of the state '''
        if self.withSinCos:
            return np.vstack([ np.vstack([np.cos(qi),np.sin(qi)]) for qi in x[:self.nq] ] 
                             + [x[self.nq:]],)
        else: return x.copy()

    def tip(self, q):
        '''Return the altitude of pendulum tip'''
        pin.framesKinematics(self.model, self.data,q)
        return self.data.oMf[1].translation[2,0]

    def dynamics(self, x, u, display=True):
        '''
        Dynamic function: x,u -> xnext=f(x,y).
        Put the result in x (the initial value is destroyed). 
        Also compute the cost of taking this step.
        Return x for convenience along with the cost.
        '''

        modulePi = lambda th: (th+np.pi)%(2*np.pi)-np.pi
        sumsq    = lambda x : np.sum(np.square(x))

        cost = 0.0
        if self.nx == 2:
            q = modulePi(x[:self.nq])
            v = x[self.nq:]
        else:
            q = modulePi(x[0])
            v = x[1]

        if self.nu == 2: 
             u = [u, 0]
       
        u = np.clip(np.reshape(np.array(u),self.nu),-self.umax,self.umax)

        DT = self.DT/self.NDT
        for i in range(self.NDT):
            pin.computeAllTerms(self.model, self.data, q, v)
            M   = self.data.M
            b   = self.data.nle

            a   = np.dot(inv(M),(u-self.Kf*v-b))

            a   = a.reshape(self.nv) + np.random.randn(self.nv)*self.noise_stddev
            self.a = a
            q    += (v+0.5*DT*a)*DT
            v    += a*DT

            q_first = q[0]

            q_goal = 0
    
            delta_q = 0
            
            if len(q)==2:
                delta_q += 0.7*sumsq(q_goal-abs(q_first))
                q_second = q[1]
                #q2_vertical = q_first + q_second
                #delta_q += sumsq(q_goal-abs(q2_vertical))
                delta_q += 0.3*sumsq(q_goal-abs(q_second))
                
                # q2_vertical = q_first + q_second
                # if (q2_vertical > np.pi):
                #     q2_vertical = -(2*np.pi-q2_vertical)

                # delta_q += 0.15*sumsq(q_goal-abs(q2_vertical))

            else:
                delta_q += sumsq(q_goal-abs(q_first))      
            
            #Wq = 0.7
            #Wv = 1-Wq
            #cost += Wq * delta_q + Wv * np.exp(-delta_q/2) * np.sum(v)
                
            cost += delta_q

            if display:
                self.display(q)
                time.sleep(1e-2)

        
        if self.nx == 4:
            x[0] = modulePi(q)
            x[1] = np.clip(v,-self.vmax,self.vmax)

        else:
            x[:self.nq] = modulePi(q)
            x[self.nq:] = np.clip(v,-self.vmax,self.vmax)

        return x, -cost
     
    def render(self):
        q = self.x[:self.nq]
        self.display(q)
        #time.sleep(self.DT/10)


    # Continuous to discrete
    def c2dq(self, q):
        q = (q+np.pi)%(2*np.pi)
        return int(np.floor(q/self.DQ))  % self.nq
    
    def c2dv(self, v):
        v = np.clip(v,-self.vmax+1e-3,self.vmax-1e-3)
        return int(np.floor((v+self.vmax)/self.DV))
    
    # Discrete to continuous
    def d2cq(self, iq):
        iq = np.clip(iq,0,self.nq-1)
        return iq*self.DQ - np.pi + 0.5*self.DQ
    
    def d2cv(self, iv):
        iv = np.clip(iv,0,self.nv-1) - (self.nv-1)/2
        return iv*self.DV



# Example of simulation
if __name__ == "__main__":
    env = Pendulum(2)
    x = env.reset(np.array([0.,0.,0.,0.]))
    # print(x)
    q1 = x[0]
    q2 = x[1]
    q2_vertical = q1 + q2
    if (q2_vertical>np.pi):
         q2_vertical = -(2*np.pi-q2_vertical)
    env.render()

    for i in range(10000):
        x, reward = env.step()
        env.render()
        time.sleep(2)
        
    time.sleep(20)


