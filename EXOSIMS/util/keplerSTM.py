import numpy as np
pi = np.pi

'''
Kepler State Transition Matrix

Class container for defining a planetary system (or group of planets in multiple
systems) via their gravitational parameters and state vectors.  Contains methods
for propagating state vectors forward in time via the Kepler state transition
matrix.

Constructor take the following arguments:
    x0 (ndarray):
        6n vector of stacked positions and velocities for n planets
    mu (ndarray):
        n vector of standard gravitational parameters mu = G(m+m_s) where m is 
        the planet mass, m_s is the star mass and G is the gravitational 
        constant
    epsmult (float):
        default multiplier on floating point precision, used as convergence 
        metric.  Higher values mean faster convergence, but sacrifice precision.

Step function (updateState) takes the following arguments:
    dt (float):
        time step

All units must be complementary (i.e., if position is AU and velocity
is AU/day, mu must be in AU^3/day^2.

Algorithm from Shepperd, 1984, using Goodyear's universal variables
and continued fraction to solve the Kepler equation.

'''
class planSys:    
    def __init__(self, x0, mu, epsmult = 4.0):
        #determine number of planets and validate input
        nplanets = x0.size/6.;
        if (nplanets - np.floor(nplanets) > 0):
            raise Exception('The length of x0 must be a multiple of 6.');

        if (mu.size != nplanets):
            raise Exception('The length of mu must be the length of x0 divided by 6');

        self.nplanets = int(nplanets)
        self.mu = np.squeeze(mu)
        if (self.mu.size == 1):
            self.mu = np.array(mu)

        self.epsmult = epsmult
        self.updateState(np.squeeze(x0))

    def updateState(self,x0):
        self.x0 = x0
        
        #create position and velocity matrices
        tmp = np.reshape(self.x0,(self.nplanets,6)).T
        r0 = tmp[0:3]
        v0 = tmp[3:6]

        #constants and allocation
        self.r0norm = np.sqrt(sum(r0**2.,0))
        self.nu0 = sum(r0*v0,0)
        self.beta = 2*self.mu/self.r0norm - sum(v0*v0,0)
        

    def takeStep(self,dt):
        Phi = self.calcSTM(dt)
        self.updateState(np.dot(Phi,self.x0))
        

    def calcSTM(self,dt):
        #allocate
        u = np.zeros(self.nplanets)
        deltaU = np.zeros(self.beta.size)
        t = np.zeros(self.nplanets);
        counter = 0;
        
        #For elliptic orbits, calculate period effects
        eorbs = self.beta > 0
        if any(eorbs):
            P = 2*pi*self.mu[eorbs]*self.beta[eorbs]**(-3./2.)
            n = np.floor((dt + P/2 - 2*self.nu0[eorbs]/self.beta[eorbs])/P)
            deltaU[eorbs] = 2*pi*n*self.beta[eorbs]**(-5./2.)


        #loop until convergence of the time array to the time step
        while (np.max(np.abs(t-dt)) > self.epsmult*np.spacing(dt)) and (counter < 1000):
            q = self.beta*u**2./(1+self.beta*u**2.)
            U0w2 = 1. - 2.*q
            U1w2 = 2.*(1.-q)*u
            temp = self.contFrac(q)
            U = 16./15.*U1w2**5.*temp + deltaU
            U0 = 2.*U0w2**2.-1.
            U1 = 2.*U0w2*U1w2
            U2 = 2.*U1w2**2.
            U3 = self.beta*U + U1*U2/3.
            r = self.r0norm*U0 + self.nu0*U1 + self.mu*U2
            t = self.r0norm*U1 + self.nu0*U2 + self.mu*U3
            u = u - (t-dt)/(4.*(1.-q)*r)
            counter += 1

        if (counter == 1000):
            raise ValueError('Failed to converge on t: %e/%e'%(np.max(np.abs(t-dt)), self.epsmult*np.spacing(dt)))

        #Kepler solution
        f = 1 - self.mu/self.r0norm*U2
        g = self.r0norm*U1 + self.nu0*U2
        F = -self.mu*U1/r/self.r0norm
        G = 1 - self.mu/r*U2

        Phi = np.zeros([6*self.nplanets]*2)
        for j in np.arange(self.nplanets):
            st = j*6
            Phi[st:st+6,st:st+6] = np.vstack((np.hstack((np.eye(3)*f[j], np.eye(3)*g[j])),np.hstack((np.eye(3)*F[j], np.eye(3)*G[j]))))

        return Phi

    def contFrac(self, x, a = 5., b = 0., c = 5./2.):
        #initialize
        k = 1 - 2*(a-b)
        l = 2*(c-1)
        d = 4*c*(c-1)
        n = 4*b*(c-a)
        A = np.ones(x.size)
        B = np.ones(x.size)
        G = np.ones(x.size)

        Gprev = np.zeros(x.size)+2
        counter = 0
        #loop until convergence of continued fraction
        while (np.max(np.abs(G-Gprev)) > self.epsmult*np.max(np.spacing(G))) and (counter < 1000):
            k = -k
            l = l+2.
            d = d+4.*l
            n = n+(1.+k)*l
            A = d/(d - n*A*x)
            B = (A-1.)*B
            Gprev = G
            G = G + B
            counter += 1

        if (counter == 1000):
            raise ValueError('Failed to converge on G, most likely due to divergence in continued fractions.');

        return G

