import numpy as np
from scipy.linalg import lu_factor, lu_solve


class OrbitVariationalDataFirstOrder:


    """ First Order Variational Data Class
        This class is implemented about a 6-month L2 Halo orbit, and computes STMS
        for fast approximate solution to the optimal control problem of starshade orbit tranfers between star lines of site.
        Args:
            STMs (~numpy.ndarray):
                State transition matrices for first order aproximation to the initial costates for solution of the BVP.
            trvs (~numpy.ndarray):
                time, position and velocity array found by numerically integrating the variational equations over the reference orbit.
            T (~np.float64):
                reference orbit period
            exponent (int):
                2^exponent subdivisions used in precalculating variational data.
        
        Attributes: 
            STMs (~numpy.ndarray):
                State transition matrices for first order aproximation to the initial costates for solution of the BVP.
            trvs (~numpy.ndarray):
                time, position and velocity array found by numerically integrating the variational equations over the reference orbit.
            T (~np.float64):
                reference orbit period
            exponent (int):
                2^exponent subdivisions used in precalculating variational data.    
            ts (~numpy.ndarray): 
                time quadrature over which variational equations are numerically integrated.
            rs (~numpy.ndarray):
                position array found by numerically integrating the variational equations over the reference orbit.
            vs (~numpy.ndarray):
                velocity array found by numerically integrating the variational equations over the reference orbit.
            refinedList (list(~numpy.ndarray)):
                pre-compiled list STMs at all possible densities associated with the time discretization 
        """

    def __init__(self, STMs, trvs, T, exponent):
        
        self.STMs = STMs
        self.T = T
        self.ts = trvs[:, 0]
        self.rs = trvs[:, 1:4]
        self.vs = trvs[:, 4:]
        self.exponent = exponent
        self.refinedList = [STMs]
        self.constructAdditionalLevels()

    def cocycle1(self, stm10, stm21):
        """Computes state transition matrices for t0, tf given STMs  for t0, t1 and t1, tf using the generalized cocycle conditions
        Args:
            stm10 (~numpy.ndarray(float)):
                STM(t0, t1)
            stm21 (~numpy.ndarray(float)):
                STM(t1, tf)
        Returns: 
            ~np.ndarray(float)
                The state transition matrix associated with t0, tf
        """
        stm20 = np.matmul(stm21, stm10)
        return stm20

    def constructAdditionalLevels(self):
        """Constructs STMs for precomputation.
        Args:
        Returns: 
        """
        for i in range(self.exponent):
            stms1 = []
            for j in range(2 ** (self.exponent - i - 1)):
                STMCombined = self.cocycle1(
                    self.refinedList[i][2 * j], self.refinedList[i][2 * j + 1]
                )
                stms1.append(STMCombined)
            self.refinedList.append(stms1)

    def findSTMAux(self, t0, tf):
        """helper method for findSTM
        Args:
            t0 (float):
                initial time in canonical CRTBP units
            tf (float):
                final time in canonical CRTBP units 
        Returns: 
            ~np.ndarray(float)
                Returns the stm associated with t0, and tf
        """
        foundCoarsestLevel = False
        for i in range(self.exponent + 1):
            j = self.exponent - i
            stepLength = self.T / (2.0**i)
            location0 = (int)(t0 // stepLength)
            locationf = (int)(tf // stepLength)
            if not foundCoarsestLevel:
                if locationf - location0 >= 2:
                    foundCoarsestLevel = True
                    leftPoint = (location0 + 1) * (2**j)
                    rightPoint = locationf * (2**j)
                    stm = self.refinedList[j][location0 + 1]
                    # if more than 2 periods
                    if locationf - location0 == 3:
                        stm = self.cocycle1(stm, self.refinedList[j][location0 + 2])
            else:
                # left and right points of the already constructed STM
                lp = (int)(leftPoint // ((2**j)))
                rp = (int)(rightPoint // ((2**j)))
                if lp - location0 == 2:
                    stm = self.cocycle1(self.refinedList[j][location0 + 1], stm)
                    leftPoint -= 2**j
                if locationf - rp == 1:
                    stm = self.cocycle1(stm, self.refinedList[j][rp])
                    rightPoint += 2**j
        # approximate at the endpoints
        if not foundCoarsestLevel:
            smallestPieceTime = self.T / 2.0**self.exponent
            location0 = (int)(t0 // smallestPieceTime)
            locationf = (int)(tf // smallestPieceTime)
            if location0 == locationf:
                stm = np.identity(6) + (tf - t0) / smallestPieceTime * (
                    self.STMs[location0] - np.identity(6)
                )
            else:
                line = smallestPieceTime * locationf
                stmLeft = np.identity(6) + (line - t0) / smallestPieceTime * (
                    self.STMs[location0] - np.identity(6)
                )
                stmRight = np.identity(6) + (tf - line) / smallestPieceTime * (
                    self.STMs[locationf] - np.identity(6)
                )
                stm = self.cocycle1(stmLeft, stmRight)
        else:
            smallestPieceTime = self.T / 2.0**self.exponent
            leftPointT = smallestPieceTime * leftPoint
            rightPointT = smallestPieceTime * rightPoint
            leftContribution = np.identity(6) + (
                leftPointT - t0
            ) / smallestPieceTime * (self.STMs[leftPoint - 1] - np.identity(6))
            stm = self.cocycle1(leftContribution, stm)
            rightContribution = np.identity(6) + (
                tf - rightPointT
            ) / smallestPieceTime * (self.STMs[rightPoint] - np.identity(6))
            stm = self.cocycle1(stm, rightContribution)
        return stm

    def findSTM(self, t0, tf):
        """finds STM associated with t0, tf
        Args:
            t0 (float):
                initial time in canonical CRTBP units
            tf (float):
                final time in canonical CRTBP units
        Returns:
            ~np.ndarray(float)
                Returns the stm associated with t0, and tf
        """
        if t0 > tf:
            print("STM requested with t0>tf.")
        left = (int)(t0 // self.T)
        right = (int)(tf // self.T)
        t0 = t0 % self.T
        tf = tf % self.T
        if left == right:
            stm = self.findSTMAux(t0, tf)
        else:
            stm = self.findSTMAux(t0, self.T - 10e-12)
            stmf = self.findSTMAux(0.0, tf)
            if right - left > 1:
                stmmid = self.refinedList[-1][0]
                for i in range(right - left - 1):
                    stmmid = self.cocycle1(stmmid, self.refinedList[-1][0])
                stm = self.cocycle1(stm, stmmid)
            stm = self.cocycle1(stm, stmf)
        return stm

    # find relative rotating framevelocity that gives inertial relative velocity of zero
    def findRotRelVel(self, rrel):
        """find relative rotating frame velocity that gives inertial relative velocity of zero
        Args:
            rrel(~numpy.ndarray(float)):
                initial relative rotating position velocity
        Returns:
            ~np.ndarray(float)
                Returns the stm associated with t0, and tf
        """
        
        return np.array([rrel[1], -1.0 * rrel[0], 0.0])

    def precompute_lu(self, t0, tf):
        """precompute necessary quantities for repeating calling of different transfers with the same t0, tf
        Args:
            t0 (float):
                initial time
            tf (float):
                final time
        Returns: 
            tuple:
                stmrr (~np.ndarray(float)):
                    The componenet of the STM that describes the effect of perturbations to the initial position on perturbations to the final position
                stmvv (~np.ndarray(float)):
                   The componenet of the STM that describes the effect of perturbations to the initial velocity on perturbations to the final velocity
                stmvr (~np.ndarray(float)):
                    The componenet of the STM that describes the effect of perturbations to the initial position on perturbations to the final velocity
                lu (~np.ndarray(float)):
                    The lu component in the lu factorization of the STM
                piv (~np.ndarray(float)):
                    The piv componenet in the lu factorization of the STM
        """
        stm = self.findSTM(t0, tf)
        lu, piv = lu_factor(stm[0:3, 3:6])
        return stm[0:3, 0:3], stm[3:6, 3:6], stm[3:6, 0:3], lu, piv

    def deltaV(self, stmrr, stmvv, stmvr, r0, rf, v0, vftarget, lu, piv):
        """compute delta v for slew between initial and final relative positions, with terminal times t0 and tf
        Args:
            stmrr (~numpy.ndarray(float)):
                position position component of stm
            stmvv (~numpy.ndarray(float)):
                velocity velocity component of stm
            stmvr (~numpy.ndarray(float)):
                velocity position component of stm
            v0 (~numpy.ndarray(float)):
                initial relative velocity
            vftarget (~numpy.ndarray(float)):
                target final relative velocity
            lu (~numpy.ndarray(float)):
                lu factorization of stm
            piv (~numpy.ndarray(float)):
                part of lu factorizaiton of stm
            r0 (~numpy.ndarray(float)):
                position relative coordinates initial
            rf (~numpy.ndarray(float)):
                position relative coordinates final
        Returns:
            float: 
                delta-v associated with a relative transfer between r0 and rf given v0 and vftarget
        """

        v0_req = lu_solve((lu, piv), rf - stmrr @ r0)
        deltaV0 = v0_req - v0
        vf_transfer = stmvr @ r0 + stmvv @ (v0 + deltaV0)
        deltaV1 = vftarget - vf_transfer
        return np.linalg.norm(deltaV0) + np.linalg.norm(deltaV1)

    def posKMtoAU(self, pos):
        """precompute necessary quantities for repeating calling of different transfers with the same t0, tf
        Args:
            pos (~numpy.ndarray(float)):
                position in km 
        Returns: 
            (~numpy.ndarray(float)):
                position in AU 
        """
        return pos / 149597870.7

    def solve_deltaV_convenience(self, precomputeData, r0rel, rfrel):
        """solve boundary value problem convenience method
        Args:
            precomputeData (tuple(~numpy.ndarray(float))):
                lu and piv factorization
            r0rel (~numpy.ndarray(float)):
                position relative coordinates initial
            rfrel (~numpy.ndarray(float)):
                position relative coordinates final
        Returns: 
            float: 
                delta-v cost associated with relative transfers between r0rel and rfrel such that terminal inertial relative velocities are 0
        """

        stmrr, stmvv, stmvr, lu, piv = precomputeData
        r01rel = self.posKMtoAU(r0rel)
        rf1rel = self.posKMtoAU(rfrel)
        v0rel = self.findRotRelVel(r01rel)
        vfrel = self.findRotRelVel(rf1rel)
        dV = self.deltaV(stmrr, stmvv, stmvr, r01rel, rf1rel, v0rel, vfrel, lu, piv)
        return dV
