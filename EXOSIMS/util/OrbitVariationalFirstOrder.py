import numpy as np
from sympy import *
from scipy.linalg import lu_factor, lu_solve


class OrbitVariationalDataFirstOrder:
    """Second Order Variational Data Class
    This class is implemented about a 6-month L2 Halo orbit, and computes STMS 
    for fast approximate solution to the optimal control problem of starshade orbit tranfers between star lines of site.
    """
    def __init__(self, STMs, trvs, T, exponent):

        """Initializes First Order Variational data class.
        Args:
            STMs:
                State transition tensors for more accurate second order aproximation to the initial costates for solution of the BVP.
            trvs: 
                time, position and velocity array found by numerically integrating the variational equations over the reference orbit.
            T:
                reference orbit period
            exponent: 
                2^exponent subdivisions used in precalculating variational data.

        """


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
            stm10: 
                STM(t0, t1)
            stm21: 
                STM(t1, tf)
        """
        stm20 = np.matmul(stm21, stm10)
        return stm20

    def constructAdditionalLevels(self):
        """ Constructs STMs and STTs for precomputation.
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
        """ helper method for findSTM
        Args:
            t0: 
                initial time
            tf: 
                final time
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
        """ finds STM and STT associated with t0, tf
        Args:
            t0: 
                initial time
            tf: 
                final time
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
        """ find relative rotating frame velocity that gives inertial relative velocity of zero
        """
        return np.array([rrel[1], -1.0 * rrel[0], 0.0])

    def precompute_lu(self, t0, tf):
        """ precompute necessary quantities for repeating calling of different transfers with the same t0, tf
        Args:
            t0: 
                initial time
            tf: 
                final time 
        """
        stm = self.findSTM(t0, tf)
        lu, piv = lu_factor(stm[0:3, 3:6])
        return stm[0:3, 0:3], stm[3:6, 3:6], stm[3:6, 0:3], lu, piv

    def deltaV(self, stmrr, stmvv, stmvr, r0, rf, v0, vftarget, lu, piv):
        """ compute delta v for slew between initial and final relative positions, with terminal times t0 and tf 

        Args:
            stmrr:
                position position component of stm
            stmvv:
                velocity velocity component of stm
            stmvr:
                velocity position component of stm 
            v0:
                initial relative velocity 
            vftarget:
                target final relative velocity
            lu:
                lu factorization of stm
            piv:
                part of lu factorizaiton of stm 
            r0:
                position relative coordinates initial
            rf:
                position relative coordinates final
        """
        
        v0_req = lu_solve((lu, piv), rf - stmrr @ r0)
        deltaV0 = v0_req - v0
        vf_transfer = stmvr @ r0 + stmvv @ (v0 + deltaV0)
        deltaV1 = vftarget - vf_transfer
        return np.linalg.norm(deltaV0) + np.linalg.norm(deltaV1)

    def posKMtoAU(self, pos):
        """ helper method for converting positions to km 
        """
        return pos / 149597870.7

    def solve_deltaV_convenience(self, precomputeData, r0rel, rfrel):
        """ solve boundary value problem convenience method

        Args:
            precomputeData:
                lu and piv factorization
            r0Rel:
                position relative coordinates initial
            rfrel:
                position relative coordinates final
        """
        
        stmrr, stmvv, stmvr, lu, piv = precomputeData
        r01rel = self.posKMtoAU(r0rel)
        rf1rel = self.posKMtoAU(rfrel)
        v0rel = self.findRotRelVel(r01rel)
        vfrel = self.findRotRelVel(rf1rel)
        dV = self.deltaV(stmrr, stmvv, stmvr, r01rel, rf1rel, v0rel, vfrel, lu, piv)
        return dV
