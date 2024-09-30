import numpy as np
import numpy.linalg as la
from sympy import *
from scipy.linalg import lu_factor, lu_solve
from STMint.STMint import STMint
import matplotlib.pyplot as plt
import math 
import numba as nb 
import scipy.integrate as integrate


# class containing variational data and ability to solve BVPs
class OrbitVariationalDataSecondOrder:
    # lowest level STMs, exponent for 2^exponent of these STMs
    def __init__(self, STTs, STMs, trvs, T, exponent):
        self.STMs = STMs
        # stts are energy output only
        self.STTs = STTs
        self.T = T
        self.ts = trvs[:, 0]
        self.rs = trvs[:, 1:4]
        self.vs = trvs[:, 4:]
        self.exponent = exponent
        self.refinedList = [STMs]
        self.refinedListSTTs = [STTs]
        self.constructAdditionalLevels()
     #  print(self.T / (2 ** self.exponent) + 10e-12)

    # Function to find STM and STT along two combined subintervals
    # The cocycle conditon equation is used to find Phi(t2,t0)=Phi(t2,t1)*Phi(t1,t0)
    # and the generalized cocycle condition is used to find Psi(t2,t0)
    def cocycle2(self, stm10, stt10, stm21, stt21):
        stm20 = np.matmul(stm21, stm10)
        # stt20 = np.einsum('il,ljk->ijk', stm21, stt10) + np.einsum('ilm,lj,mk->ijk', stt21, stm10, stm10)
        # cocycles for stt with energy output only
        stt20 = stt10 + np.einsum("lm,lj,mk->jk", stt21, stm10, stm10)
        return [stm20, stt20]

    # create structure with most refined STMs at [0], and the monodromy matrix at [exponent]
    def constructAdditionalLevels(self):
        for i in range(self.exponent):
            stms1 = []
            stts1 = []
            for j in range(2 ** (self.exponent - i - 1)):
                STMCombined, STTCombined = self.cocycle2(
                    self.refinedList[i][2 * j],
                    self.refinedListSTTs[i][2 * j],
                    self.refinedList[i][2 * j + 1],
                    self.refinedListSTTs[i][2 * j + 1],
                )
                stms1.append(STMCombined)
                stts1.append(STTCombined)
            self.refinedListSTTs.append(stts1)
            self.refinedList.append(stms1)

    # takes self.exponent number matrix mults/cocycle conditions in a binary search style
    def findSTMAux(self, t0, tf):
        foundCoarsestLevel = False
        for i in range(self.exponent + 1):
            j = self.exponent - i
            stepLength = self.T / (2.0**i)
            location0 = (int)(t0 // stepLength)
            locationf = (int)(tf // stepLength)
            # this is the coarsest level at which there is a full precomputed STM between the two times
            if not foundCoarsestLevel:
                if locationf - location0 >= 2:
                    foundCoarsestLevel = True
                    leftPoint = (location0 + 1) * (2**j)
                    rightPoint = locationf * (2**j)
                    stm = self.refinedList[j][location0 + 1]
                    stt = self.refinedListSTTs[j][location0 + 1]
                    # if more than 2 periods
                    if locationf - location0 == 3:
                        stm, stt = self.cocycle2(
                            stm,
                            stt,
                            self.refinedList[j][location0 + 2],
                            self.refinedListSTTs[j][location0 + 2],
                        )
            else:
                # left and right points of the already constructed STM
                lp = (int)(leftPoint // ((2**j)))
                rp = (int)(rightPoint // ((2**j)))
                if lp - location0 == 2:
                    stm, stt = self.cocycle2(
                        self.refinedList[j][location0 + 1],
                        self.refinedListSTTs[j][location0 + 1],
                        stm,
                        stt,
                    )
                    leftPoint -= 2**j
                if locationf - rp == 1:
                    stm, stt = self.cocycle2(
                        stm, stt, self.refinedList[j][rp], self.refinedListSTTs[j][rp]
                    )
                    rightPoint += 2**j
        # approximate at the endpoints
        if not foundCoarsestLevel:
            smallestPieceTime = self.T / 2.0**self.exponent
            location0 = (int)(t0 // smallestPieceTime)
            locationf = (int)(tf // smallestPieceTime)
            if location0 == locationf:
                stm = np.identity(12) + (tf - t0) / smallestPieceTime * (
                    self.STMs[location0] - np.identity(12)
                )
                stt = (tf - t0) / smallestPieceTime * self.STTs[location0]
            else:
                line = smallestPieceTime * locationf
                stmLeft = np.identity(12) + (line - t0) / smallestPieceTime * (
                    self.STMs[location0] - np.identity(12)
                )
                sttLeft = (line - t0) / smallestPieceTime * self.STTs[location0]
                stmRight = np.identity(12) + (tf - line) / smallestPieceTime * (
                    self.STMs[locationf] - np.identity(12)
                )
                sttRight = (tf - line) / smallestPieceTime * self.STTs[locationf]
                stm, stt = self.cocycle2(stmLeft, sttLeft, stmRight, sttRight)
        else:
            smallestPieceTime = self.T / 2.0**self.exponent
            leftPointT = smallestPieceTime * leftPoint
            rightPointT = smallestPieceTime * rightPoint
            leftContribution = np.identity(12) + (
                leftPointT - t0
            ) / smallestPieceTime * (self.STMs[leftPoint - 1] - np.identity(12))
            leftContributionSTT = (
                (leftPointT - t0) / smallestPieceTime * (self.STTs[leftPoint - 1])
            )
            stm, stt = self.cocycle2(leftContribution, leftContributionSTT, stm, stt)
            rightContribution = np.identity(12) + (
                tf - rightPointT
            ) / smallestPieceTime * (self.STMs[rightPoint] - np.identity(12))
            rightContributionSTT = (
                (tf - rightPointT) / smallestPieceTime * self.STTs[rightPoint]
            )
            stm, stt = self.cocycle2(stm, stt, rightContribution, rightContributionSTT)
        return stm, stt

    # calculate the STM (and STT) for a given start and end time
    def findSTM(self, t0, tf):
        assert tf >= t0 
        left = (int)(t0 // self.T)
        right = (int)(tf // self.T)
        t0 = t0 % self.T
        tf = tf % self.T
        if left == right:
            stm, stt = self.findSTMAux(t0, tf)
        else:
            stm, stt = self.findSTMAux(t0, self.T - 10e-12)
            stmf, sttf = self.findSTMAux(0.0, tf)
            if right - left > 1:
                # stmmid = np.linalg.matrix_power(self.refinedList[-1][0], right-left-1)
                stmmid = self.refinedList[-1][0]
                sttmid = self.refinedListSTTs[-1][0]
                for i in range(right - left - 1):
                    stmmid, sttmid = self.cocycle2(
                        stmmid,
                        sttmid,
                        self.refinedList[-1][0],
                        self.refinedListSTTs[-1][0],
                    )
                stm, stt = self.cocycle2(stm, stt, stmmid, sttmid)
            stm, stt = self.cocycle2(stm, stt, stmf, sttf)
        return stm, stt

    # find relative rotating frame velocity that gives inertial relative velocity of zero
    def findRotRelVel(self, rrel):
        return np.array([rrel[1], -1.0 * rrel[0], 0.0])

    # precompute necessary quantities for repeated calling of different transfers in same time ranges
    def precompute_lu(self, t0, tf):
        stm, stt = self.findSTM(t0, tf)
        lu, piv = lu_factor(stm[:6, 6:12])
        return (stm[:6, :6], stm[6:12, :6], stm[6:12, 6:], stt, lu, piv)

    # find the approximate cost of a relative transfer (for repeated calls with same initial and final times)
    # positions supplied in au
    # rotating frame velocities in canonical units
    # return energy cost in canonical units
    def solve_bvp_cost(self, stmxx, stmlx, stmll, stt, lu, piv, x0rel, xfrel):
        l0rel = lu_solve((lu, piv), xfrel - np.matmul(stmxx, x0rel))
        print(l0rel)
        relAugState = np.concatenate((x0rel, l0rel))
        return np.einsum("jk,j,k->", stt, relAugState, relAugState)

    # find the approximate cost of a relative transfer (for repeated calls with same initial and final times)
    # takes in the output of precompute_lu in the precomputeData field
    # positions supplied in km
    # Assume inertial relative velocities are zero
    # return energy in canonical units
    def solve_bvp_cost_convenience(self, precomputeData, r0rel, rfrel):
        (stmxx, stmlx, stmll, stt, lu, piv) = precomputeData
        r0rel = self.posKMtoAU(r0rel)
        rfrel = self.posKMtoAU(rfrel)
        v0rel = self.findRotRelVel(r0rel)
        vfrel = self.findRotRelVel(rfrel)
        # all in canonical units at this point-can be fed into bvp_solver function
        en = self.solve_bvp_cost(
            stmxx,
            stmlx,
            stmll,
            stt,
            lu,
            piv,
            np.concatenate((r0rel, v0rel)),
            np.concatenate((rfrel, vfrel)),
        )
        return en



    def fetchQuad(self, t0, tf):
        ts = np.array(self.ts)
        STMSS = np.array(self.STMs)
        assert tf >= t0

        part_length = self.T / (2 ** self.exponent)
        num_parts = self.T // part_length
        num_multiples = math.floor(tf / part_length) - math.floor( t0 / part_length + 1) + 1
        if num_multiples == 0:
            quad = [self.findSTM(t0, tf)[0]]
            dts = [tf - t0]
        if num_multiples == 1:
            if t0 % part_length == 0:
                quad = [self.STMs[(t0 % self.T) // part_length]]
                dts = [part_length]
            else:
                quad = [self.findSTM(t0, tf)[0]]
                dts = [tf - t0]
        if num_multiples >= 2:
            if t0 % part_length <= 10e-12 and tf % part_length <= 10e-12:
                first_idx = (int) ((t0 % self.T) // part_length)
                indics = np.arange(0, num_multiples, 1)
                indics = indics + first_idx
                indics = indics % num_parts
                quad = list(STMSS[indics.astype(int)])
                dts = np.full_like(indics, part_length)
            elif t0 % part_length <= 10e-12 and tf % part_length >= 10e-12:
                first_idx = (int) ((t0 % self.T) // part_length)
                indics = np.arange(0, num_multiples, 1)
                indics = indics + first_idx
                indics = indics % num_parts

                quad = list(STMSS[indics.astype(int)])
                dts = np.full_like(indics, part_length)
                dts = np.append(dts, tf - (t0 + num_multiples * part_length))
                quad.append(self.findSTM(t0 + num_multiples * part_length, tf)[0])

            elif t0 % part_length >= 10e-12 and tf % part_length <= 10e-12:
                first_time = ((t0 + part_length) // part_length) * part_length
                quad = [self.findSTM(t0, first_time)[0]]
                dts = np.array([first_time - t0])
                indics = np.arange(0, num_multiples - 1, 1)
                first_idx = (int) ((first_time % self.T) // part_length)
                indics = indics + first_idx
                indics = indics % num_parts
                quad.extend(list(STMSS[indics.astype(int)]))
                dts = np.concatenate((dts, np.full_like(indics, part_length)))
                quad.insert(7,self.refinedList[-1][0])
            else:
                first_time = ((t0 + part_length) // part_length) * part_length
                quad = [self.findSTM(t0, first_time)[0]]
                dts = np.array([first_time - t0])
                indics = np.arange(0, num_multiples - 1, 1)
                first_idx = (int) ((first_time % self.T) // part_length)
                indics = indics + first_idx
                indics = indics % num_parts

                quad.extend(list(STMSS[indics.astype(int)]))
                dts = np.concatenate((dts, np.full_like(indics, part_length)))
                last_time = first_time + (num_multiples - 1) * part_length
                quad.append(self.findSTM(last_time, tf)[0])
                dts = np.append(dts, tf - last_time)

        return quad, dts

    def deltaV(self, stmxx, lu, piv, r0rel, rfrel, t0, tf):

        r0rel = self.posKMtoAU(r0rel)
        rfrel = self.posKMtoAU(rfrel)
        v0rel = self.findRotRelVel(r0rel)
        vfrel = self.findRotRelVel(rfrel)


        x0rel = np.concatenate((r0rel, v0rel))
        xfrel = np.concatenate((rfrel, vfrel))

        l0rel = lu_solve((lu, piv), xfrel - np.matmul(stmxx, x0rel)) 


        quadrature, dts = self.fetchQuad(t0, tf)
        vec = np.concatenate((x0rel, l0rel))
        dts = list(dts)
        dts.insert(0, 0)
        dts = np.array(dts)
        
        quadrature = np.array(quadrature)
        @nb.jit(nopython=True)
        def quick_integrate(quadrature, x0):
            testing = [x0]
            acc = x0
            for i in range(len(quadrature)):
                acc = quadrature[i] @ acc
                testing.append(acc)
            return testing 
        states = quick_integrate(quadrature, vec.flatten())
        states = np.array(states)[:, 9 : 12]
        lams = np.linalg.norm(states, axis=1)
        
        dV = integrate.simpson(lams, np.full(dts.shape[0], t0) + np.cumsum(dts))
        return dV

    # convert position from KM to AU
    def posKMtoAU(self, pos):
        return pos / 149597870.7
