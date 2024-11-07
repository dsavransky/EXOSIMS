import numpy as np
from scipy.linalg import lu_factor, lu_solve
import math
import numba as nb
import scipy.integrate as integrate


# class containing variational data and ability to solve BVPs
class OrbitVariationalDataSecondOrder:

    """Initializes First Order Variational data class. Second Order Variational Data Class
    This class is implemented about a 6-month L2 Halo orbit, and computes first order STMs and optionally STTs
    for fast approximate solution to the optimal control problem of starshade orbit tranfers between star lines of site.
    Args:
        STTs:
            State transition tensors for more accurate second order approximation to the initial costates for solution of the BVP.
        STMs:
            State transition matrices for first order aproximation to the initial costates for solution of the BVP.
        trvs:
            time, position and velocity array found by numerically integrating the variational equations over the reference orbit.
        T:
            reference orbit period
        exponent:
            2^exponent subdivisions used in precalculating variational data.
    
    Attributes: 
        STMs:
            State transition tensors for more accurate second order aproximation to the initial costates for solution of the BVP.
        trvs:
            time, position and velocity array found by numerically integrating the variational equations over the reference orbit.
        T:
            reference orbit period
        exponent:
            2^exponent subdivisions used in precalculating variational data.    
        ts: 
            time quadrature over which variational equations are numerically integrated.
        rs:
            position array found by numerically integrating the variational equations over the reference orbit.
        vs:
            velocity array found by numerically integrating the variational equations over the reference orbit.
        refinedList:
            pre-compiled list STMs at all possible densities associated with the time discretization 
    """


   

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

    def cocycle2(self, stm10, stt10, stm21, stt21):
        """Computes state transition matrices and state transition tensors for t0, tf given STMs and STTS for t0, t1 and t1, tf using the generalized cocycle conditions
        Args:
            stm10:
                STM(t0, t1)
            stt10:
                STT(t0, t1)
            stm21:
                STM(t1, tf)
            stt21:
                STT(t1, tf)
        Returns: 
            list(~np.ndarray(float)):
                The state transition matrix and tensor associated with t0, tf
        """
        stm20 = np.matmul(stm21, stm10)
        stt20 = stt10 + np.einsum("lm,lj,mk->jk", stt21, stm10, stm10)
        return [stm20, stt20]

    def constructAdditionalLevels(self):
        """Constructs STMs and STTs for precomputation.
        Args:
        Returns: 
        """
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

    def findSTMAux(self, t0, tf):
        """helper method for findSTM
        Args:
            t0 (float):
                initial time in canonical CRTBP units
            tf (float):
                final time in canonical CRTBP units 
        Returns: 
            Tuple:
                stm (~np.ndarray(float)):
                    Returns the stm associated with t0, and tf
                stt (~np.ndarray(float)):
                    Returns the stt associated with t0, and tf
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

    def findSTM(self, t0, tf):
        """finds STM associated with t0, tf
        Args:
            t0 (float):
                initial time in canonical CRTBP units
            tf (float):
                final time in canonical CRTBP units
        Returns: 
            Tuple:
                stm (~np.ndarray(float)):
                    Returns the stm associated with t0, and tf
                stt (~np.ndarray(float)):
                    Returns the stt associated with t0, and tf
        """
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
        """find relative rotating frame velocity that gives inertial relative velocity of zero
        Args:
            rrel(float):
                initial relative rotating position velocity
        Returns:
            ~np.ndarray(float)
                Returns the stm associated with t0, and tf
        """
        return np.array([rrel[1], -1.0 * rrel[0], 0.0])

    # precompute necessary quantities for repeated calling of different transfers in same time ranges
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

        stm, stt = self.findSTM(t0, tf)
        lu, piv = lu_factor(stm[:6, 6:12])
        return (stm[:6, :6], stm[6:12, :6], stm[6:12, 6:], stt, lu, piv)

    def fetchQuad(self, t0, tf):
        """computes STM quadrature for integration of control effort

        Args:
            t0 (float):
                initial time in canonical CRTBP units
            tf (float):
                final time in canonical CRTBP units 
        Returns:
            tuple:
                quad (list(~np.ndarray(float))):
                    quadrature over which to compute energy cost of the integral
                dts (list(float)):
                    delta ts associated with the quadrature
        """
        STMSS = np.array(self.STMs)
        assert tf >= t0

        part_length = self.T / (2**self.exponent)
        num_parts = self.T // part_length
        num_multiples = (
            math.floor(tf / part_length) - math.floor(t0 / part_length + 1) + 1
        )
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
                first_idx = (int)((t0 % self.T) // part_length)
                indics = np.arange(0, num_multiples, 1)
                indics = indics + first_idx
                indics = indics % num_parts
                quad = list(STMSS[indics.astype(int)])
                dts = np.full_like(indics, part_length)
            elif t0 % part_length <= 10e-12 and tf % part_length >= 10e-12:
                first_idx = (int)((t0 % self.T) // part_length)
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
                first_idx = (int)((first_time % self.T) // part_length)
                indics = indics + first_idx
                indics = indics % num_parts
                quad.extend(list(STMSS[indics.astype(int)]))
                dts = np.concatenate((dts, np.full_like(indics, part_length)))
                quad.insert(7, self.refinedList[-1][0])
            else:
                first_time = ((t0 + part_length) // part_length) * part_length
                quad = [self.findSTM(t0, first_time)[0]]
                dts = np.array([first_time - t0])
                indics = np.arange(0, num_multiples - 1, 1)
                first_idx = (int)((first_time % self.T) // part_length)
                indics = indics + first_idx
                indics = indics % num_parts

                quad.extend(list(STMSS[indics.astype(int)]))
                dts = np.concatenate((dts, np.full_like(indics, part_length)))
                last_time = first_time + (num_multiples - 1) * part_length
                quad.append(self.findSTM(last_time, tf)[0])
                dts = np.append(dts, tf - last_time)

        return quad, dts

    def deltaV(self, stmxx, lu, piv, r0rel, rfrel, t0, tf):
        """compute delta v for slew between initial and final relative positions, with terminal times t0 and tf

        Args:
            stmxx:
                position position component of stm
            lu:
                lu factorization of stm
            piv:
                part of lu factorizaiton of stm
            r0rel:
                position relative coordinates initial
            rfrel:
                position relative coordinates final
            t0:
                initial time
            tf:
                final time
        Returns:
            float:
                energy-optimal delta-v associated with a relative transfer between points 
        """

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

       # @nb.jit(nopython=True)
        def quick_integrate(quadrature, x0):
            testing = [x0]
            acc = x0
            for i in range(len(quadrature)):
                acc = quadrature[i] @ acc
                testing.append(acc)
            return testing

        states = quick_integrate(quadrature, vec.flatten())
        states = np.array(states)[:, 9:12]
        lams = np.linalg.norm(states, axis=1)

        dV = integrate.simpson(y=lams, x=np.full(dts.shape[0], t0) + np.cumsum(dts))
        return dV

    # convert position from KM to AU
    def posKMtoAU(self, pos):
        """precompute necessary quantities for repeating calling of different transfers with the same t0, tf
        Args:
            pos (float):
                position in km 
        Returns: 
            float:
                position in AU 
        """
        return pos / 149597870.7
