import numpy as np
import numpy.linalg as la
from sympy import *
from scipy.linalg import lu_factor, lu_solve
from STMint.STMint import STMint
import matplotlib.pyplot as plt

class OrbitVariationalDataFirstOrder:
	def __init__(self, STMs, trvs, T, exponent):
		self.STMs = STMs
		self.T = T
		self.ts = trvs[:, 0]
		self.rs = trvs[:, 1:4]
		self.vs = trvs[:, 4:]
		self.exponent = exponent
		self.refinedList = [STMs]
		self.constructAdditionalLevels()

	# Function to find STM and STT along two combined subintervals
	# The cocycle conditon equation is used to find Phi(t2,t0)=Phi(t2,t1)*Phi(t1,t0)
	# and the generalized cocycle condition is used to find Psi(t2,t0)
	def cocycle1(self, stm10, stm21):
		stm20 = np.matmul(stm21, stm10)
		# stt20 = np.einsum('il,ljk->ijk', stm21, stt10) + np.einsum('ilm,lj,mk->ijk', stt21, stm10, stm10)
		# cocycles for stt with energy output only
		# stt20 = stt10 + np.einsum('lm,lj,mk->jk', stt21, stm10, stm10)
		return stm20

	# create structure with most refined STMs at [0], and the monodromy matrix at [exponent]
	def constructAdditionalLevels(self):
		for i in range(self.exponent):
			stms1 = []
			for j in range(2 ** (self.exponent - i - 1)):
				STMCombined = self.cocycle1(
					self.refinedList[i][2 * j], self.refinedList[i][2 * j + 1]
				)
				stms1.append(STMCombined)
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

	# calculate the STM (and STT) for a given start and end time
	def findSTM(self, t0, tf):
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
				# stmmid = np.linalg.matrix_power(self.refinedList[-1][0], right-left-1)
				stmmid = self.refinedList[-1][0]
				for i in range(right - left - 1):
					stmmid = self.cocycle1(stmmid, self.refinedList[-1][0])
				stm = self.cocycle1(stm, stmmid)
			stm = self.cocycle1(stm, stmf)
		return stm

	# find relative rotating frame velocity that gives inertial relative velocity of zero
	def findRotRelVel(self, rrel):
		return np.array([rrel[1], -1.0 * rrel[0], 0.0])

	def precompute_lu(self, t0, tf):
		stm = self.findSTM(t0, tf)
		lu, piv = lu_factor(stm[0:3, 3:6])
		return stm[0:3, 0:3], stm[3 : 6, 3 : 6], stm[3 : 6, 0 : 3],  lu, piv

	#positions supplied in au
	#rotating frame velocities in canonical units
	#remmber to convert to proper units before using this 
	def deltaV(self, stmrr, stmvv, stmvr, r0, rf, v0, vftarget, lu, piv): 
		v0_req = lu_solve((lu, piv), rf - stmrr @ r0)
		deltaV0 = v0_req - v0
		vf_transfer = stmvr @ r0 + stmvv @ (v0 + deltaV0)
		deltaV1 = vftarget - vf_transfer
		return np.linalg.norm(deltaV0) + np.linalg.norm(deltaV1)

	def posKMtoAU(self, pos):
		return pos / 149597870.7
	
	#find the approximate cost of a relative transfer (for repeated calls with same initial and final times)
	#takes in the output of precompute_lu in the precomputeData field
	#positions supplied in km
	#Assume inertial relative velocities are zero
	#return deltaV in canonical units
	def solve_deltaV_convenience(self, precomputeData, r0rel, rfrel):
		stmrr, stmvv, stmvr, lu, piv = precomputeData
		r01rel = self.posKMtoAU(r0rel)
		rf1rel = self.posKMtoAU(rfrel)
		v0rel = self.findRotRelVel(r01rel)
		vfrel = self.findRotRelVel(rf1rel)
		dV = self.deltaV(stmrr, stmvv, stmvr, r01rel, rf1rel, v0rel, vfrel, lu, piv)
		return dV



