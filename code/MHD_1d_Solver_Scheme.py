"""
project: MHD_1d_Solver_Scheme.py
Author: Xuyao Hu
Date: Dec. 11, 2016
Description:
	utilize piecewise linear method (PLM), HLLD Riemann solver, 
	CFL GetDt function to construct a complete solver 
	for 1d MHD problem 
"""
import numpy as np

def minmod(x,y,z):
	return 0.25 * abs(np.sign(x)+np.sign(y)) * (np.sign(x) + np.sign(z)) * min(abs(x),abs(y),abs(z))

def GetDt(gamma,N,delta_x,CFL_num,U,Bx):
	#lambda_fast = vx+c_f using cell-centered values
	lambda_fast = np.zeros(N+4,float)
	c_f = np.zeros(N+4,float)
	#primitive variables q
	q = np.zeros([7,N+4],float)
	q[0,:] = U[0,:]
	q[1,:] = U[1,:] / U[0,:]
	q[2,:] = U[2,:] / U[0,:]
	q[3,:] = U[3,:] / U[0,:]
	q[4,:] = (gamma - 1.) * (U[4,:] - 0.5 * q[0,:] * (q[1,:]*q[1,:] + q[2,:]*q[2,:] + q[3,:]*q[3,:]) - 0.5 * (Bx[:]*Bx[:] + U[5,:]*U[5,:] + U[6,:]*U[6,:]))
	q[5,:] = U[5,:]
	q[6,:] = U[6,:]

	c_f[:] = ((gamma * q[4,:] + (Bx[:]*Bx[:] + q[5,:]*q[5,:] + q[6,:]*q[6,:]) + np.sqrt((gamma * q[4,:] + (Bx[:]*Bx[:] + q[5,:]*q[5,:] + q[6,:]*q[6,:]))**2 - 4*gamma*q[4,:]*Bx[:]*Bx[:]))/(2*q[0,:]))**0.5
	lambda_fast[:] = q[1,:] + c_f[:]

	delta_t = CFL_num * delta_x / max(lambda_fast[2:N+2])
	return delta_t

def PCM(gamma,N,delta_x,U,Bx):
	"""
	Piecewise Constant Method
	"""
	"""
	Receive a U vector (conserved quantities), convert it 
	into a Q vector (primitive quantities), which is also a 7*(N+4) 
	[including ghost cells] vector
	"""
	Q = np.zeros([7,N+4],float)
	Q[0,:] = U[0,:]
	Q[1,:] = U[1,:] / U[0,:]
	Q[2,:] = U[2,:] / U[0,:]
	Q[3,:] = U[3,:] / U[0,:]
	Q[4,:] = (gamma - 1.) * (U[4,:] - 0.5 * Q[0,:] * (Q[1,:]*Q[1,:] + Q[2,:]*Q[2,:] + Q[3,:]*Q[3,:]) - 0.5 * (Bx[:]*Bx[:] + U[5,:]*U[5,:] + U[6,:]*U[6,:]))
	Q[5,:] = U[5,:]
	Q[6,:] = U[6,:]

	"""
	Using piecewise linear method to obtain the "left" and "right"
	interface state
	"""
	Q_L = np.zeros([7,N+4],float)
	Q_R = np.zeros([7,N+4],float)

	Q_L[0,1:N+1] = Q[0,1:N+1]
	Q_L[1,1:N+1] = Q[1,1:N+1]
	Q_L[2,1:N+1] = Q[2,1:N+1]
	Q_L[3,1:N+1] = Q[3,1:N+1]
	Q_L[4,1:N+1] = Q[4,1:N+1]
	Q_L[5,1:N+1] = Q[5,1:N+1]
	Q_L[6,1:N+1] = Q[6,1:N+1]

	Q_R[0,1:N+1] = Q[0,2:N+2]
	Q_R[1,1:N+1] = Q[1,2:N+2]
	Q_R[2,1:N+1] = Q[2,2:N+2]
	Q_R[3,1:N+1] = Q[3,2:N+2]
	Q_R[4,1:N+1] = Q[4,2:N+2]
	Q_R[5,1:N+1] = Q[5,2:N+2]
	Q_R[6,1:N+1] = Q[6,2:N+2]

	#First calculate fast magneto-acoustic speed c_f_L (for left state) and c_f_R (for right state)
	#two 1*(N+4) matrices
	c_f_L = np.zeros(N+4,float)
	c_f_R = np.zeros(N+4,float)

	c_f_L[1:N+2] = ((gamma * Q_L[4,1:N+2] + (Bx[1:N+2]*Bx[1:N+2] + Q_L[5,1:N+2]*Q_L[5,1:N+2] + Q_L[6,1:N+2]*Q_L[6,1:N+2]) + np.sqrt((gamma * Q_L[4,1:N+2] + (Bx[1:N+2]*Bx[1:N+2] + Q_L[5,1:N+2]*Q_L[5,1:N+2] + Q_L[6,1:N+2]*Q_L[6,1:N+2]))**2 - 4*gamma*Q_L[4,1:N+2]*Bx[1:N+2]*Bx[1:N+2]))/(2*Q_L[0,1:N+2]))**0.5
	c_f_R[1:N+2] = ((gamma * Q_R[4,1:N+2] + (Bx[1:N+2]*Bx[1:N+2] + Q_R[5,1:N+2]*Q_R[5,1:N+2] + Q_R[6,1:N+2]*Q_R[6,1:N+2]) + np.sqrt((gamma * Q_R[4,1:N+2] + (Bx[1:N+2]*Bx[1:N+2] + Q_R[5,1:N+2]*Q_R[5,1:N+2] + Q_R[6,1:N+2]*Q_R[6,1:N+2]))**2 - 4*gamma*Q_R[4,1:N+2]*Bx[1:N+2]*Bx[1:N+2]))/(2*Q_R[0,1:N+2]))**0.5

	#make up lambda_1_L, lambda_1_R, lambda_7_L, lambda_7_R
	lambda_1_L = np.zeros(N+4,float)
	lambda_1_R = np.zeros(N+4,float)
	lambda_7_L = np.zeros(N+4,float)
	lambda_7_R = np.zeros(N+4,float)

	lambda_1_L = Q_L[1,:] - c_f_L[:]
	lambda_1_R = Q_R[1,:] - c_f_R[:]
	lambda_7_L = Q_L[1,:] + c_f_L[:]
	lambda_7_R = Q_R[1,:] + c_f_R[:]

	"""
	make up the minimum signal speed S_L (negative) and the maximum signal speed S_R (positve)
	using lambda_1_L, lambda_1_R, lambda_7_L, lambda_7_R
	S_L = min(lambda_1_L, lambda_1_R)
	S_R = max(lambda_7_L, lambda_7_R)
	"""
	S_L = np.zeros(N+4,float)
	S_R = np.zeros(N+4,float)

	for i in range(N+4):
		S_L[i] = min(lambda_1_L[i], lambda_1_R[i])
		S_R[i] = max(lambda_7_L[i], lambda_7_R[i])

	"""
	make up U_L, U_R, F_L, F_R based on Q_L, Q_L obtained
	"""
	U_L = np.zeros([7,N+4],float)
	U_R = np.zeros([7,N+4],float)
	F_L = np.zeros([7,N+4],float)
	F_R = np.zeros([7,N+4],float)

	U_L[0,:] = Q_L[0,:]
	U_L[1,:] = Q_L[0,:] * Q_L[1,:]
	U_L[2,:] = Q_L[0,:] * Q_L[2,:]
	U_L[3,:] = Q_L[0,:] * Q_L[3,:]
	U_L[4,:] = 1./(gamma - 1.) * Q_L[4,:] + 0.5 * Q_L[0,:] * (Q_L[1,:]*Q_L[1,:] + Q_L[2,:]*Q_L[2,:] + Q_L[3,:]*Q_L[3,:]) + 0.5 * (Bx[:]*Bx[:] + Q_L[5,:]*Q_L[5,:] + Q_L[6,:]*Q_L[6,:])
	U_L[5,:] = Q_L[5,:]
	U_L[6,:] = Q_L[6,:]

	F_L[0,:] = U_L[1,:]
	F_L[1,:] = U_L[1,:] * Q_L[1,:] + Q_L[4,:] + 0.5 * (Bx[:]*Bx[:] + Q_L[5,:]*Q_L[5,:] + Q_L[6,:]*Q_L[6,:]) - Bx[:]*Bx[:]
	F_L[2,:] = U_L[1,:] * Q_L[2,:] - Bx[:] * Q_L[5,:]
	F_L[3,:] = U_L[1,:] * Q_L[3,:] - Bx[:] * Q_L[6,:]
	F_L[4,:] = (U_L[4,:] + Q_L[4,:] + 0.5 * (Bx[:]*Bx[:] + Q_L[5,:]*Q_L[5,:] + Q_L[6,:]*Q_L[6,:])) * Q_L[1,:] - (Bx[:]*Q_L[1,:] + Q_L[5,:]*Q_L[2,:] + Q_L[6,:]*Q_L[3,:]) * Bx[:]
	F_L[5,:] = Q_L[5,:] * Q_L[1,:] - Bx[:] * Q_L[2,:]
	F_L[6,:] = Q_L[6,:] * Q_L[1,:] - Bx[:] * Q_L[3,:]

	U_R[0,:] = Q_R[0,:]
	U_R[1,:] = Q_R[0,:] * Q_R[1,:]
	U_R[2,:] = Q_R[0,:] * Q_R[2,:]
	U_R[3,:] = Q_R[0,:] * Q_R[3,:]
	U_R[4,:] = 1./(gamma - 1.) * Q_R[4,:] + 0.5 * Q_R[0,:] * (Q_R[1,:]*Q_R[1,:] + Q_R[2,:]*Q_R[2,:] + Q_R[3,:]*Q_R[3,:]) + 0.5 * (Bx[:]*Bx[:] + Q_R[5,:]*Q_R[5,:] + Q_R[6,:]*Q_R[6,:])
	U_R[5,:] = Q_R[5,:]
	U_R[6,:] = Q_R[6,:]

	F_R[0,:] = U_R[1,:]
	F_R[1,:] = U_R[1,:] * Q_R[1,:] + Q_R[4,:] + 0.5 * (Bx[:]*Bx[:] + Q_R[5,:]*Q_R[5,:] + Q_R[6,:]*Q_R[6,:]) - Bx[:]*Bx[:]
	F_R[2,:] = U_R[1,:] * Q_R[2,:] - Bx[:] * Q_R[5,:]
	F_R[3,:] = U_R[1,:] * Q_R[3,:] - Bx[:] * Q_R[6,:]
	F_R[4,:] = (U_R[4,:] + Q_R[4,:] + 0.5 * (Bx[:]*Bx[:] + Q_R[5,:]*Q_R[5,:] + Q_R[6,:]*Q_R[6,:])) * Q_R[1,:] - (Bx[:]*Q_R[1,:] + Q_R[5,:]*Q_R[2,:] + Q_R[6,:]*Q_R[3,:]) * Bx[:]
	F_R[5,:] = Q_R[5,:] * Q_R[1,:] - Bx[:] * Q_R[2,:]
	F_R[6,:] = Q_R[6,:] * Q_R[1,:] - Bx[:] * Q_R[3,:]

	#maybe need to add some more
	return c_f_L, c_f_R, Q_L, Q_R, S_L, S_R, U_L, U_R, F_L, F_R

def PLM(gamma,N,delta_x,U,Bx):
	"""
	Using piecewise linear method (PLM), 
	receive a 7*(N+4) [including 4 ghost cells] matrix U (the vector of the current conserved quantities)
		
	Bx is a 1*(N+4) matrix (x-dir magnetic field)

	receive a 7*(N+4) [including 4 ghost cells] matrix U_init (the initial vector of conserved quantities) 
	to set up Dirichlet B.C.

	delta_x is the grid spacing;
	N is the number of the cells;
	"""
	#Basic parameters###########################################
	theta = 1.1
	############################################################
	#boundary condition switch (should be set up outside PLM)
	# BC_choice = 0 #(0 for Dirichlet B.C., 1 for Neumann B.C.)

	"""
	Receive a U vector (conserved quantities), convert it 
	into a Q vector (primitive quantities), which is also a 7*(N+4) 
	[including ghost cells] vector
	"""
	Q = np.zeros([7,N+4],float)
	Q[0,:] = U[0,:]
	Q[1,:] = U[1,:] / U[0,:]
	Q[2,:] = U[2,:] / U[0,:]
	Q[3,:] = U[3,:] / U[0,:]
	Q[4,:] = (gamma - 1.) * (U[4,:] - 0.5 * Q[0,:] * (Q[1,:]*Q[1,:] + Q[2,:]*Q[2,:] + Q[3,:]*Q[3,:]) - 0.5 * (Bx[:]*Bx[:] + U[5,:]*U[5,:] + U[6,:]*U[6,:]))
	Q[5,:] = U[5,:]
	Q[6,:] = U[6,:]

	"""
	Using piecewise linear method to obtain the "left" and "right"
	interface state
	"""
	Q_L = np.zeros([7,N+4],float)
	Q_R = np.zeros([7,N+4],float)
	"""
	real interfaces are i = 1,2, ..., N+1, recall that real cells are
	i=2,3, ..., N+1

	We only use i=1,2,...,N+1 in Q_L and Q_R whose index goes from 
	0 to N+3. It means that i=0,N+2,N+3 are not be used
	"""
	for j in range(7):
		for i in range(1,N+2):
			Q_L[j,i] = Q[j,i] + 0.5 * minmod(theta * (Q[j,i] - Q[j,i-1]), 0.5 * (Q[j,i+1] - Q[j,i-1]), theta * (Q[j,i+1] - Q[j,i]))


	for j in range(7):
		for i in range(1,N+2):
			Q_R[j,i] = Q[j,i+1] - 0.5 * minmod(theta * (Q[j,i+1] - Q[j,i]), 0.5 * (Q[j,i+2] - Q[j,i]), theta * (Q[j,i+2] - Q[j,i+1]))

	"""
	Use "left" and "right" primitive quantities at each interface 
	to construct the minimum and maximum eigenvalues for 
	the "left" and "right" states for each interface.
	"""
	#First calculate fast magneto-acoustic speed c_f_L (for left state) and c_f_R (for right state)
	#two 1*(N+4) matrices
	c_f_L = np.zeros(N+4,float)
	c_f_R = np.zeros(N+4,float)

	c_f_L[1:N+2] = ((gamma * Q_L[4,1:N+2] + (Bx[1:N+2]*Bx[1:N+2] + Q_L[5,1:N+2]*Q_L[5,1:N+2] + Q_L[6,1:N+2]*Q_L[6,1:N+2]) + np.sqrt((gamma * Q_L[4,1:N+2] + (Bx[1:N+2]*Bx[1:N+2] + Q_L[5,1:N+2]*Q_L[5,1:N+2] + Q_L[6,1:N+2]*Q_L[6,1:N+2]))**2 - 4*gamma*Q_L[4,1:N+2]*Bx[1:N+2]*Bx[1:N+2]))/(2*Q_L[0,1:N+2]))**0.5
	c_f_R[1:N+2] = ((gamma * Q_R[4,1:N+2] + (Bx[1:N+2]*Bx[1:N+2] + Q_R[5,1:N+2]*Q_R[5,1:N+2] + Q_R[6,1:N+2]*Q_R[6,1:N+2]) + np.sqrt((gamma * Q_R[4,1:N+2] + (Bx[1:N+2]*Bx[1:N+2] + Q_R[5,1:N+2]*Q_R[5,1:N+2] + Q_R[6,1:N+2]*Q_R[6,1:N+2]))**2 - 4*gamma*Q_R[4,1:N+2]*Bx[1:N+2]*Bx[1:N+2]))/(2*Q_R[0,1:N+2]))**0.5

	# print(c_f_L,c_f_R)

	#make up lambda_1_L, lambda_1_R, lambda_7_L, lambda_7_R
	lambda_1_L = np.zeros(N+4,float)
	lambda_1_R = np.zeros(N+4,float)
	lambda_7_L = np.zeros(N+4,float)
	lambda_7_R = np.zeros(N+4,float)

	lambda_1_L = Q_L[1,:] - c_f_L[:]
	lambda_1_R = Q_R[1,:] - c_f_R[:]
	lambda_7_L = Q_L[1,:] + c_f_L[:]
	lambda_7_R = Q_R[1,:] + c_f_R[:]

	"""
	make up the minimum signal speed S_L (negative) and the maximum signal speed S_R (positve)
	using lambda_1_L, lambda_1_R, lambda_7_L, lambda_7_R
	S_L = min(lambda_1_L, lambda_1_R)
	S_R = max(lambda_7_L, lambda_7_R)
	"""
	S_L = np.zeros(N+4,float)
	S_R = np.zeros(N+4,float)

	for i in range(N+4):
		S_L[i] = min(lambda_1_L[i], lambda_1_R[i])
		S_R[i] = max(lambda_7_L[i], lambda_7_R[i])

	"""
	make up U_L, U_R, F_L, F_R based on Q_L, Q_L obtained
	"""
	U_L = np.zeros([7,N+4],float)
	U_R = np.zeros([7,N+4],float)
	F_L = np.zeros([7,N+4],float)
	F_R = np.zeros([7,N+4],float)

	U_L[0,:] = Q_L[0,:]
	U_L[1,:] = Q_L[0,:] * Q_L[1,:]
	U_L[2,:] = Q_L[0,:] * Q_L[2,:]
	U_L[3,:] = Q_L[0,:] * Q_L[3,:]
	U_L[4,:] = 1./(gamma - 1.) * Q_L[4,:] + 0.5 * Q_L[0,:] * (Q_L[1,:]*Q_L[1,:] + Q_L[2,:]*Q_L[2,:] + Q_L[3,:]*Q_L[3,:]) + 0.5 * (Bx[:]*Bx[:] + Q_L[5,:]*Q_L[5,:] + Q_L[6,:]*Q_L[6,:])
	U_L[5,:] = Q_L[5,:]
	U_L[6,:] = Q_L[6,:]

	F_L[0,:] = U_L[1,:]
	F_L[1,:] = U_L[1,:] * Q_L[1,:] + Q_L[4,:] + 0.5 * (Bx[:]*Bx[:] + Q_L[5,:]*Q_L[5,:] + Q_L[6,:]*Q_L[6,:]) - Bx[:]*Bx[:]
	F_L[2,:] = U_L[1,:] * Q_L[2,:] - Bx[:] * Q_L[5,:]
	F_L[3,:] = U_L[1,:] * Q_L[3,:] - Bx[:] * Q_L[6,:]
	F_L[4,:] = (U_L[4,:] + Q_L[4,:] + 0.5 * (Bx[:]*Bx[:] + Q_L[5,:]*Q_L[5,:] + Q_L[6,:]*Q_L[6,:])) * Q_L[1,:] - (Bx[:]*Q_L[1,:] + Q_L[5,:]*Q_L[2,:] + Q_L[6,:]*Q_L[3,:]) * Bx[:]
	F_L[5,:] = Q_L[5,:] * Q_L[1,:] - Bx[:] * Q_L[2,:]
	F_L[6,:] = Q_L[6,:] * Q_L[1,:] - Bx[:] * Q_L[3,:]

	U_R[0,:] = Q_R[0,:]
	U_R[1,:] = Q_R[0,:] * Q_R[1,:]
	U_R[2,:] = Q_R[0,:] * Q_R[2,:]
	U_R[3,:] = Q_R[0,:] * Q_R[3,:]
	U_R[4,:] = 1./(gamma - 1.) * Q_R[4,:] + 0.5 * Q_R[0,:] * (Q_R[1,:]*Q_R[1,:] + Q_R[2,:]*Q_R[2,:] + Q_R[3,:]*Q_R[3,:]) + 0.5 * (Bx[:]*Bx[:] + Q_R[5,:]*Q_R[5,:] + Q_R[6,:]*Q_R[6,:])
	U_R[5,:] = Q_R[5,:]
	U_R[6,:] = Q_R[6,:]

	F_R[0,:] = U_R[1,:]
	F_R[1,:] = U_R[1,:] * Q_R[1,:] + Q_R[4,:] + 0.5 * (Bx[:]*Bx[:] + Q_R[5,:]*Q_R[5,:] + Q_R[6,:]*Q_R[6,:]) - Bx[:]*Bx[:]
	F_R[2,:] = U_R[1,:] * Q_R[2,:] - Bx[:] * Q_R[5,:]
	F_R[3,:] = U_R[1,:] * Q_R[3,:] - Bx[:] * Q_R[6,:]
	F_R[4,:] = (U_R[4,:] + Q_R[4,:] + 0.5 * (Bx[:]*Bx[:] + Q_R[5,:]*Q_R[5,:] + Q_R[6,:]*Q_R[6,:])) * Q_R[1,:] - (Bx[:]*Q_R[1,:] + Q_R[5,:]*Q_R[2,:] + Q_R[6,:]*Q_R[3,:]) * Bx[:]
	F_R[5,:] = Q_R[5,:] * Q_R[1,:] - Bx[:] * Q_R[2,:]
	F_R[6,:] = Q_R[6,:] * Q_R[1,:] - Bx[:] * Q_R[3,:]

	#maybe need to add some more
	return c_f_L, c_f_R, Q_L, Q_R, S_L, S_R, U_L, U_R, F_L, F_R


def F_HLLD_generator(gamma, N, c_f_L, c_f_R, Q_L, Q_R, S_L, S_R, U_L, U_R, F_L, F_R, Bx):
	"""
	Read Q_L, Q_R, F_L, F_R (all are 7* (N+4) matrices)
	S_L and S_R (both are 1*(N+4) matrices)

	calculate Q_L_star (Q_L^*), Q_R_star (Q_R^*), Q_L_dbstar (Q_L^**), Q_R_dbstar (Q_R^**)
	and F_L_star (F_L^*), F_R_star (F_R^*), F_L_dbstar (F_L^**), F_R_dbstar (F_L^**)

	determine F_HLLD
	"""
	"""
	0. make up P_T_L and P_T_R
	"""
	P_T_L = np.zeros(N+4,float)
	P_T_R = np.zeros(N+4,float)

	P_T_L[:] = Q_L[4,:] + 0.5 * (Bx[:]*Bx[:] + Q_L[5,:]*Q_L[5,:] + Q_L[6,:]*Q_L[6,:])
	P_T_R[:] = Q_R[4,:] + 0.5 * (Bx[:]*Bx[:] + Q_R[5,:]*Q_R[5,:] + Q_R[6,:]*Q_R[6,:])
	"""
	1. calculate S_M
	"""
	S_M = np.zeros(N+4,float)

	S_M[:] = ((S_R[:] - Q_R[1,:]) * Q_R[0,:] * Q_R[1,:] - (S_L[:] - Q_L[1,:]) * Q_L[0,:] * Q_L[1,:] - P_T_R[:] + P_T_L[:] ) / ( (S_R[:] - Q_R[1,:]) * Q_R[0,:] - (S_L[:] - Q_L[1,:]) * Q_L[0,:] )

	"""
	2. calculate P_T_star
	"""
	P_T_star = np.zeros(N+4,float)

	P_T_star[:] = ( (S_R[:] - Q_R[1,:]) * Q_R[0,:] * P_T_L[:] - (S_L[:] - Q_L[1,:]) * Q_L[0,:] * P_T_R[:] + Q_L[0,:] * Q_R[0,:] * (S_R[:] - Q_R[1,:]) * (S_L[:] - Q_L[1,:]) * (Q_R[1,:] - Q_L[1,:]) ) / ( (S_R[:] - Q_R[1,:]) * Q_R[0,:] - (S_L[:] - Q_L[1,:]) * Q_L[0,:] )

	"""
	3. make up vx_L_star, vx_L_dbstar, vx_R_star, vx_R_dbstar
	make up P_T_L_star, P_T_L_dbstar, P_T_R_star, P_T_R_dbstar
	"""
	vx_L_star = np.zeros(N+4,float)
	vx_L_dbstar = np.zeros(N+4,float)
	vx_R_star = np.zeros(N+4,float)
	vx_R_dbstar = np.zeros(N+4,float)

	P_T_L_star = np.zeros(N+4,float)
	P_T_L_dbstar = np.zeros(N+4,float)
	P_T_R_star = np.zeros(N+4,float)
	P_T_R_dbstar = np.zeros(N+4,float)

	vx_L_star[:] = S_M[:]
	vx_L_dbstar[:] = S_M[:]
	vx_R_star[:] = S_M[:]
	vx_R_dbstar[:] = S_M[:]

	P_T_L_star[:] = P_T_star[:]
	P_T_L_dbstar[:] = P_T_star[:]
	P_T_R_star[:] = P_T_star[:]
	P_T_R_dbstar[:] = P_T_star[:]

	"""
	4. make up rho_L_star and rho_R_star
	"""
	rho_L_star = np.zeros(N+4,float)
	rho_R_star = np.zeros(N+4,float)

	rho_L_star[:] = Q_L[0,:] * (S_L[:] - Q_L[1,:]) / (S_L[:] - S_M[:])
	rho_R_star[:] = Q_R[0,:] * (S_R[:] - Q_R[1,:]) / (S_R[:] - S_M[:])

	"""
	5. make up vy_L_star, By_L_star,vy_R_star, By_R_star
	"""
	vy_L_star = np.zeros(N+4,float)
	vy_R_star = np.zeros(N+4,float)
	By_L_star = np.zeros(N+4,float)
	By_R_star = np.zeros(N+4,float)

	vy_L_star[:] = Q_L[2,:] - Bx[:] * Q_L[5,:] * (S_M[:] - Q_L[1,:]) / ( Q_L[0,:] * (S_L[:] - Q_L[1,:]) * (S_L[:] - S_M[:]) - Bx[:]*Bx[:] )
	vy_R_star[:] = Q_R[2,:] - Bx[:] * Q_R[5,:] * (S_M[:] - Q_R[1,:]) / ( Q_R[0,:] * (S_R[:] - Q_R[1,:]) * (S_R[:] - S_M[:]) - Bx[:]*Bx[:] )

	By_L_star[:] = Q_L[5,:] * ( Q_L[0,:] * (S_L[:] - Q_L[1,:]) * (S_L[:] - Q_L[1,:]) - Bx[:] * Bx[:] ) / ( Q_L[0,:] * (S_L[:] - Q_L[1,:]) * (S_L[:] - S_M[:]) - Bx[:]*Bx[:] )
	By_R_star[:] = Q_R[5,:] * ( Q_R[0,:] * (S_R[:] - Q_R[1,:]) * (S_R[:] - Q_R[1,:]) - Bx[:] * Bx[:] ) / ( Q_R[0,:] * (S_R[:] - Q_R[1,:]) * (S_R[:] - S_M[:]) - Bx[:]*Bx[:] )

	"""
	6. make up vz_L_star, Bz_L_star,vz_R_star, Bz_R_star
	"""	
	vz_L_star = np.zeros(N+4,float)
	vz_R_star = np.zeros(N+4,float)
	Bz_L_star = np.zeros(N+4,float)
	Bz_R_star = np.zeros(N+4,float)

	vz_L_star[:] = Q_L[3,:] - Bx[:] * Q_L[6,:] * (S_M[:] - Q_L[1,:]) / ( Q_L[0,:] * (S_L[:] - Q_L[1,:]) * (S_L[:] - S_M[:]) - Bx[:]*Bx[:] )
	vz_R_star[:] = Q_R[3,:] - Bx[:] * Q_R[6,:] * (S_M[:] - Q_R[1,:]) / ( Q_R[0,:] * (S_R[:] - Q_R[1,:]) * (S_R[:] - S_M[:]) - Bx[:]*Bx[:] )

	Bz_L_star[:] = Q_L[6,:] * ( Q_L[0,:] * (S_L[:] - Q_L[1,:]) * (S_L[:] - Q_L[1,:]) - Bx[:] * Bx[:] ) / ( Q_L[0,:] * (S_L[:] - Q_L[1,:]) * (S_L[:] - S_M[:]) - Bx[:]*Bx[:] )
	Bz_R_star[:] = Q_R[6,:] * ( Q_R[0,:] * (S_R[:] - Q_R[1,:]) * (S_R[:] - Q_R[1,:]) - Bx[:] * Bx[:] ) / ( Q_R[0,:] * (S_R[:] - Q_R[1,:]) * (S_R[:] - S_M[:]) - Bx[:]*Bx[:] )

	#check (44)-(47)
	# for i in range(1,N+2):
	# 	if abs(S_M[i] - Q_L[1,i])<1e-6 or abs(S_L[i] - (Q_L[1,i]+c_f_L[i]))<1e-6 or abs(S_L[i] - (Q_L[1,i]-c_f_L[i]))<1e-6 or ( abs(Q_L[5,i] - Q_L[6,i])<1e-6 and abs(Q_L[6,i])<1e-6) or Bx[i]**2 - gamma * Q_L[4,i]>1e-6:
	# 		print(vy_L_star[i],Q_L[2,i])
	# 		print(vz_L_star[i],Q_L[3,i])

	"""
	7. make up E_L_star and E_R_star
	"""
	E_L_star = np.zeros(N+4,float)
	E_R_star = np.zeros(N+4,float)

	E_L_star[:] = ( (S_L[:] - Q_L[1,:]) * U_L[4,:] - P_T_L[:] * Q_L[1,:] + P_T_star[:] * S_M[:] + Bx[:] * ( (Q_L[1,:]*Bx[:] + Q_L[2,:]*Q_L[5,:] + Q_L[3,:]*Q_L[6,:]) - (vx_L_star[:]*Bx[:] + vy_L_star[:]*By_L_star[:] + vz_L_star[:]*Bz_L_star[:]) ) ) / (S_L[:] - S_M[:])
	E_R_star[:] = ( (S_R[:] - Q_R[1,:]) * U_R[4,:] - P_T_R[:] * Q_R[1,:] + P_T_star[:] * S_M[:] + Bx[:] * ( (Q_R[1,:]*Bx[:] + Q_R[2,:]*Q_R[5,:] + Q_R[3,:]*Q_R[6,:]) - (vx_R_star[:]*Bx[:] + vy_R_star[:]*By_R_star[:] + vz_R_star[:]*Bz_R_star[:]) ) ) / (S_R[:] - S_M[:])

	"""
	8. make up rho_L_dbstar, rho_R_dbstar
	"""
	rho_L_dbstar = np.zeros(N+4,float)
	rho_R_dbstar = np.zeros(N+4,float)

	rho_L_dbstar[:] = rho_L_star[:]
	rho_R_dbstar[:] = rho_R_star[:]

	"""
	9. make up S_L_star, S_R_star
	"""	
	S_L_star = np.zeros(N+4,float)
	S_R_star = np.zeros(N+4,float)

	S_L_star[:] = S_M[:] - abs(Bx[:]) / np.sqrt(rho_L_star[:])
	S_R_star[:] = S_M[:] + abs(Bx[:]) / np.sqrt(rho_R_star[:])

	"""
	10. make up vy_L_dbstar,vy_R_dbstar, By_L_dbstar, By_R_dbstar
	"""
	vy_L_dbstar = np.zeros(N+4,float)
	vy_R_dbstar = np.zeros(N+4,float)
	By_L_dbstar = np.zeros(N+4,float)
	By_R_dbstar = np.zeros(N+4,float)

	vy_L_dbstar[:] = ( np.sqrt(rho_L_star[:]) * vy_L_star[:] + np.sqrt(rho_R_star[:]) * vy_R_star[:] + (By_R_star[:] - By_L_star[:]) * np.sign(Bx[:]) ) / ( np.sqrt(rho_L_star[:]) + np.sqrt(rho_R_star[:]) )
	vy_R_dbstar[:] = vy_L_dbstar[:]

	By_L_dbstar[:] = ( np.sqrt(rho_L_star[:]) * By_R_star[:] + np.sqrt(rho_R_star[:]) * By_L_star[:] + np.sqrt(rho_L_star[:] * rho_R_star[:]) * (vy_R_star[:] - vy_L_star[:]) * np.sign(Bx[:]) ) / ( np.sqrt(rho_L_star[:]) + np.sqrt(rho_R_star[:]) )
	By_R_dbstar[:] = By_L_dbstar[:]

	"""
	11. make up vz_L_dbstar, vz_R_dbstar, Bz_L_dbstar, Bz_R_dbstar
	"""
	vz_L_dbstar = np.zeros(N+4,float)
	vz_R_dbstar = np.zeros(N+4,float)
	Bz_L_dbstar = np.zeros(N+4,float)
	Bz_R_dbstar = np.zeros(N+4,float)

	vz_L_dbstar[:] = ( np.sqrt(rho_L_star[:]) * vz_L_star[:] + np.sqrt(rho_R_star[:]) * vz_R_star[:] + (Bz_R_star[:] - Bz_L_star[:]) * np.sign(Bx[:]) ) / ( np.sqrt(rho_L_star[:]) + np.sqrt(rho_R_star[:]) )
	vz_R_dbstar[:] = vz_L_dbstar[:]

	Bz_L_dbstar[:] = ( np.sqrt(rho_L_star[:]) * Bz_R_star[:] + np.sqrt(rho_R_star[:]) * Bz_L_star[:] + np.sqrt(rho_L_star[:] * rho_R_star[:]) * (vz_R_star[:] - vz_L_star[:]) * np.sign(Bx[:]) ) / ( np.sqrt(rho_L_star[:]) + np.sqrt(rho_R_star[:]) )
	Bz_R_dbstar[:] = Bz_L_dbstar[:]

	"""
	12. make up E_L_dbstar, E_R_dbstar
	"""
	E_L_dbstar = np.zeros(N+4,float)
	E_R_dbstar = np.zeros(N+4,float)

	E_L_dbstar[:] = E_L_star[:] - np.sqrt(rho_L_star[:]) * ( (vx_L_star[:]*Bx[:] + vy_L_star[:]*By_L_star[:] + vz_L_star[:]*Bz_L_star[:]) - (vx_L_dbstar[:]*Bx[:] + vy_L_dbstar[:]*By_L_dbstar[:] + vz_L_dbstar[:]*Bz_L_dbstar[:]) ) * np.sign(Bx[:])
	E_R_dbstar[:] = E_R_star[:] + np.sqrt(rho_R_star[:]) * ( (vx_R_star[:]*Bx[:] + vy_R_star[:]*By_R_star[:] + vz_R_star[:]*Bz_R_star[:]) - (vx_R_dbstar[:]*Bx[:] + vy_R_dbstar[:]*By_R_dbstar[:] + vz_R_dbstar[:]*Bz_R_dbstar[:]) ) * np.sign(Bx[:])
	
	"""
	13. make up F_L_star, F_R_star, F_L_dbstar, F_R_dbstar
	"""
	F_L_star = np.zeros([7,N+4],float)
	F_R_star = np.zeros([7,N+4],float)
	F_L_dbstar = np.zeros([7,N+4],float)
	F_R_dbstar = np.zeros([7,N+4],float)

	F_L_star[0,:] = rho_L_star[:] * vx_L_star[:]
	F_L_star[1,:] = rho_L_star[:] * vx_L_star[:] * vx_L_star[:] + P_T_L_star[:] - Bx[:]*Bx[:]
	F_L_star[2,:] = rho_L_star[:] * vx_L_star[:] * vy_L_star[:] - Bx[:] * By_L_star[:]
	F_L_star[3,:] = rho_L_star[:] * vx_L_star[:] * vz_L_star[:] - Bx[:] * Bz_L_star[:]
	F_L_star[4,:] = (E_L_star[:] + P_T_L_star[:]) * vx_L_star[:] - (Bx[:]*vx_L_star[:] + By_L_star[:]*vy_L_star[:] + Bz_L_star[:]*vz_L_star[:]) * Bx[:]
	F_L_star[5,:] = By_L_star[:] * vx_L_star[:] - Bx[:] * vy_L_star[:]
	F_L_star[6,:] = Bz_L_star[:] * vx_L_star[:] - Bx[:] * vz_L_star[:]

	F_R_star[0,:] = rho_R_star[:] * vx_R_star[:]
	F_R_star[1,:] = rho_R_star[:] * vx_R_star[:] * vx_R_star[:] + P_T_R_star[:] - Bx[:]*Bx[:]
	F_R_star[2,:] = rho_R_star[:] * vx_R_star[:] * vy_R_star[:] - Bx[:] * By_R_star[:]
	F_R_star[3,:] = rho_R_star[:] * vx_R_star[:] * vz_R_star[:] - Bx[:] * Bz_R_star[:]
	F_R_star[4,:] = (E_R_star[:] + P_T_R_star[:]) * vx_R_star[:] - (Bx[:]*vx_R_star[:] + By_R_star[:]*vy_R_star[:] + Bz_R_star[:]*vz_R_star[:]) * Bx[:]
	F_R_star[5,:] = By_R_star[:] * vx_R_star[:] - Bx[:] * vy_R_star[:]
	F_R_star[6,:] = Bz_R_star[:] * vx_R_star[:] - Bx[:] * vz_R_star[:]

	F_L_dbstar[0,:] = rho_L_dbstar[:] * vx_L_dbstar[:]
	F_L_dbstar[1,:] = rho_L_dbstar[:] * vx_L_dbstar[:] * vx_L_dbstar[:] + P_T_L_dbstar[:] - Bx[:]*Bx[:]
	F_L_dbstar[2,:] = rho_L_dbstar[:] * vx_L_dbstar[:] * vy_L_dbstar[:] - Bx[:] * By_L_dbstar[:]
	F_L_dbstar[3,:] = rho_L_dbstar[:] * vx_L_dbstar[:] * vz_L_dbstar[:] - Bx[:] * Bz_L_dbstar[:]
	F_L_dbstar[4,:] = (E_L_dbstar[:] + P_T_L_dbstar[:]) * vx_L_dbstar[:] - (Bx[:]*vx_L_dbstar[:] + By_L_dbstar[:]*vy_L_dbstar[:] + Bz_L_dbstar[:]*vz_L_dbstar[:]) * Bx[:]
	F_L_dbstar[5,:] = By_L_dbstar[:] * vx_L_dbstar[:] - Bx[:] * vy_L_dbstar[:]
	F_L_dbstar[6,:] = Bz_L_dbstar[:] * vx_L_dbstar[:] - Bx[:] * vz_L_dbstar[:]

	F_R_dbstar[0,:] = rho_R_dbstar[:] * vx_R_dbstar[:]
	F_R_dbstar[1,:] = rho_R_dbstar[:] * vx_R_dbstar[:] * vx_R_dbstar[:] + P_T_R_dbstar[:] - Bx[:]*Bx[:]
	F_R_dbstar[2,:] = rho_R_dbstar[:] * vx_R_dbstar[:] * vy_R_dbstar[:] - Bx[:] * By_R_dbstar[:]
	F_R_dbstar[3,:] = rho_R_dbstar[:] * vx_R_dbstar[:] * vz_R_dbstar[:] - Bx[:] * Bz_R_dbstar[:]
	F_R_dbstar[4,:] = (E_R_dbstar[:] + P_T_R_dbstar[:]) * vx_R_dbstar[:] - (Bx[:]*vx_R_dbstar[:] + By_R_dbstar[:]*vy_R_dbstar[:] + Bz_R_dbstar[:]*vz_R_dbstar[:]) * Bx[:]
	F_R_dbstar[5,:] = By_R_dbstar[:] * vx_R_dbstar[:] - Bx[:] * vy_R_dbstar[:]
	F_R_dbstar[6,:] = Bz_R_dbstar[:] * vx_R_dbstar[:] - Bx[:] * vz_R_dbstar[:]


	"""
	14. choose F_HLLD from F_L, F_R, F_L_star, F_R_star, F_L_dbstar, F_R_dbstar
	based on the controlment condition for EACH interface
	"""
	F_HLLD = np.zeros([7,N+4],float)

	#only care about real interfaces i=1,2,...,N+1
	for i in range(1,N+2):
		if S_L[i] - 0. > 1e-10:
			#S_L>0, F_HLLD = F_L
			F_HLLD[:,i] = F_L[:,i]
		elif S_L_star[i] - 0. > 1e-10:
			#S_L<=0<=S_L_star, F_HLLD = F_L_star
			F_HLLD[:,i] = F_L_star[:,i]
		elif S_M[i] - 0. > 1e-10:
			#S_L_star<=0<=S_M, F_HLLD = F_L_dbstar
			F_HLLD[:,i] = F_L_dbstar[:,i]
		elif S_R_star[i] - 0. > 1e-10:
			#S_M<=0<=S_R_star, F_HLLD = F_R_dbstar
			F_HLLD[:,i] = F_R_dbstar[:,i]
		elif S_R[i] - 0. > 1e-10:
			#S_R_star<=0<=S_R, F_HLLD = F_R_star
			F_HLLD[:,i] = F_R_star[:,i]
		else:
			F_HLLD[:,i] = F_R[:,i]

	return F_HLLD


def F_HLL_generator(gamma, N, c_f_L, c_f_R, Q_L, Q_R, S_L, S_R, U_L, U_R, F_L, F_R, Bx):
	"""
	Read Q_L, Q_R, F_L, F_R (all are 7* (N+4) matrices)
	S_L and S_R (both are 1*(N+4) matrices)

	calculate Q_L_star (Q_L^*), Q_R_star (Q_R^*), Q_L_dbstar (Q_L^**), Q_R_dbstar (Q_R^**)
	and F_L_star (F_L^*), F_R_star (F_R^*), F_L_dbstar (F_L^**), F_R_dbstar (F_L^**)

	determine F_HLL
	"""
	F_star = np.zeros([7,N+4],float)
	F_HLL = np.zeros([7,N+4],float)

	F_star[0,:] = ( S_R[:] * F_L[0,:] - S_L[:] * F_R[0,:] + S_R[:] * S_L[:] * (U_R[0,:] - U_L[0,:]) ) /(S_R[:] - S_L[:])
	F_star[1,:] = ( S_R[:] * F_L[1,:] - S_L[:] * F_R[1,:] + S_R[:] * S_L[:] * (U_R[1,:] - U_L[1,:]) ) /(S_R[:] - S_L[:])
	F_star[2,:] = ( S_R[:] * F_L[2,:] - S_L[:] * F_R[2,:] + S_R[:] * S_L[:] * (U_R[2,:] - U_L[2,:]) ) /(S_R[:] - S_L[:])
	F_star[3,:] = ( S_R[:] * F_L[3,:] - S_L[:] * F_R[3,:] + S_R[:] * S_L[:] * (U_R[3,:] - U_L[3,:]) ) /(S_R[:] - S_L[:])
	F_star[4,:] = ( S_R[:] * F_L[4,:] - S_L[:] * F_R[4,:] + S_R[:] * S_L[:] * (U_R[4,:] - U_L[4,:]) ) /(S_R[:] - S_L[:])
	F_star[5,:] = ( S_R[:] * F_L[5,:] - S_L[:] * F_R[5,:] + S_R[:] * S_L[:] * (U_R[5,:] - U_L[5,:]) ) /(S_R[:] - S_L[:])
	F_star[6,:] = ( S_R[:] * F_L[6,:] - S_L[:] * F_R[6,:] + S_R[:] * S_L[:] * (U_R[6,:] - U_L[6,:]) ) /(S_R[:] - S_L[:])

	#only care about real interfaces i=1,2,...,N+1
	for i in range(1,N+2):
		if S_L[i] - 0. > 1e-10:
			#S_L>0, F_HLL = F_L
			F_HLL[:,i] = F_L[:,i]
		elif S_R[i] - 0. > 1e-10:
			#S_L<=0<=S_R, F_HLL = F_star
			F_HLL[:,i] = F_star[:,i]
		else:
			#S_R<0, F_HLL = F_R
			F_HLL[:,i] = F_R[:,i]
	
	return F_HLL

def Lu_PLM_HLLD(gamma,N,delta_x,U,Bx):
	"""
	Get L(u) = -(F_rightinterface - F_leftintefece)/delta_x
	only update REAL cells, i.e. i=2,3,..., N+1
	"""
	Lu_HLLD = np.zeros([7,N+4],float)

	c_f_L = np.zeros(N+4,float)
	c_f_R = np.zeros(N+4,float)
	Q_L = np.zeros([7,N+4],float)
	Q_R = np.zeros([7,N+4],float) 
	S_L = np.zeros(N+4,float)
	S_R = np.zeros(N+4,float)
	U_L = np.zeros([7,N+4],float) 
	U_R = np.zeros([7,N+4],float) 
	F_L = np.zeros([7,N+4],float) 
	F_R = np.zeros([7,N+4],float)

	c_f_L, c_f_R, Q_L, Q_R, S_L, S_R, U_L, U_R, F_L, F_R = PLM(gamma,N,delta_x,U,Bx)

	F_HLLD = np.zeros([7,N+4],float)
	F_HLLD = F_HLLD_generator(gamma, N, c_f_L, c_f_R, Q_L, Q_R, S_L, S_R, U_L, U_R, F_L, F_R, Bx)

	Lu_HLLD[:,2:N+2] = - (F_HLLD[:,2:N+2] - F_HLLD[:,1:N+1]) / delta_x 

	return Lu_HLLD

def Lu_PLM_HLL(gamma,N,delta_x,U,Bx):
	"""
	Get L(u) = -(F_rightinterface - F_leftintefece)/delta_x
	only update REAL cells, i.e. i=2,3,..., N+1
	"""
	Lu_HLL = np.zeros([7,N+4],float)

	c_f_L = np.zeros(N+4,float)
	c_f_R = np.zeros(N+4,float)
	Q_L = np.zeros([7,N+4],float)
	Q_R = np.zeros([7,N+4],float) 
	S_L = np.zeros(N+4,float)
	S_R = np.zeros(N+4,float)
	U_L = np.zeros([7,N+4],float) 
	U_R = np.zeros([7,N+4],float) 
	F_L = np.zeros([7,N+4],float) 
	F_R = np.zeros([7,N+4],float)

	c_f_L, c_f_R, Q_L, Q_R, S_L, S_R, U_L, U_R, F_L, F_R = PLM(gamma,N,delta_x,U,Bx)

	F_HLL = np.zeros([7,N+4],float)
	F_HLL = F_HLL_generator(gamma, N, c_f_L, c_f_R, Q_L, Q_R, S_L, S_R, U_L, U_R, F_L, F_R, Bx)

	Lu_HLL[:,2:N+2] = - (F_HLL[:,2:N+2] - F_HLL[:,1:N+1]) / delta_x 

	return Lu_HLL




