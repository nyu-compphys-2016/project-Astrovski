"""
Project: MHD_1d_main.py
Description: 1D MHD code with several tests
	1. 
Author: Xuyao Hu
Date: Dec. 11, 2016
Version: 1.0
"""
import numpy as np
import matplotlib.pyplot as plt
import MHD_1d_Solver_Scheme as MHD1dSS

print("Which kind of test do you want to run?")
print("For Dai & Woodward test, input 'DW'.")
print("For Brio & Wu test, input 'BW'.")
print("For Alfven Waves test, input 'AW'.")
print("For Fast Switch-on shock test, input 'FS'.")
print("For Slow Switch-off shock test, input 'SS'.")
print("For Fast Rarefaction waves test, input 'FR'.")
print("For Slow Rarefaction waves test, input 'SR'.")


Test_type = input("Input the type of the test: ")

if Test_type == "DW":
	"""
	Dai & Woodward test********************************************************
	"""
	print("-------------------Dai & Woodward (DW) shock test------------------------")
	"""
	Grid (cell) initilization
	"""
	x_min = -0.5
	x_max = 0.5
	N = 800 			#N = 800				#number of REAL cells
	delta_x = (x_max - x_min) / N 	#width of each cell
	########################################################

	"""
	Basic parameters and constants
	"""
	gamma = 5./3.
	CFL_num = 0.8
	#######################################################
	"""
	Time initialization
	"""
	t_init = 0.		#initial time
	t_final = 0.201	#final time

	t_plot = 0.2	#time to produce the plot
	#########################################################
	"""
	Initial conditions list
	"""
	t = t_init
	#left half region
	rho_L = 1.08
	P_L = 0.95
	vx_L = 1.2
	vy_L = 0.01
	vz_L = 0.5

	Bx_const = 4./np.sqrt(4.*np.pi)

	By_L = 3.6/np.sqrt(4.*np.pi)
	Bz_L = 2./np.sqrt(4.*np.pi)

	#right half region
	rho_R = 1.
	P_R = 1.
	vx_R = 0.
	vy_R = 0.
	vz_R = 0.

	Bx_const = 4./np.sqrt(4.*np.pi)

	By_R = 4./np.sqrt(4.*np.pi)
	Bz_R = 2./np.sqrt(4.*np.pi)
	#############################################################
	"""
	Vector initialization and initial condition setting
	"""
	rho = np.zeros(N+4,float)			#rho is the array of the density
	vx = np.zeros(N+4,float)			#vx is the array of the x-comp velocity
	vy = np.zeros(N+4,float)			#vy is the array of the y-comp velocity
	vz = np.zeros(N+4,float)			#vz is the array of the z-comp velocity
	P = np.zeros(N+4,float)				#P is the array of the pressure
	Bx = np.zeros(N+4,float)			#Bx is the array of x-comp of magnetic field
	By = np.zeros(N+4,float)			#By is the array of y-comp of magnetic field
	Bz = np.zeros(N+4,float)			#Bz is the array of z-comp of magnetic field
	E = np.empty(N+4,float)				#E is the array of the Energy density

	rho_hll = np.zeros(N+4,float)			#rho is the array of the density
	vx_hll = np.zeros(N+4,float)			#vx is the array of the x-comp velocity
	vy_hll = np.zeros(N+4,float)			#vy is the array of the y-comp velocity
	vz_hll = np.zeros(N+4,float)			#vz is the array of the z-comp velocity
	P_hll = np.zeros(N+4,float)				#P is the array of the pressure
	Bx_hll = np.zeros(N+4,float)			#Bx is the array of x-comp of magnetic field
	By_hll = np.zeros(N+4,float)			#By is the array of y-comp of magnetic field
	Bz_hll = np.zeros(N+4,float)			#Bz is the array of z-comp of magnetic field
	E_hll = np.empty(N+4,float)				#E is the array of the Energy density
	
	#initial values
	rho_iv = np.zeros(N+4,float)
	vx_iv = np.zeros(N+4,float)
	vy_iv = np.zeros(N+4,float)
	vz_iv = np.zeros(N+4,float)
	P_iv = np.zeros(N+4,float)
	Bx_iv = np.zeros(N+4,float)
	By_iv = np.zeros(N+4,float)
	Bz_iv = np.zeros(N+4,float)
	E_iv = np.zeros(N+4,float)
	#initial condtion setting for real cells
	rho[2 : N//2+2] = rho_L
	rho[N//2+2 : N+2] = rho_R

	vx[2 : N//2+2] = vx_L
	vx[N//2+2 : N+2] = vx_R
	vy[2 : N//2+2] = vy_L
	vy[N//2+2 : N+2] = vy_R
	vz[2 : N//2+2] = vz_L
	vz[N//2+2 : N+2] = vz_R

	P[2 : N//2+2] = P_L
	P[N//2+2 : N+2] = P_R

	Bx[:] = Bx_const

	By[2 : N//2+2] = By_L
	By[N//2+2 : N+2] = By_R
	Bz[2 : N//2+2] = Bz_L
	Bz[N//2+2 : N+2] = Bz_R

	E[2 : N+2] = P[2 : N+2]/(gamma - 1.) + 0.5 * rho[2 : N+2] * (vx[2 : N+2]*vx[2 : N+2] + vy[2 : N+2]*vy[2 : N+2] + vz[2 : N+2]*vz[2 : N+2]) + 0.5 * (Bx[2 : N+2]*Bx[2 : N+2] + By[2 : N+2]*By[2 : N+2] + Bz[2 : N+2]*Bz[2 : N+2])

	#initial condition setting for ghost cells
	rho[0:2] = rho[2]
	rho[-2:] = rho[N+1]

	vx[0:2] = vx[2]
	vx[-2:] = vx[N+1]

	vy[0:2] = vy[2]
	vy[-2:] = vy[N+1]

	vz[0:2] = vz[2]
	vz[-2:] = vz[N+1]

	P[0:2] = P[2]
	P[-2:] = P[N+1]

	By[0:2] = By[2]
	By[-2:] = By[N+1]

	Bz[0:2] = Bz[2]
	Bz[-2:] = Bz[N+1]

	E[0:2] = E[2]
	E[-2:] = E[N+1]

	rho_iv[:] = rho[:]
	vx_iv[:] = vx[:]
	vy_iv[:] = vy[:]
	vz_iv[:] = vz[:]
	P_iv[:] = P[:]
	Bx_iv[:] = Bx[:]
	By_iv[:] = By[:]
	Bz_iv[:] = Bz[:]
	E_iv[:] = E[:]
	################################################################

	"""
	Convert all INITIAL physical quantities into the vector of conserved variables U
	U is a 7 * (N+4) matrix
	"""
	U = np.zeros([7,N+4],float)			# HLLD data
	U_hll = np.zeros([7,N+4],float)	 	# HLL data

	U[0,:] = rho[:]
	U[1,:] = rho[:] * vx[:]
	U[2,:] = rho[:] * vy[:]
	U[3,:] = rho[:] * vz[:]
	U[4,:] = E[:]
	U[5,:] = By[:]
	U[6,:] = Bz[:]

	#just for INITIAL conditions can we use these relations
	#otherwise, we should have things like U_hll[0,:] = rho_hll[:]
	U_hll[0,:] = rho[:]
	U_hll[1,:] = rho[:] * vx[:]
	U_hll[2,:] = rho[:] * vy[:]
	U_hll[3,:] = rho[:] * vz[:]
	U_hll[4,:] = E[:]
	U_hll[5,:] = By[:]
	U_hll[6,:] = Bz[:]

	"""
	make a copy of U for applying Dirichlet B.C. 
	"""
	U_init = np.zeros([7,N+4],float)
	U_init = np.copy(U)

	"""
	Temporary U for updating U using 3rd Runge-Kutta scheme (Shu & Osher)
	"""
	U_temp_1 = np.zeros([7,N+4],float)
	U_temp_2 = np.zeros([7,N+4],float)

	U_hll_temp_1 = np.zeros([7,N+4],float)
	U_hll_temp_2 = np.zeros([7,N+4],float)
	##################################################################

	"""
	Choice of the type of boundary condtion:
	0. Dirichlet
	1. Neumann
	"""
	Boundary_type = float(input("Please choose the type of boundary condition (0 for Dirichlet B.C., 1 for Neumann B.C.):"))
	if Boundary_type == 0:
		print("The Boundary type is Dirichlet B.C.")
	elif Boundary_type == 1:
		print("The Boundary type is Neumann B.C.")
	else:
		print("Please choose '0' or '1' or '2'!")

	while t<t_final:
		delta_t = MHD1dSS.GetDt(gamma,N,delta_x,CFL_num,U,Bx)
		delta_t_hll = MHD1dSS.GetDt(gamma,N,delta_x,CFL_num,U_hll,Bx)
		#REAL cells are updated using Lu and 3rd-order Runge-Kutta
		#GHOST cells are updated using boundary condition (Dirichlet or Neumann)
		U_temp_1 = U + delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U,Bx)
		U_temp_2 = 3./4. * U + 1./4. * U_temp_1 + 1./4. * delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U_temp_1,Bx)
		U = 1./3. * U + 2./3. * U_temp_2 + 2./3. * delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U_temp_2,Bx) 

		U_hll_temp_1 = U_hll + delta_t_hll * MHD1dSS.Lu_PLM_HLL(gamma,N,delta_x,U_hll,Bx)
		U_hll_temp_2 = 3./4. * U_hll + 1./4. * U_hll_temp_1 + 1./4. * delta_t_hll * MHD1dSS.Lu_PLM_HLL(gamma,N,delta_x,U_hll_temp_1,Bx)
		U_hll = 1./3. * U_hll + 2./3. * U_hll_temp_2 + 2./3. * delta_t_hll * MHD1dSS.Lu_PLM_HLL(gamma,N,delta_x,U_hll_temp_2,Bx) 
		#adjust ghost cells
		if Boundary_type == 0:
			#Dirichlet
			U[:,0] = U_init[:,0]
			U[:,1] = U_init[:,1]
			U[:,-2] = U_init[:,-2]
			U[:,-1] = U_init[:,-1]

			U_hll[:,0] = U_init[:,0]
			U_hll[:,1] = U_init[:,1]
			U_hll[:,-2] = U_init[:,-2]
			U_hll[:,-1] = U_init[:,-1]
		elif Boundary_type == 1:
			#Neumann
			U[:,0] = U[:,2]
			U[:,1] = U[:,2]
			U[:,-2] = U[:,N+1]
			U[:,-1] = U[:,N+1]

			U_hll[:,0] = U_hll[:,2]
			U_hll[:,1] = U_hll[:,2]
			U_hll[:,-2] = U_hll[:,N+1]
			U_hll[:,-1] = U_hll[:,N+1]

		t += delta_t

		if abs(t - t_plot) < 0.5*delta_t:
			print("Plot at time t = ", t)
			rho[:] = U[0,:]
			vx[:] = U[1,:] / U[0,:]
			vy[:] = U[2,:] / U[0,:]
			vz[:] = U[3,:] / U[0,:]
			E[:] = U[4,:]
			By[:] = U[5,:]
			Bz[:] = U[6,:]
			P[:] = (gamma - 1.) * (E[:] - 0.5 * rho[:] * (vx[:]*vx[:] + vy[:]*vy[:] + vz[:]*vz[:]) - 0.5 * (Bx[:]*Bx[:] + By[:]*By[:] + Bz[:]*Bz[:]) )
			
			rho_hll[:] = U_hll[0,:]
			vx_hll[:] = U_hll[1,:] / U_hll[0,:]
			vy_hll[:] = U_hll[2,:] / U_hll[0,:]
			vz_hll[:] = U_hll[3,:] / U_hll[0,:]
			E_hll[:] = U_hll[4,:]
			By_hll[:] = U_hll[5,:]
			Bz_hll[:] = U_hll[6,:]
			P_hll[:] = (gamma - 1.) * (E_hll[:] - 0.5 * rho_hll[:] * (vx_hll[:]*vx_hll[:] + vy_hll[:]*vy_hll[:] + vz_hll[:]*vz_hll[:]) - 0.5 * (Bx[:]*Bx[:] + By_hll[:]*By_hll[:] + Bz_hll[:]*Bz_hll[:]) )

			x_plot = np.empty(N,float)
			for i in range(N):
				x_plot[i] = x_min + (i+0.5) * delta_x

			# add plot
			fig0 = plt.figure()
			ax0=fig0.add_subplot(2,2,1)
			ax0.plot(x_plot[:],rho[2:N+2],color="orange",linewidth=2,label="HLLD (t = {:.2f})".format(t))	#label="HLLD (t={1:03d})".format(t)
			ax0.plot(x_plot[:],rho_hll[2:N+2],"g--",linewidth=2,label="HLL (t = {:.2f})".format(t))
			ax0.plot(x_plot[:],rho_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax0.set_xlabel(r"$x$",fontsize=24)
			ax0.set_ylabel(r"$\rho$",fontsize=24)
			ax0.set_xlim([x_min,x_max])
			ax0.set_ylim([min(rho[2:N+2])-0.1, max(rho[2:N+2])+0.1])
			ax0.legend(loc="upper left")

			ax1=fig0.add_subplot(2,2,2)
			ax1.plot(x_plot[:],P[2:N+2],color="orange",linewidth=2,label="HLLD (t = {:.2f})".format(t))
			ax1.plot(x_plot[:],P_hll[2:N+2],"g--",linewidth=2,label="HLL (t = {:.2f})".format(t))
			ax1.plot(x_plot[:],P_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax1.set_xlabel(r"$x$",fontsize=24)
			ax1.set_ylabel(r"$P$",fontsize=24)
			ax1.set_xlim([x_min,x_max])
			ax1.set_ylim([min(P[2:N+2])-0.1, max(P[2:N+2]+0.1)])
			ax1.legend(loc="upper left")

			ax2=fig0.add_subplot(2,2,3)
			ax2.plot(x_plot[:],vx[2:N+2],color="orange",linewidth=2,label="HLLD (t = {:.2f})".format(t))
			ax2.plot(x_plot[:],vx_hll[2:N+2],"g--",linewidth=2,label="HLL (t = {:.2f})".format(t))
			ax2.plot(x_plot[:],vx_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax2.set_xlabel(r"$x$",fontsize=24)
			ax2.set_ylabel(r"$v_x$",fontsize=24)
			ax2.set_xlim([x_min,x_max])
			ax2.set_ylim([min(vx[2:N+2])-0.1, max(vx[2:N+2])+0.1])
			ax2.legend()

			ax3=fig0.add_subplot(2,2,4)
			ax3.plot(x_plot[:],Bx[2:N+2],color="orange",linewidth=2,label="HLLD (t = {:.2f})".format(t))
			ax3.plot(x_plot[:],Bx_hll[2:N+2],"g--",linewidth=2,label="HLL (t = {:.2f})".format(t))
			ax3.plot(x_plot[:],Bx_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax3.set_xlabel(r"$x$",fontsize=24)
			ax3.set_ylabel(r"$B_x$",fontsize=24)
			ax3.set_xlim([x_min,x_max])
			ax3.set_ylim([min(Bx[2:N+2])-0.1, max(Bx[2:N+2])+0.1])
			ax3.legend()

			fig1 = plt.figure()
			ax4=fig1.add_subplot(2,2,1)
			ax4.plot(x_plot[:],vy[2:N+2],color="orange",linewidth=2,label="HLLD (t = {:.2f})".format(t))
			ax4.plot(x_plot[:],vy_hll[2:N+2],"g--",linewidth=2,label="HLL (t = {:.2f})".format(t))
			ax4.plot(x_plot[:],vy_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax4.set_xlabel(r"$x$",fontsize=24)
			ax4.set_ylabel(r"$v_y$",fontsize=24)
			ax4.set_xlim([x_min,x_max])
			ax4.set_ylim([min(vy[2:N+2])-0.2, max(vy[2:N+2])+0.2])
			ax4.legend()

			ax5=fig1.add_subplot(2,2,2)
			ax5.plot(x_plot[:],By[2:N+2],color="orange",linewidth=2,label="HLLD (t = {:.2f})".format(t))
			ax5.plot(x_plot[:],By_hll[2:N+2],"g--",linewidth=2,label="HLL (t = {:.2f})".format(t))
			ax5.plot(x_plot[:],By_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax5.set_xlabel(r"$x$",fontsize=24)
			ax5.set_ylabel(r"$B_y$",fontsize=24)
			ax5.set_xlim([x_min,x_max])
			ax5.legend(loc="upper left")

			ax6=fig1.add_subplot(2,2,3)
			ax6.plot(x_plot[:],vz[2:N+2],color="orange",linewidth=2,label="HLLD (t = {:.2f})".format(t))
			ax6.plot(x_plot[:],vz_hll[2:N+2],"g--",linewidth=2,label="HLL (t = {:.2f})".format(t))
			ax6.plot(x_plot[:],vz_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax6.set_xlabel(r"$x$",fontsize=24)
			ax6.set_ylabel(r"$v_z$",fontsize=24)
			ax6.set_xlim([x_min,x_max])
			ax6.legend()

			ax7=fig1.add_subplot(2,2,4)
			ax7.plot(x_plot[:],Bz[2:N+2],color="orange",linewidth=2,label="HLLD (t = {:.2f})".format(t))
			ax7.plot(x_plot[:],Bz_hll[2:N+2],"g--",linewidth=2,label="HLL (t = {:.2f})".format(t))
			ax7.plot(x_plot[:],Bz_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax7.set_xlabel(r"$x$",fontsize=24)
			ax7.set_ylabel(r"$B_z$",fontsize=24)
			ax7.set_xlim([x_min,x_max])
			ax7.set_ylim([min(Bz[2:N+2])-0.1,max(Bz[2:N+2])+0.1])
			ax7.legend(loc="upper left")
	print("-------------------Dai & Woodward (DW) shock test DONE!-------------------")
elif Test_type == "BW":
	"""
	Brio & Wu test********************************************************
	"""
	print("-------------------Brio & Wu (BW) test------------------------")
	"""
	Grid (cell) initilization
	"""
	x_min = 0.
	x_max = 1.
	N = 800 			#N = 800				#number of REAL cells
	delta_x = (x_max - x_min) / N 	#width of each cell
	########################################################

	"""
	Basic parameters and constants
	"""
	gamma = 2.
	CFL_num = 0.3
	#######################################################
	"""
	Time initialization
	"""
	t_init = 0.		#initial time
	t_final = 0.09	#final time

	t_plot = 0.08	#time to produce the plot
	#########################################################
	"""
	Initial conditions list
	"""
	t = t_init
	#left half region
	rho_L = 1.0
	P_L = 1.0
	vx_L = 0.
	vy_L = 0.
	vz_L = 0.

	Bx_const = 0.75

	By_L = 1.
	Bz_L = 0.

	#right half region
	rho_R = 0.125
	P_R = 0.1
	vx_R = 0.
	vy_R = 0.
	vz_R = 0.

	Bx_const = 0.75

	By_R = -1.
	Bz_R = 0.
	#############################################################
	"""
	Vector initialization and initial condition setting
	"""
	rho = np.zeros(N+4,float)			#rho is the array of the density
	vx = np.zeros(N+4,float)			#vx is the array of the x-comp velocity
	vy = np.zeros(N+4,float)			#vy is the array of the y-comp velocity
	vz = np.zeros(N+4,float)			#vz is the array of the z-comp velocity
	P = np.zeros(N+4,float)				#P is the array of the pressure
	Bx = np.zeros(N+4,float)			#Bx is the array of x-comp of magnetic field
	By = np.zeros(N+4,float)			#By is the array of y-comp of magnetic field
	Bz = np.zeros(N+4,float)			#Bz is the array of z-comp of magnetic field
	E = np.empty(N+4,float)				#E is the array of the Energy density

	rho_hll = np.zeros(N+4,float)			#rho is the array of the density
	vx_hll = np.zeros(N+4,float)			#vx is the array of the x-comp velocity
	vy_hll = np.zeros(N+4,float)			#vy is the array of the y-comp velocity
	vz_hll = np.zeros(N+4,float)			#vz is the array of the z-comp velocity
	P_hll = np.zeros(N+4,float)				#P is the array of the pressure
	Bx_hll = np.zeros(N+4,float)			#Bx is the array of x-comp of magnetic field
	By_hll = np.zeros(N+4,float)			#By is the array of y-comp of magnetic field
	Bz_hll = np.zeros(N+4,float)			#Bz is the array of z-comp of magnetic field
	E_hll = np.empty(N+4,float)				#E is the array of the Energy density
	
	#initial values
	rho_iv = np.zeros(N+4,float)
	vx_iv = np.zeros(N+4,float)
	vy_iv = np.zeros(N+4,float)
	vz_iv = np.zeros(N+4,float)
	P_iv = np.zeros(N+4,float)
	Bx_iv = np.zeros(N+4,float)
	By_iv = np.zeros(N+4,float)
	Bz_iv = np.zeros(N+4,float)
	E_iv = np.zeros(N+4,float)

	#initial condtion setting for real cells
	rho[2 : N//2+2] = rho_L
	rho[N//2+2 : N+2] = rho_R

	vx[2 : N//2+2] = vx_L
	vx[N//2+2 : N+2] = vx_R
	vy[2 : N//2+2] = vy_L
	vy[N//2+2 : N+2] = vy_R
	vz[2 : N//2+2] = vz_L
	vz[N//2+2 : N+2] = vz_R

	P[2 : N//2+2] = P_L
	P[N//2+2 : N+2] = P_R

	Bx[:] = Bx_const

	By[2 : N//2+2] = By_L
	By[N//2+2 : N+2] = By_R
	Bz[2 : N//2+2] = Bz_L
	Bz[N//2+2 : N+2] = Bz_R

	E[2 : N+2] = P[2 : N+2]/(gamma - 1.) + 0.5 * rho[2 : N+2] * (vx[2 : N+2]*vx[2 : N+2] + vy[2 : N+2]*vy[2 : N+2] + vz[2 : N+2]*vz[2 : N+2]) + 0.5 * (Bx[2 : N+2]*Bx[2 : N+2] + By[2 : N+2]*By[2 : N+2] + Bz[2 : N+2]*Bz[2 : N+2])

	#initial condition setting for ghost cells
	rho[0:2] = rho[2]
	rho[-2:] = rho[N+1]

	vx[0:2] = vx[2]
	vx[-2:] = vx[N+1]

	vy[0:2] = vy[2]
	vy[-2:] = vy[N+1]

	vz[0:2] = vz[2]
	vz[-2:] = vz[N+1]

	P[0:2] = P[2]
	P[-2:] = P[N+1]

	By[0:2] = By[2]
	By[-2:] = By[N+1]

	Bz[0:2] = Bz[2]
	Bz[-2:] = Bz[N+1]

	E[0:2] = E[2]
	E[-2:] = E[N+1]

	rho_iv[:] = rho[:]
	vx_iv[:] = vx[:]
	vy_iv[:] = vy[:]
	vz_iv[:] = vz[:]
	P_iv[:] = P[:]
	Bx_iv[:] = Bx[:]
	By_iv[:] = By[:]
	Bz_iv[:] = Bz[:]
	E_iv[:] = E[:]

	################################################################

	"""
	Convert all INITIAL physical quantities into the vector of conserved variables U
	U is a 7 * (N+4) matrix
	"""
	U = np.zeros([7,N+4],float)			# HLLD data
	U_hll = np.zeros([7,N+4],float)	 	# HLL data

	U[0,:] = rho[:]
	U[1,:] = rho[:] * vx[:]
	U[2,:] = rho[:] * vy[:]
	U[3,:] = rho[:] * vz[:]
	U[4,:] = E[:]
	U[5,:] = By[:]
	U[6,:] = Bz[:]

	#just for INITIAL conditions can we use these relations
	#otherwise, we should have things like U_hll[0,:] = rho_hll[:]
	U_hll[0,:] = rho[:]
	U_hll[1,:] = rho[:] * vx[:]
	U_hll[2,:] = rho[:] * vy[:]
	U_hll[3,:] = rho[:] * vz[:]
	U_hll[4,:] = E[:]
	U_hll[5,:] = By[:]
	U_hll[6,:] = Bz[:]

	"""
	make a copy of U for applying Dirichlet B.C. 
	"""
	U_init = np.zeros([7,N+4],float)
	U_init = np.copy(U)

	"""
	Temporary U for updating U using 3rd Runge-Kutta scheme (Shu & Osher)
	"""
	U_temp_1 = np.zeros([7,N+4],float)
	U_temp_2 = np.zeros([7,N+4],float)

	U_hll_temp_1 = np.zeros([7,N+4],float)
	U_hll_temp_2 = np.zeros([7,N+4],float)
	##################################################################

	"""
	Choice of the type of boundary condtion:
	0. Dirichlet
	1. Neumann
	"""
	Boundary_type = float(input("Please choose the type of boundary condition (0 for Dirichlet B.C., 1 for Neumann B.C.):"))
	if Boundary_type == 0:
		print("The Boundary type is Dirichlet B.C.")
	elif Boundary_type == 1:
		print("The Boundary type is Neumann B.C.")
	else:
		print("Please choose '0' or '1' or '2'!")

	while t<t_final:
		delta_t = MHD1dSS.GetDt(gamma,N,delta_x,CFL_num,U,Bx)
		delta_t_hll = MHD1dSS.GetDt(gamma,N,delta_x,CFL_num,U_hll,Bx)
		#REAL cells are updated using Lu and 3rd-order Runge-Kutta
		#GHOST cells are updated using boundary condition (Dirichlet or Neumann)
		U_temp_1 = U + delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U,Bx)
		U_temp_2 = 3./4. * U + 1./4. * U_temp_1 + 1./4. * delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U_temp_1,Bx)
		U = 1./3. * U + 2./3. * U_temp_2 + 2./3. * delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U_temp_2,Bx) 

		U_hll_temp_1 = U_hll + delta_t_hll * MHD1dSS.Lu_PLM_HLL(gamma,N,delta_x,U_hll,Bx)
		U_hll_temp_2 = 3./4. * U_hll + 1./4. * U_hll_temp_1 + 1./4. * delta_t_hll * MHD1dSS.Lu_PLM_HLL(gamma,N,delta_x,U_hll_temp_1,Bx)
		U_hll = 1./3. * U_hll + 2./3. * U_hll_temp_2 + 2./3. * delta_t_hll * MHD1dSS.Lu_PLM_HLL(gamma,N,delta_x,U_hll_temp_2,Bx) 
		#adjust ghost cells
		if Boundary_type == 0:
			#Dirichlet
			U[:,0] = U_init[:,0]
			U[:,1] = U_init[:,1]
			U[:,-2] = U_init[:,-2]
			U[:,-1] = U_init[:,-1]

			U_hll[:,0] = U_init[:,0]
			U_hll[:,1] = U_init[:,1]
			U_hll[:,-2] = U_init[:,-2]
			U_hll[:,-1] = U_init[:,-1]
		elif Boundary_type == 1:
			#Neumann
			U[:,0] = U[:,2]
			U[:,1] = U[:,2]
			U[:,-2] = U[:,N+1]
			U[:,-1] = U[:,N+1]

			U_hll[:,0] = U_hll[:,2]
			U_hll[:,1] = U_hll[:,2]
			U_hll[:,-2] = U_hll[:,N+1]
			U_hll[:,-1] = U_hll[:,N+1]

		t += delta_t

		if abs(t - t_plot) < 0.5*delta_t:
			print("Plot at time t = ", t)
			rho[:] = U[0,:]
			vx[:] = U[1,:] / U[0,:]
			vy[:] = U[2,:] / U[0,:]
			vz[:] = U[3,:] / U[0,:]
			E[:] = U[4,:]
			By[:] = U[5,:]
			Bz[:] = U[6,:]
			P[:] = (gamma - 1.) * (E[:] - 0.5 * rho[:] * (vx[:]*vx[:] + vy[:]*vy[:] + vz[:]*vz[:]) - 0.5 * (Bx[:]*Bx[:] + By[:]*By[:] + Bz[:]*Bz[:]) )
			
			rho_hll[:] = U_hll[0,:]
			vx_hll[:] = U_hll[1,:] / U_hll[0,:]
			vy_hll[:] = U_hll[2,:] / U_hll[0,:]
			vz_hll[:] = U_hll[3,:] / U_hll[0,:]
			E_hll[:] = U_hll[4,:]
			By_hll[:] = U_hll[5,:]
			Bz_hll[:] = U_hll[6,:]
			P_hll[:] = (gamma - 1.) * (E_hll[:] - 0.5 * rho_hll[:] * (vx_hll[:]*vx_hll[:] + vy_hll[:]*vy_hll[:] + vz_hll[:]*vz_hll[:]) - 0.5 * (Bx[:]*Bx[:] + By_hll[:]*By_hll[:] + Bz_hll[:]*Bz_hll[:]) )

			x_plot = np.empty(N,float)
			for i in range(N):
				x_plot[i] = x_min + (i+0.5) * delta_x

			# add plot
			fig0 = plt.figure()
			ax0=fig0.add_subplot(2,2,1)
			ax0.plot(x_plot[:],rho[2:N+2],color="orange",linewidth=1,label="HLLD (t = {:.2f})".format(t))
			ax0.plot(x_plot[:],rho_hll[2:N+2],"g--",linewidth=1,label="HLL (t = {:.2f})".format(t))
			ax0.plot(x_plot[:],rho_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax0.set_xlabel(r"$x$",fontsize=24)
			ax0.set_ylabel(r"$\rho$",fontsize=24)
			ax0.set_xlim([x_min,x_max])
			ax0.set_ylim([min(rho[2:N+2])-0.1, max(rho[2:N+2])+0.1])
			ax0.legend()

			ax1=fig0.add_subplot(2,2,2)
			ax1.plot(x_plot[:],P[2:N+2],color="orange",linewidth=1,label="HLLD (t = {:.2f})".format(t))
			ax1.plot(x_plot[:],P_hll[2:N+2],"g--",linewidth=1,label="HLL (t = {:.2f})".format(t))
			ax1.plot(x_plot[:],P_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax1.set_xlabel(r"$x$",fontsize=24)
			ax1.set_ylabel(r"$P$",fontsize=24)
			ax1.set_xlim([x_min,x_max])
			ax1.set_ylim([min(P[2:N+2])-0.1, max(P[2:N+2]+0.1)])
			ax1.legend()

			ax2=fig0.add_subplot(2,2,3)
			ax2.plot(x_plot[:],vx[2:N+2],color="orange",linewidth=1,label="HLLD (t = {:.2f})".format(t))
			ax2.plot(x_plot[:],vx_hll[2:N+2],"g--",linewidth=1,label="HLL (t = {:.2f})".format(t))
			ax2.plot(x_plot[:],vx_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax2.set_xlabel(r"$x$",fontsize=24)
			ax2.set_ylabel(r"$v_x$",fontsize=24)
			ax2.set_xlim([x_min,x_max])
			ax2.set_ylim([min(vx[2:N+2])-0.1, max(vx[2:N+2])+0.1])
			ax2.legend(loc="upper left")

			ax3=fig0.add_subplot(2,2,4)
			ax3.plot(x_plot[:],By[2:N+2],color="orange",linewidth=1,label="HLLD (t = {:.2f})".format(t))
			ax3.plot(x_plot[:],By_hll[2:N+2],"g--",linewidth=1,label="HLL (t = {:.2f})".format(t))
			ax3.plot(x_plot[:],By_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax3.set_xlabel(r"$x$",fontsize=24)
			ax3.set_ylabel(r"$B_y$",fontsize=24)
			ax3.set_xlim([x_min,x_max])
			ax3.set_ylim([min(By[2:N+2])-0.1, max(By[2:N+2])+0.1])
			ax3.legend()

			fig1 = plt.figure()
			ax4=fig1.add_subplot(2,2,1)
			ax4.plot(x_plot[:],vy[2:N+2],color="orange",linewidth=1,label="HLLD (t = {:.2f})".format(t))
			ax4.plot(x_plot[:],vy_hll[2:N+2],"g--",linewidth=1,label="HLL (t = {:.2f})".format(t))
			ax4.plot(x_plot[:],vy_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax4.set_xlabel(r"$x$",fontsize=24)
			ax4.set_ylabel(r"$v_y$",fontsize=24)
			ax4.set_xlim([x_min,x_max])
			ax4.set_ylim([min(vy[2:N+2])-0.2, max(vy[2:N+2])+0.2])
			ax4.legend(loc="lower left")

			ax5=fig1.add_subplot(2,2,2)
			ax5.plot(x_plot[:],Bx[2:N+2],color="orange",linewidth=1,label="HLLD (t = {:.2f})".format(t))
			ax5.plot(x_plot[:],Bx[2:N+2],"g--",linewidth=1,label="HLL (t = {:.2f})".format(t))
			ax5.plot(x_plot[:],Bx_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax5.set_xlabel(r"$x$",fontsize=24)
			ax5.set_ylabel(r"$B_x$",fontsize=24)
			ax5.set_xlim([x_min,x_max])
			ax5.legend()

			ax6=fig1.add_subplot(2,2,3)
			ax6.plot(x_plot[:],Bz[2:N+2],color="orange",linewidth=1,label="HLLD (t = {:.2f})".format(t))
			ax6.plot(x_plot[:],Bz_hll[2:N+2],"g--",linewidth=1,label="HLL (t = {:.2f})".format(t))
			ax6.plot(x_plot[:],Bz_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax6.set_xlabel(r"$x$",fontsize=24)
			ax6.set_ylabel(r"$B_z$",fontsize=24)
			ax6.set_xlim([x_min,x_max])
			ax6.legend()

			ax7=fig1.add_subplot(2,2,4)
			ax7.plot(x_plot[:],E[2:N+2],color="orange",linewidth=1,label="HLLD (t = {:.2f})".format(t))
			ax7.plot(x_plot[:],E_hll[2:N+2],"g--",linewidth=1,label="HLL (t = {:.2f})".format(t))
			ax7.plot(x_plot[:],E_iv[2:N+2],color="red",linestyle="dotted",linewidth=2,label="t = 0")
			ax7.set_xlabel(r"$x$",fontsize=24)
			ax7.set_ylabel(r"$E$",fontsize=24)
			ax7.set_xlim([x_min,x_max])
			ax7.set_ylim([min(E[2:N+2])-0.1,max(E[2:N+2])+0.1])
			ax7.legend()

	print("-------------------Brio & Wu (BW) test DONE!-------------------")
elif Test_type == "AW":
	"""
	Grid (cell) initilization
	"""
	x_min = -1.
	x_max = 1.
	N = 800 			#N = 800				#number of REAL cells
	delta_x = (x_max - x_min) / N 	#width of each cell
	########################################################
	"""
	Alfven Waves test********************************************************
	"""
	print("-------------------Alfven Waves (AW) test------------------------")
	"""
	Initial functions for vy,vz,By,Bz
	"""
	def vy_init(x):
		alpha = 0
		return 0.1 * np.sin(2 * np.pi * x * np.cos(alpha))

	def By_init(x):
		alpha = 0
		return 0.1 * np.sin(2 * np.pi * x * np.cos(alpha))
	
	def vz_init(x):
		alpha = 0
		return 0.1 * np.cos(2 * np.pi * x * np.cos(alpha))
	
	def Bz_init(x):
		alpha = 0
		return 0.1 * np.cos(2 * np.pi * x * np.cos(alpha))
	"""
	Basic parameters and constants
	"""
	gamma = 5./3.
	CFL_num = 0.6
	#######################################################
	"""
	Time initialization
	"""
	t_init = 0.		#initial time
	t_final = 2.501	#final time

	t_plot_1 = 1.00
	t_plot_2 = 2.50	#time to produce the plot
	#########################################################
	"""
	Initial conditions list
	"""
	t = t_init
	#left half region
	rho_L = 1.0
	P_L = 1.0
	vx_L = 0.
	#vy_L = 1.
	#vz_L = 1.

	Bx_const = 1.

	#By_L = 1.
	#Bz_L = 0.

	#right half region
	rho_R = 1.
	P_R = 1.
	vx_R = 0.
	#vy_R = 1.
	#vz_R = 1.

	Bx_const = 1.

	#By_R = 1.
	#Bz_R = 0.
	#############################################################
	"""
	Vector initialization and initial condition setting
	"""
	rho = np.zeros(N+4,float)			#rho is the array of the density
	vx = np.zeros(N+4,float)			#vx is the array of the x-comp velocity
	vy = np.zeros(N+4,float)			#vy is the array of the y-comp velocity
	vz = np.zeros(N+4,float)			#vz is the array of the z-comp velocity
	P = np.zeros(N+4,float)				#P is the array of the pressure
	Bx = np.zeros(N+4,float)			#Bx is the array of x-comp of magnetic field
	By = np.zeros(N+4,float)			#By is the array of y-comp of magnetic field
	Bz = np.zeros(N+4,float)			#Bz is the array of z-comp of magnetic field
	E = np.empty(N+4,float)				#E is the array of the Energy density

	#initial values
	rho_iv = np.zeros(N+4,float)
	vx_iv = np.zeros(N+4,float)
	vy_iv = np.zeros(N+4,float)
	vz_iv = np.zeros(N+4,float)
	P_iv = np.zeros(N+4,float)
	Bx_iv = np.zeros(N+4,float)
	By_iv = np.zeros(N+4,float)
	Bz_iv = np.zeros(N+4,float)
	E_iv = np.zeros(N+4,float)


	#initial condtion setting for real cells
	rho[2 : N//2+2] = rho_L
	rho[N//2+2 : N+2] = rho_R

	vx[2 : N//2+2] = vx_L
	vx[N//2+2 : N+2] = vx_R

	# vy[2 : N//2+2] = vy_L
	# vy[N//2+2 : N+2] = vy_R
	# vz[2 : N//2+2] = vz_L
	# vz[N//2+2 : N+2] = vz_R

	for i in range(2,N+2):
		vy[i] = vy_init(x_min + (i-1.5) * delta_x)
		vz[i] = vz_init(x_min + (i-1.5) * delta_x)
		By[i] = By_init(x_min + (i-1.5) * delta_x)
		Bz[i] = Bz_init(x_min + (i-1.5) * delta_x)


	P[2 : N//2+2] = P_L
	P[N//2+2 : N+2] = P_R

	Bx[:] = Bx_const

	# By[2 : N//2+2] = By_L
	# By[N//2+2 : N+2] = By_R
	# Bz[2 : N//2+2] = Bz_L
	# Bz[N//2+2 : N+2] = Bz_R

	E[2 : N+2] = P[2 : N+2]/(gamma - 1.) + 0.5 * rho[2 : N+2] * (vx[2 : N+2]*vx[2 : N+2] + vy[2 : N+2]*vy[2 : N+2] + vz[2 : N+2]*vz[2 : N+2]) + 0.5 * (Bx[2 : N+2]*Bx[2 : N+2] + By[2 : N+2]*By[2 : N+2] + Bz[2 : N+2]*Bz[2 : N+2])

	#initial condition setting for ghost cells
	rho[0:2] = rho[2]
	rho[-2:] = rho[N+1]

	vx[0:2] = vx[2]
	vx[-2:] = vx[N+1]

	vy[0:2] = vy[2]
	vy[-2:] = vy[N+1]

	vz[0:2] = vz[2]
	vz[-2:] = vz[N+1]

	P[0:2] = P[2]
	P[-2:] = P[N+1]

	By[0:2] = By[2]
	By[-2:] = By[N+1]

	Bz[0:2] = Bz[2]
	Bz[-2:] = Bz[N+1]

	E[0:2] = E[2]
	E[-2:] = E[N+1]


	rho_iv[:] = rho[:]
	vx_iv[:] = vx[:]
	vy_iv[:] = vy[:]
	vz_iv[:] = vz[:]
	P_iv[:] = P[:]
	Bx_iv[:] = Bx[:]
	By_iv[:] = By[:]
	Bz_iv[:] = Bz[:]
	E_iv[:] = E[:]
	################################################################
	"""
	Convert all physical quantities into the vector of conserved variables U
	U is a 7 * (N+4) matrix
	"""
	U = np.zeros([7,N+4],float)
	U[0,:] = rho[:]
	U[1,:] = rho[:] * vx[:]
	U[2,:] = rho[:] * vy[:]
	U[3,:] = rho[:] * vz[:]
	U[4,:] = E[:]
	U[5,:] = By[:]
	U[6,:] = Bz[:]

	"""
	make a copy of U for applying Dirichlet B.C. 
	"""
	U_init = np.zeros([7,N+4],float)
	U_init = np.copy(U)

	"""
	Temporary U for updating U using 3rd Runge-Kutta scheme (Shu & Osher)
	"""
	U_temp_1 = np.zeros([7,N+4],float)
	U_temp_2 = np.zeros([7,N+4],float)
	##################################################################

	"""
	Choice of the type of boundary condtion:
	0. Dirichlet
	1. Neumann
	2. periodic
	"""
	Boundary_type = 2	#default value is 2
	Boundary_type = float(input("Please choose the type of boundary condition (0 for Dirichlet B.C., 1 for Neumann B.C., 2 for periodic B.C.): "))
	if Boundary_type == 0:
		print("The Boundary type is Dirichlet B.C.")
	elif Boundary_type == 1:
		print("The Boundary type is Neumann B.C.")
	elif Boundary_type == 2:
		print("The Boundary type is periodic B.C.")
	else:
		print("Please choose '0' or '1'!")

	counter=0
	while t<t_final:
		delta_t = MHD1dSS.GetDt(gamma,N,delta_x,CFL_num,U,Bx)
		#REAL cells are updated using Lu and 3rd-order Runge-Kutta
		#GHOST cells are updated using boundary condition (Dirichlet or Neumann)
		U_temp_1 = U + delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U,Bx)
		U_temp_2 = 3./4. * U + 1./4. * U_temp_1 + 1./4. * delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U_temp_1,Bx)
		U = 1./3. * U + 2./3. * U_temp_2 + 2./3. * delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U_temp_2,Bx) 
		#adjust ghost cells
		if Boundary_type == 0:
			#Dirichlet
			U[:,0] = U_init[:,0]
			U[:,1] = U_init[:,1]
			U[:,-2] = U_init[:,-2]
			U[:,-1] = U_init[:,-1]
		elif Boundary_type == 1:
			#Neumann
			U[:,0] = U[:,2]
			U[:,1] = U[:,2]
			U[:,-2] = U[:,N+1]
			U[:,-1] = U[:,N+1]
		elif Boundary_type == 2:
			#periodic
			U[:,0] = U[:,N]
			U[:,1] = U[:,N+1]
			U[:,N+2] = U[:,2]
			U[:,N+3] = U[:,3]

		t += delta_t

		if abs(t - t_plot_1) < 0.5*delta_t:
		# if t - t_plot < 0.5*delta_t:
			#counting for figures output
			counter += 1
			print(counter)
			# print("Plot at time t = ", t)
			rho[:] = U[0,:]
			vx[:] = U[1,:] / U[0,:]
			vy[:] = U[2,:] / U[0,:]
			vz[:] = U[3,:] / U[0,:]
			E[:] = U[4,:]
			By[:] = U[5,:]
			Bz[:] = U[6,:]
			P[:] = (gamma - 1.) * (E[:] - 0.5 * rho[:] * (vx[:]*vx[:] + vy[:]*vy[:] + vz[:]*vz[:]) - 0.5 * (Bx[:]*Bx[:] + By[:]*By[:] + Bz[:]*Bz[:]) )
			
			x_plot = np.empty(N,float)
			for i in range(N):
				x_plot[i] = x_min + (i+0.5) * delta_x

			#add plot
			fig0 = plt.figure()
			ax0=fig0.add_subplot(1,2,1)
			ax0.plot(x_plot[:],rho_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax0.plot(x_plot[:],rho[2:N+2],color="green",linestyle="dotted",linewidth=2,label="t = {:.2f}".format(t))
			ax0.set_xlabel(r"$x$",fontsize=24)
			ax0.set_ylabel(r"$\rho$",fontsize=24)
			ax0.set_xlim([x_min,x_max])
			ax0.set_ylim([min(rho[2:N+2])-0.1, max(rho[2:N+2])+0.1])

			ax1=fig0.add_subplot(1,2,2)
			ax1.plot(x_plot[:],P_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax1.plot(x_plot[:],P[2:N+2],color="green",linestyle="dotted",linewidth=2,label="t = {:.2f}".format(t))
			ax1.set_xlabel(r"$x$",fontsize=24)
			ax1.set_ylabel(r"$P$",fontsize=24)
			ax1.set_xlim([x_min,x_max])
			ax1.set_ylim([min(P[2:N+2])-0.1, max(P[2:N+2])+0.1])

			fig1 = plt.figure()
			ax2=fig1.add_subplot(3,2,1)
			ax2.plot(x_plot[:],vx_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax2.plot(x_plot[:],vx[2:N+2],color="green",linestyle="dotted",linewidth=2,label="t = {:.2f}".format(t))
			ax2.set_xlabel(r"$x$",fontsize=17)
			ax2.set_ylabel(r"$v_x$",fontsize=17)
			ax2.set_xlim([x_min,x_max])
			ax2.set_ylim([min(vx[2:N+2])-0.1, max(vx[2:N+2])+0.1])

			ax5=fig1.add_subplot(3,2,2)
			ax5.plot(x_plot[:],Bx_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax5.plot(x_plot[:],Bx[2:N+2],color="green",linestyle="dotted",linewidth=2,label="t = {:.2f}".format(t))
			ax5.set_xlabel(r"$x$",fontsize=17)
			ax5.set_ylabel(r"$B_x$",fontsize=17)
			ax5.set_xlim([x_min,x_max])
			ax5.set_ylim([min(Bx[2:N+2])-0.1, max(Bx[2:N+2])+0.1])

			# ax5.yaxis.set_label_position("right")

			ax4=fig1.add_subplot(3,2,3)
			ax4.plot(x_plot[:],vy_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax4.plot(x_plot[:],vy[2:N+2],color="green",linestyle="dotted",linewidth=2,label="t = {:.2f}".format(t))
			ax4.set_xlabel(r"$x$",fontsize=17)
			ax4.set_ylabel(r"$v_y$",fontsize=17)
			ax4.set_xlim([x_min,x_max])
			ax4.set_ylim([min(vy[2:N+2])-0.1, max(vy[2:N+2])+0.1])

			ax3=fig1.add_subplot(3,2,4)
			ax3.plot(x_plot[:],By_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax3.plot(x_plot[:],By[2:N+2],color="green",linestyle="dotted",linewidth=2,label="t = {:.2f}".format(t))
			ax3.set_xlabel(r"$x$",fontsize=17)
			ax3.set_ylabel(r"$B_y$",fontsize=17)
			ax3.set_xlim([x_min,x_max])
			ax3.set_ylim([min(By[2:N+2])-0.1, max(By[2:N+2])+0.1])

			# ax3.yaxis.set_label_position("right")

			ax8=fig1.add_subplot(3,2,5)
			ax8.plot(x_plot[:],vz_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax8.plot(x_plot[:],vz[2:N+2],color="green",linestyle="dotted",linewidth=2,label="t = {:.2f}".format(t))
			ax8.set_xlabel(r"$x$",fontsize=17)
			ax8.set_ylabel(r"$v_z$",fontsize=17)
			ax8.set_xlim([x_min,x_max])
			ax8.set_ylim([min(vz[2:N+2])-0.1, max(vz[2:N+2])+0.1])

			ax6=fig1.add_subplot(3,2,6)
			ax6.plot(x_plot[:],Bz_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax6.plot(x_plot[:],Bz[2:N+2],color="green",linestyle="dotted",linewidth=2,label="t = {:.2f}".format(t))
			ax6.set_xlabel(r"$x$",fontsize=17)
			ax6.set_ylabel(r"$B_z$",fontsize=17)
			ax6.set_xlim([x_min,x_max])
			ax6.set_ylim([min(Bz[2:N+2])-0.1, max(Bz[2:N+2])+0.1])

			# ax6.yaxis.set_label_position("right")

			# fig1.suptitle("t = {0:.5g}".format(t))
			# fig1.savefig("{0:s}_{1:05d}.{2:s}".format("AW2",counter,"png"))
			# plt.close(fig1)

		if abs(t - t_plot_2) < 0.5*delta_t:
			# if t - t_plot < 0.5*delta_t:
			#counting for figures output
			# counter += 1
			# print(counter)
			# print("Plot at time t = ", t)
			rho[:] = U[0,:]
			vx[:] = U[1,:] / U[0,:]
			vy[:] = U[2,:] / U[0,:]
			vz[:] = U[3,:] / U[0,:]
			E[:] = U[4,:]
			By[:] = U[5,:]
			Bz[:] = U[6,:]
			P[:] = (gamma - 1.) * (E[:] - 0.5 * rho[:] * (vx[:]*vx[:] + vy[:]*vy[:] + vz[:]*vz[:]) - 0.5 * (Bx[:]*Bx[:] + By[:]*By[:] + Bz[:]*Bz[:]) )
			
			x_plot = np.empty(N,float)
			for i in range(N):
				x_plot[i] = x_min + (i+0.5) * delta_x

			ax0.plot(x_plot[:],rho[2:N+2],color="blue",linewidth=2,label="t = {:.2f}".format(t))
			ax1.plot(x_plot[:],P[2:N+2],color="blue",linewidth=2,label="t = {:.2f}".format(t))

			ax2.plot(x_plot[:],vx[2:N+2],color="blue",linewidth=2,label="t = {:.2f}".format(t))
			ax5.plot(x_plot[:],Bx[2:N+2],color="blue",linewidth=2,label="t = {:.2f}".format(t))
			ax4.plot(x_plot[:],vy[2:N+2],color="blue",linewidth=2,label="t = {:.2f}".format(t))
			ax3.plot(x_plot[:],By[2:N+2],color="blue",linewidth=2,label="t = {:.2f}".format(t))
			ax8.plot(x_plot[:],vz[2:N+2],color="blue",linewidth=2,label="t = {:.2f}".format(t))
			ax6.plot(x_plot[:],Bz[2:N+2],color="blue",linewidth=2,label="t = {:.2f}".format(t))

			ax0.legend()
			ax1.legend()
			ax2.legend()
			ax3.legend()
			ax4.legend()
			ax5.legend()
			ax6.legend()
			ax8.legend()



	print("-------------------Alfven Waves (AW) test DONE!------------------------")
elif Test_type == "FS":
	"""
	Grid (cell) initilization
	"""
	x_min = -0.5
	x_max = 3.
	N = 400 			#N = 800				#number of REAL cells
	delta_x = (x_max - x_min) / N 	#width of each cell
	########################################################
	"""
	Fast switch-on Shock test********************************************************
	"""
	print("-------------------Fast Switch-on (FS) shock test------------------------")
	"""
	Basic parameters and constants
	"""
	gamma = 5./3.
	CFL_num = 0.3
	#######################################################
	"""
	Time initialization
	"""
	t_init = 0.		#initial time
	t_final = 0.41	#final time

	t_plot = 0.4	#time to produce the plot
	#########################################################
	"""
	Initial conditions list
	"""
	t = t_init
	#left half region
	rho_L = 3.0
	P_L = 16.33
	vx_L = -0.732
	vy_L = -1.333
	vz_L = 0.

	Bx_const = 3.

	By_L = 2.309
	Bz_L = 1.

	#right half region
	rho_R = 1.
	P_R = 1.
	vx_R = -4.196
	vy_R = 0.
	vz_R = 0.

	Bx_const = 3.

	By_R = 0.
	Bz_R = 0.
	#############################################################
	"""
	Vector initialization and initial condition setting
	"""
	rho = np.zeros(N+4,float)			#rho is the array of the density
	vx = np.zeros(N+4,float)			#vx is the array of the x-comp velocity
	vy = np.zeros(N+4,float)			#vy is the array of the y-comp velocity
	vz = np.zeros(N+4,float)			#vz is the array of the z-comp velocity
	P = np.zeros(N+4,float)				#P is the array of the pressure
	Bx = np.zeros(N+4,float)			#Bx is the array of x-comp of magnetic field
	By = np.zeros(N+4,float)			#By is the array of y-comp of magnetic field
	Bz = np.zeros(N+4,float)			#Bz is the array of z-comp of magnetic field
	E = np.empty(N+4,float)				#E is the array of the Energy density

	#initial values
	rho_iv = np.zeros(N+4,float)
	vx_iv = np.zeros(N+4,float)
	vy_iv = np.zeros(N+4,float)
	vz_iv = np.zeros(N+4,float)
	P_iv = np.zeros(N+4,float)
	Bx_iv = np.zeros(N+4,float)
	By_iv = np.zeros(N+4,float)
	Bz_iv = np.zeros(N+4,float)
	E_iv = np.zeros(N+4,float)

	#initial condtion setting for real cells
	rho[2 : N//2+2] = rho_L
	rho[N//2+2 : N+2] = rho_R

	vx[2 : N//2+2] = vx_L
	vx[N//2+2 : N+2] = vx_R
	vy[2 : N//2+2] = vy_L
	vy[N//2+2 : N+2] = vy_R
	vz[2 : N//2+2] = vz_L
	vz[N//2+2 : N+2] = vz_R

	P[2 : N//2+2] = P_L
	P[N//2+2 : N+2] = P_R

	Bx[:] = Bx_const

	By[2 : N//2+2] = By_L
	By[N//2+2 : N+2] = By_R
	Bz[2 : N//2+2] = Bz_L
	Bz[N//2+2 : N+2] = Bz_R

	E[2 : N+2] = P[2 : N+2]/(gamma - 1.) + 0.5 * rho[2 : N+2] * (vx[2 : N+2]*vx[2 : N+2] + vy[2 : N+2]*vy[2 : N+2] + vz[2 : N+2]*vz[2 : N+2]) + 0.5 * (Bx[2 : N+2]*Bx[2 : N+2] + By[2 : N+2]*By[2 : N+2] + Bz[2 : N+2]*Bz[2 : N+2])

	#initial condition setting for ghost cells
	rho[0:2] = rho[2]
	rho[-2:] = rho[N+1]

	vx[0:2] = vx[2]
	vx[-2:] = vx[N+1]

	vy[0:2] = vy[2]
	vy[-2:] = vy[N+1]

	vz[0:2] = vz[2]
	vz[-2:] = vz[N+1]

	P[0:2] = P[2]
	P[-2:] = P[N+1]

	By[0:2] = By[2]
	By[-2:] = By[N+1]

	Bz[0:2] = Bz[2]
	Bz[-2:] = Bz[N+1]

	E[0:2] = E[2]
	E[-2:] = E[N+1]

	rho_iv[:] = rho[:]
	vx_iv[:] = vx[:]
	vy_iv[:] = vy[:]
	vz_iv[:] = vz[:]
	P_iv[:] = P[:]
	Bx_iv[:] = Bx[:]
	By_iv[:] = By[:]
	Bz_iv[:] = Bz[:]
	E_iv[:] = E[:]
	################################################################
	"""
	Convert all physical quantities into the vector of conserved variables U
	U is a 7 * (N+4) matrix
	"""
	U = np.zeros([7,N+4],float)
	U[0,:] = rho[:]
	U[1,:] = rho[:] * vx[:]
	U[2,:] = rho[:] * vy[:]
	U[3,:] = rho[:] * vz[:]
	U[4,:] = E[:]
	U[5,:] = By[:]
	U[6,:] = Bz[:]

	"""
	make a copy of U for applying Dirichlet B.C. 
	"""
	U_init = np.zeros([7,N+4],float)
	U_init = np.copy(U)

	"""
	Temporary U for updating U using 3rd Runge-Kutta scheme (Shu & Osher)
	"""
	U_temp_1 = np.zeros([7,N+4],float)
	U_temp_2 = np.zeros([7,N+4],float)
	##################################################################

	"""
	Choice of the type of boundary condtion:
	0. Dirichlet
	1. Neumann
	"""
	Boundary_type = float(input("Please choose the type of boundary condition (0 for Dirichlet B.C., 1 for Neumann B.C.): "))
	if Boundary_type == 0:
		print("The Boundary type is Dirichlet B.C.")
	elif Boundary_type == 1:
		print("The Boundary type is Neumann B.C.")
	else:
		print("Please choose '0' or '1'!")

	while t<t_final:
		delta_t = MHD1dSS.GetDt(gamma,N,delta_x,CFL_num,U,Bx)
		#REAL cells are updated using Lu and 3rd-order Runge-Kutta
		#GHOST cells are updated using boundary condition (Dirichlet or Neumann)
		U_temp_1 = U + delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U,Bx)
		U_temp_2 = 3./4. * U + 1./4. * U_temp_1 + 1./4. * delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U_temp_1,Bx)
		U = 1./3. * U + 2./3. * U_temp_2 + 2./3. * delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U_temp_2,Bx) 
		#adjust ghost cells
		if Boundary_type == 0:
			#Dirichlet
			U[:,0] = U_init[:,0]
			U[:,1] = U_init[:,1]
			U[:,-2] = U_init[:,-2]
			U[:,-1] = U_init[:,-1]
		elif Boundary_type == 1:
			#Neumann
			U[:,0] = U[:,2]
			U[:,1] = U[:,2]
			U[:,-2] = U[:,N+1]
			U[:,-1] = U[:,N+1]

		t += delta_t

		if abs(t - t_plot) < 0.5*delta_t:
			print("Plot at time t = ", t)
			rho[:] = U[0,:]
			vx[:] = U[1,:] / U[0,:]
			vy[:] = U[2,:] / U[0,:]
			vz[:] = U[3,:] / U[0,:]
			E[:] = U[4,:]
			By[:] = U[5,:]
			Bz[:] = U[6,:]
			P[:] = (gamma - 1.) * (E[:] - 0.5 * rho[:] * (vx[:]*vx[:] + vy[:]*vy[:] + vz[:]*vz[:]) - 0.5 * (Bx[:]*Bx[:] + By[:]*By[:] + Bz[:]*Bz[:]) )
			
			x_plot = np.empty(N,float)
			for i in range(N):
				x_plot[i] = x_min + (i+0.5) * delta_x

			# add plot
			fig0 = plt.figure()
			ax0=fig0.add_subplot(1,2,1)
			ax0.plot(x_plot[:],rho_iv[2:N+2],"r--",linewidth=2, label="t = 0")
			ax0.plot(x_plot[:],rho[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax0.set_xlabel(r"$x$",fontsize=24)
			ax0.set_ylabel(r"$\rho$",fontsize=24)
			ax0.set_xlim([x_min,x_max])
			ax0.set_ylim([min(rho[2:N+2])-0.1, max(rho[2:N+2])+0.1])
			ax0.legend()
		

			ax1=fig0.add_subplot(1,2,2)
			ax1.plot(x_plot[:],P_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax1.plot(x_plot[:],P[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax1.set_xlabel(r"$x$",fontsize=24)
			ax1.set_ylabel(r"$P$",fontsize=24)
			ax1.set_xlim([x_min,x_max])
			ax1.set_ylim([min(P[2:N+2])-1., max(P[2:N+2])+1.])
			ax1.legend()

			# ax7=fig0.add_subplot(1,3,3)
			# ax7.plot(x_plot[:],E_iv[2:N+2],"r--",linewidth=2)
			# ax7.plot(x_plot[:],E[2:N+2],"b",linewidth=2)
			# ax7.set_xlabel(r"$x$",fontsize=15)
			# ax7.set_ylabel(r"$E$",fontsize=13)
			# ax7.set_xlim([x_min,x_max])
			# ax7.set_ylim([min(E[2:N+2])-0.1, max(E[2:N+2])+0.1])

			fig1 = plt.figure()
			ax2=fig1.add_subplot(3,2,1)
			ax2.plot(x_plot[:],vx_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax2.plot(x_plot[:],vx[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax2.set_xlabel(r"$x$",fontsize=17)
			ax2.set_ylabel(r"$v_x$",fontsize=17)
			ax2.set_xlim([x_min,x_max])
			ax2.set_ylim([min(vx[2:N+2])-0.5, max(vx[2:N+2])+0.5])
			ax2.legend()

			ax4=fig1.add_subplot(3,2,2)
			ax4.plot(x_plot[:],vy_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax4.plot(x_plot[:],vy[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax4.set_xlabel(r"$x$",fontsize=17)
			ax4.set_ylabel(r"$v_y$",fontsize=17)
			ax4.set_xlim([x_min,x_max])
			ax4.set_ylim([min(vy[2:N+2])-0.1, max(vy[2:N+2])+0.1])
			ax4.legend()

			ax8=fig1.add_subplot(3,2,3)
			ax8.plot(x_plot[:],vz_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax8.plot(x_plot[:],vz[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax8.set_xlabel(r"$x$",fontsize=17)
			ax8.set_ylabel(r"$v_z$",fontsize=17)
			ax8.set_xlim([x_min,x_max])
			ax8.set_ylim([min(vz[2:N+2])-0.1, max(vz[2:N+2])+0.1])
			ax8.legend()

			ax5=fig1.add_subplot(3,2,4)
			ax5.plot(x_plot[:],Bx_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax5.plot(x_plot[:],Bx[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax5.set_xlabel(r"$x$",fontsize=17)
			ax5.set_ylabel(r"$B_x$",fontsize=17)
			ax5.set_xlim([x_min,x_max])
			ax5.set_ylim([min(Bx[2:N+2])-0.1, max(Bx[2:N+2])+0.1])
			ax5.legend()

			ax3=fig1.add_subplot(3,2,5)
			ax3.plot(x_plot[:],By_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax3.plot(x_plot[:],By[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax3.set_xlabel(r"$x$",fontsize=17)
			ax3.set_ylabel(r"$B_y$",fontsize=17)
			ax3.set_xlim([x_min,x_max])
			ax3.set_ylim([min(By[2:N+2])-0.1, max(By[2:N+2])+0.1])
			ax3.legend()

			# fig1 = plt.figure()
			# ax4=fig1.add_subplot(2,2,1)
			# ax4.plot(x_plot[:],vy_iv[2:N+2],"r--",linewidth=2)
			# ax4.plot(x_plot[:],vy[2:N+2],"b",linewidth=2)
			# ax4.set_xlabel(r"$x$",fontsize=24)
			# ax4.set_ylabel(r"$v_y$",fontsize=24)
			# ax4.set_xlim([x_min,x_max])
			# ax4.set_ylim([min(vy[2:N+2])-0.1, max(vy[2:N+2])+0.1])

			# ax5=fig1.add_subplot(2,2,2)
			# ax5.plot(x_plot[:],Bx_iv[2:N+2],"r--",linewidth=2)
			# ax5.plot(x_plot[:],Bx[2:N+2],"b",linewidth=2)
			# ax5.set_xlabel(r"$x$",fontsize=24)
			# ax5.set_ylabel(r"$B_x$",fontsize=24)
			# ax5.set_xlim([x_min,x_max])
			# ax5.set_ylim([min(Bx[2:N+2])-0.1, max(Bx[2:N+2])+0.1])

			ax6=fig1.add_subplot(3,2,6)
			ax6.plot(x_plot[:],Bz_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax6.plot(x_plot[:],Bz[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax6.set_xlabel(r"$x$",fontsize=17)
			ax6.set_ylabel(r"$B_z$",fontsize=17)
			ax6.set_xlim([x_min,x_max])
			ax6.set_ylim([min(Bz[2:N+2])-0.1, max(Bz[2:N+2])+0.1])
			ax6.legend()

			# ax7=fig1.add_subplot(2,2,4)
			# ax7.plot(x_plot[:],E_iv[2:N+2],"r--",linewidth=2)
			# ax7.plot(x_plot[:],E[2:N+2],"b",linewidth=2)
			# ax7.set_xlabel(r"$x$",fontsize=24)
			# ax7.set_ylabel(r"$E$",fontsize=24)
			# ax7.set_xlim([x_min,x_max])
			# ax7.set_ylim([min(E[2:N+2])-0.1, max(E[2:N+2])+0.1])

	print("-------------------Fast Switch-on (FS) shock test DONE!------------------------")
elif Test_type == "SS":
	"""
	Grid (cell) initilization
	"""
	x_min = -0.5
	x_max = 1.5
	N = 400 			#N = 800				#number of REAL cells
	delta_x = (x_max - x_min) / N 	#width of each cell
	########################################################
	"""
	Fast switch-on Shock test********************************************************
	"""
	print("-------------------Slow Switch-off (SS) shock test------------------------")
	"""
	Basic parameters and constants
	"""
	gamma = 5./3.
	CFL_num = 0.3
	#######################################################
	"""
	Time initialization
	"""
	t_init = 0.		#initial time
	t_final = 0.51	#final time

	t_plot = 0.5	#time to produce the plot
	#########################################################
	"""
	Initial conditions list
	"""
	t = t_init
	#left half region
	rho_L = 1.368
	P_L = 1.769
	vx_L = 0.269
	vy_L = 1.0
	vz_L = 0.

	Bx_const = 1.

	By_L = 0.
	Bz_L = 0.

	#right half region
	rho_R = 1.
	P_R = 1.
	vx_R = 0.
	vy_R = 0.
	vz_R = 0.

	Bx_const = 1.

	By_R = 1.
	Bz_R = 0.
	#############################################################
	"""
	Vector initialization and initial condition setting
	"""
	rho = np.zeros(N+4,float)			#rho is the array of the density
	vx = np.zeros(N+4,float)			#vx is the array of the x-comp velocity
	vy = np.zeros(N+4,float)			#vy is the array of the y-comp velocity
	vz = np.zeros(N+4,float)			#vz is the array of the z-comp velocity
	P = np.zeros(N+4,float)				#P is the array of the pressure
	Bx = np.zeros(N+4,float)			#Bx is the array of x-comp of magnetic field
	By = np.zeros(N+4,float)			#By is the array of y-comp of magnetic field
	Bz = np.zeros(N+4,float)			#Bz is the array of z-comp of magnetic field
	E = np.empty(N+4,float)				#E is the array of the Energy density

	#initial values
	rho_iv = np.zeros(N+4,float)
	vx_iv = np.zeros(N+4,float)
	vy_iv = np.zeros(N+4,float)
	vz_iv = np.zeros(N+4,float)
	P_iv = np.zeros(N+4,float)
	Bx_iv = np.zeros(N+4,float)
	By_iv = np.zeros(N+4,float)
	Bz_iv = np.zeros(N+4,float)
	E_iv = np.zeros(N+4,float)

	#initial condtion setting for real cells
	rho[2 : N//2+2] = rho_L
	rho[N//2+2 : N+2] = rho_R

	vx[2 : N//2+2] = vx_L
	vx[N//2+2 : N+2] = vx_R
	vy[2 : N//2+2] = vy_L
	vy[N//2+2 : N+2] = vy_R
	vz[2 : N//2+2] = vz_L
	vz[N//2+2 : N+2] = vz_R

	P[2 : N//2+2] = P_L
	P[N//2+2 : N+2] = P_R

	Bx[:] = Bx_const

	By[2 : N//2+2] = By_L
	By[N//2+2 : N+2] = By_R
	Bz[2 : N//2+2] = Bz_L
	Bz[N//2+2 : N+2] = Bz_R

	E[2 : N+2] = P[2 : N+2]/(gamma - 1.) + 0.5 * rho[2 : N+2] * (vx[2 : N+2]*vx[2 : N+2] + vy[2 : N+2]*vy[2 : N+2] + vz[2 : N+2]*vz[2 : N+2]) + 0.5 * (Bx[2 : N+2]*Bx[2 : N+2] + By[2 : N+2]*By[2 : N+2] + Bz[2 : N+2]*Bz[2 : N+2])

	#initial condition setting for ghost cells
	rho[0:2] = rho[2]
	rho[-2:] = rho[N+1]

	vx[0:2] = vx[2]
	vx[-2:] = vx[N+1]

	vy[0:2] = vy[2]
	vy[-2:] = vy[N+1]

	vz[0:2] = vz[2]
	vz[-2:] = vz[N+1]

	P[0:2] = P[2]
	P[-2:] = P[N+1]

	By[0:2] = By[2]
	By[-2:] = By[N+1]

	Bz[0:2] = Bz[2]
	Bz[-2:] = Bz[N+1]

	E[0:2] = E[2]
	E[-2:] = E[N+1]

	rho_iv[:] = rho[:]
	vx_iv[:] = vx[:]
	vy_iv[:] = vy[:]
	vz_iv[:] = vz[:]
	P_iv[:] = P[:]
	Bx_iv[:] = Bx[:]
	By_iv[:] = By[:]
	Bz_iv[:] = Bz[:]
	E_iv[:] = E[:]
	################################################################
	"""
	Convert all physical quantities into the vector of conserved variables U
	U is a 7 * (N+4) matrix
	"""
	U = np.zeros([7,N+4],float)
	U[0,:] = rho[:]
	U[1,:] = rho[:] * vx[:]
	U[2,:] = rho[:] * vy[:]
	U[3,:] = rho[:] * vz[:]
	U[4,:] = E[:]
	U[5,:] = By[:]
	U[6,:] = Bz[:]

	"""
	make a copy of U for applying Dirichlet B.C. 
	"""
	U_init = np.zeros([7,N+4],float)
	U_init = np.copy(U)

	"""
	Temporary U for updating U using 3rd Runge-Kutta scheme (Shu & Osher)
	"""
	U_temp_1 = np.zeros([7,N+4],float)
	U_temp_2 = np.zeros([7,N+4],float)
	##################################################################

	"""
	Choice of the type of boundary condtion:
	0. Dirichlet
	1. Neumann
	"""
	Boundary_type = float(input("Please choose the type of boundary condition (0 for Dirichlet B.C., 1 for Neumann B.C.): "))
	if Boundary_type == 0:
		print("The Boundary type is Dirichlet B.C.")
	elif Boundary_type == 1:
		print("The Boundary type is Neumann B.C.")
	else:
		print("Please choose '0' or '1'!")

	while t<t_final:
		delta_t = MHD1dSS.GetDt(gamma,N,delta_x,CFL_num,U,Bx)
		#REAL cells are updated using Lu and 3rd-order Runge-Kutta
		#GHOST cells are updated using boundary condition (Dirichlet or Neumann)
		U_temp_1 = U + delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U,Bx)
		U_temp_2 = 3./4. * U + 1./4. * U_temp_1 + 1./4. * delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U_temp_1,Bx)
		U = 1./3. * U + 2./3. * U_temp_2 + 2./3. * delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U_temp_2,Bx) 
		#adjust ghost cells
		if Boundary_type == 0:
			#Dirichlet
			U[:,0] = U_init[:,0]
			U[:,1] = U_init[:,1]
			U[:,-2] = U_init[:,-2]
			U[:,-1] = U_init[:,-1]
		elif Boundary_type == 1:
			#Neumann
			U[:,0] = U[:,2]
			U[:,1] = U[:,2]
			U[:,-2] = U[:,N+1]
			U[:,-1] = U[:,N+1]

		t += delta_t

		if abs(t - t_plot) < 0.5*delta_t:
			print("Plot at time t = ", t)
			rho[:] = U[0,:]
			vx[:] = U[1,:] / U[0,:]
			vy[:] = U[2,:] / U[0,:]
			vz[:] = U[3,:] / U[0,:]
			E[:] = U[4,:]
			By[:] = U[5,:]
			Bz[:] = U[6,:]
			P[:] = (gamma - 1.) * (E[:] - 0.5 * rho[:] * (vx[:]*vx[:] + vy[:]*vy[:] + vz[:]*vz[:]) - 0.5 * (Bx[:]*Bx[:] + By[:]*By[:] + Bz[:]*Bz[:]) )
			
			x_plot = np.empty(N,float)
			for i in range(N):
				x_plot[i] = x_min + (i+0.5) * delta_x

			# add plot
			fig0 = plt.figure()
			ax0=fig0.add_subplot(1,2,1)
			ax0.plot(x_plot[:],rho_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax0.plot(x_plot[:],rho[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax0.set_xlabel(r"$x$",fontsize=24)
			ax0.set_ylabel(r"$\rho$",fontsize=24)
			ax0.set_xlim([x_min,x_max])
			ax0.set_ylim([min(rho[2:N+2])-0.1, max(rho[2:N+2])+0.1])
			ax0.legend()
		

			ax1=fig0.add_subplot(1,2,2)
			ax1.plot(x_plot[:],P_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax1.plot(x_plot[:],P[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax1.set_xlabel(r"$x$",fontsize=24)
			ax1.set_ylabel(r"$P$",fontsize=24)
			ax1.set_xlim([x_min,x_max])
			ax1.set_ylim([min(P[2:N+2])-0.5, max(P[2:N+2])+0.5])
			ax1.legend()

			# ax7=fig0.add_subplot(1,3,3)
			# ax7.plot(x_plot[:],E_iv[2:N+2],"r--",linewidth=2)
			# ax7.plot(x_plot[:],E[2:N+2],"b",linewidth=2)
			# ax7.set_xlabel(r"$x$",fontsize=15)
			# ax7.set_ylabel(r"$E$",fontsize=13)
			# ax7.set_xlim([x_min,x_max])
			# ax7.set_ylim([min(E[2:N+2])-0.1, max(E[2:N+2])+0.1])

			fig1 = plt.figure()
			ax2=fig1.add_subplot(3,2,1)
			ax2.plot(x_plot[:],vx_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax2.plot(x_plot[:],vx[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax2.set_xlabel(r"$x$",fontsize=17)
			ax2.set_ylabel(r"$v_x$",fontsize=17)
			ax2.set_xlim([x_min,x_max])
			ax2.set_ylim([min(vx[2:N+2])-0.1, max(vx[2:N+2])+0.1])
			ax2.legend(loc="lower left")

			ax4=fig1.add_subplot(3,2,2)
			ax4.plot(x_plot[:],vy_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax4.plot(x_plot[:],vy[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax4.set_xlabel(r"$x$",fontsize=17)
			ax4.set_ylabel(r"$v_y$",fontsize=17)
			ax4.set_xlim([x_min,x_max])
			ax4.set_ylim([min(vy[2:N+2])-0.1, max(vy[2:N+2])+0.1])
			ax4.legend(loc="lower left")

			ax8=fig1.add_subplot(3,2,3)
			ax8.plot(x_plot[:],vz_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax8.plot(x_plot[:],vz[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax8.set_xlabel(r"$x$",fontsize=17)
			ax8.set_ylabel(r"$v_z$",fontsize=17)
			ax8.set_xlim([x_min,x_max])
			ax8.set_ylim([min(vz[2:N+2])-0.1, max(vz[2:N+2])+0.1])
			ax8.legend()

			ax5=fig1.add_subplot(3,2,4)
			ax5.plot(x_plot[:],Bx_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax5.plot(x_plot[:],Bx[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax5.set_xlabel(r"$x$",fontsize=17)
			ax5.set_ylabel(r"$B_x$",fontsize=17)
			ax5.set_xlim([x_min,x_max])
			ax5.set_ylim([min(Bx[2:N+2])-0.1, max(Bx[2:N+2])+0.1])
			ax5.legend()

			ax3=fig1.add_subplot(3,2,5)
			ax3.plot(x_plot[:],By_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax3.plot(x_plot[:],By[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax3.set_xlabel(r"$x$",fontsize=17)
			ax3.set_ylabel(r"$B_y$",fontsize=17)
			ax3.set_xlim([x_min,x_max])
			ax3.set_ylim([min(By[2:N+2])-0.1, max(By[2:N+2])+0.1])
			ax3.legend(loc="upper left")

			# fig1 = plt.figure()
			# ax4=fig1.add_subplot(2,2,1)
			# ax4.plot(x_plot[:],vy_iv[2:N+2],"r--",linewidth=2)
			# ax4.plot(x_plot[:],vy[2:N+2],"b",linewidth=2)
			# ax4.set_xlabel(r"$x$",fontsize=24)
			# ax4.set_ylabel(r"$v_y$",fontsize=24)
			# ax4.set_xlim([x_min,x_max])
			# ax4.set_ylim([min(vy[2:N+2])-0.1, max(vy[2:N+2])+0.1])

			# ax5=fig1.add_subplot(2,2,2)
			# ax5.plot(x_plot[:],Bx_iv[2:N+2],"r--",linewidth=2)
			# ax5.plot(x_plot[:],Bx[2:N+2],"b",linewidth=2)
			# ax5.set_xlabel(r"$x$",fontsize=24)
			# ax5.set_ylabel(r"$B_x$",fontsize=24)
			# ax5.set_xlim([x_min,x_max])
			# ax5.set_ylim([min(Bx[2:N+2])-0.1, max(Bx[2:N+2])+0.1])

			ax6=fig1.add_subplot(3,2,6)
			ax6.plot(x_plot[:],Bz_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax6.plot(x_plot[:],Bz[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax6.set_xlabel(r"$x$",fontsize=17)
			ax6.set_ylabel(r"$B_z$",fontsize=17)
			ax6.set_xlim([x_min,x_max])
			ax6.set_ylim([min(Bz[2:N+2])-0.1, max(Bz[2:N+2])+0.1])
			ax6.legend()

			# ax7=fig1.add_subplot(2,2,4)
			# ax7.plot(x_plot[:],E_iv[2:N+2],"r--",linewidth=2)
			# ax7.plot(x_plot[:],E[2:N+2],"b",linewidth=2)
			# ax7.set_xlabel(r"$x$",fontsize=24)
			# ax7.set_ylabel(r"$E$",fontsize=24)
			# ax7.set_xlim([x_min,x_max])
			# ax7.set_ylim([min(E[2:N+2])-0.1, max(E[2:N+2])+0.1])

	print("-------------------Slow Switch-off (SS) shock test DONE!------------------------")
elif Test_type == "FR":
	"""
	Grid (cell) initilization
	"""
	x_min = -0.5
	x_max = 1.5
	N = 400 			#N = 800				#number of REAL cells
	delta_x = (x_max - x_min) / N 	#width of each cell
	########################################################
	"""
	Fast switch-on Shock test********************************************************
	"""
	print("-------------------Fast Rarefaction (FR) waves test------------------------")
	"""
	Basic parameters and constants
	"""
	gamma = 5./3.
	CFL_num = 0.3
	#######################################################
	"""
	Time initialization
	"""
	t_init = 0.		#initial time
	t_final = 0.11	#final time

	t_plot = 0.1	#time to produce the plot
	#########################################################
	"""
	Initial conditions list
	"""
	t = t_init
	#left half region
	rho_L = 1.
	P_L = 2.
	vx_L = 0.
	vy_L = 0.
	vz_L = 0.

	Bx_const = 1.

	By_L = 3.
	Bz_L = 0.

	#right half region
	rho_R = 0.2641
	P_R = 0.2175
	vx_R = 3.6
	vy_R = -2.551
	vz_R = 0.

	Bx_const = 1.

	By_R = 0.
	Bz_R = 0.
	#############################################################
	"""
	Vector initialization and initial condition setting
	"""
	rho = np.zeros(N+4,float)			#rho is the array of the density
	vx = np.zeros(N+4,float)			#vx is the array of the x-comp velocity
	vy = np.zeros(N+4,float)			#vy is the array of the y-comp velocity
	vz = np.zeros(N+4,float)			#vz is the array of the z-comp velocity
	P = np.zeros(N+4,float)				#P is the array of the pressure
	Bx = np.zeros(N+4,float)			#Bx is the array of x-comp of magnetic field
	By = np.zeros(N+4,float)			#By is the array of y-comp of magnetic field
	Bz = np.zeros(N+4,float)			#Bz is the array of z-comp of magnetic field
	E = np.empty(N+4,float)				#E is the array of the Energy density

	#initial values
	rho_iv = np.zeros(N+4,float)
	vx_iv = np.zeros(N+4,float)
	vy_iv = np.zeros(N+4,float)
	vz_iv = np.zeros(N+4,float)
	P_iv = np.zeros(N+4,float)
	Bx_iv = np.zeros(N+4,float)
	By_iv = np.zeros(N+4,float)
	Bz_iv = np.zeros(N+4,float)
	E_iv = np.zeros(N+4,float)

	#initial condtion setting for real cells
	rho[2 : N//2+2] = rho_L
	rho[N//2+2 : N+2] = rho_R

	vx[2 : N//2+2] = vx_L
	vx[N//2+2 : N+2] = vx_R
	vy[2 : N//2+2] = vy_L
	vy[N//2+2 : N+2] = vy_R
	vz[2 : N//2+2] = vz_L
	vz[N//2+2 : N+2] = vz_R

	P[2 : N//2+2] = P_L
	P[N//2+2 : N+2] = P_R

	Bx[:] = Bx_const

	By[2 : N//2+2] = By_L
	By[N//2+2 : N+2] = By_R
	Bz[2 : N//2+2] = Bz_L
	Bz[N//2+2 : N+2] = Bz_R

	E[2 : N+2] = P[2 : N+2]/(gamma - 1.) + 0.5 * rho[2 : N+2] * (vx[2 : N+2]*vx[2 : N+2] + vy[2 : N+2]*vy[2 : N+2] + vz[2 : N+2]*vz[2 : N+2]) + 0.5 * (Bx[2 : N+2]*Bx[2 : N+2] + By[2 : N+2]*By[2 : N+2] + Bz[2 : N+2]*Bz[2 : N+2])

	#initial condition setting for ghost cells
	rho[0:2] = rho[2]
	rho[-2:] = rho[N+1]

	vx[0:2] = vx[2]
	vx[-2:] = vx[N+1]

	vy[0:2] = vy[2]
	vy[-2:] = vy[N+1]

	vz[0:2] = vz[2]
	vz[-2:] = vz[N+1]

	P[0:2] = P[2]
	P[-2:] = P[N+1]

	By[0:2] = By[2]
	By[-2:] = By[N+1]

	Bz[0:2] = Bz[2]
	Bz[-2:] = Bz[N+1]

	E[0:2] = E[2]
	E[-2:] = E[N+1]

	rho_iv[:] = rho[:]
	vx_iv[:] = vx[:]
	vy_iv[:] = vy[:]
	vz_iv[:] = vz[:]
	P_iv[:] = P[:]
	Bx_iv[:] = Bx[:]
	By_iv[:] = By[:]
	Bz_iv[:] = Bz[:]
	E_iv[:] = E[:]
	################################################################
	"""
	Convert all physical quantities into the vector of conserved variables U
	U is a 7 * (N+4) matrix
	"""
	U = np.zeros([7,N+4],float)
	U[0,:] = rho[:]
	U[1,:] = rho[:] * vx[:]
	U[2,:] = rho[:] * vy[:]
	U[3,:] = rho[:] * vz[:]
	U[4,:] = E[:]
	U[5,:] = By[:]
	U[6,:] = Bz[:]

	"""
	make a copy of U for applying Dirichlet B.C. 
	"""
	U_init = np.zeros([7,N+4],float)
	U_init = np.copy(U)

	"""
	Temporary U for updating U using 3rd Runge-Kutta scheme (Shu & Osher)
	"""
	U_temp_1 = np.zeros([7,N+4],float)
	U_temp_2 = np.zeros([7,N+4],float)
	##################################################################

	"""
	Choice of the type of boundary condtion:
	0. Dirichlet
	1. Neumann
	"""
	Boundary_type = float(input("Please choose the type of boundary condition (0 for Dirichlet B.C., 1 for Neumann B.C.): "))
	if Boundary_type == 0:
		print("The Boundary type is Dirichlet B.C.")
	elif Boundary_type == 1:
		print("The Boundary type is Neumann B.C.")
	else:
		print("Please choose '0' or '1'!")

	while t<t_final:
		delta_t = MHD1dSS.GetDt(gamma,N,delta_x,CFL_num,U,Bx)
		#REAL cells are updated using Lu and 3rd-order Runge-Kutta
		#GHOST cells are updated using boundary condition (Dirichlet or Neumann)
		U_temp_1 = U + delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U,Bx)
		U_temp_2 = 3./4. * U + 1./4. * U_temp_1 + 1./4. * delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U_temp_1,Bx)
		U = 1./3. * U + 2./3. * U_temp_2 + 2./3. * delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U_temp_2,Bx) 
		#adjust ghost cells
		if Boundary_type == 0:
			#Dirichlet
			U[:,0] = U_init[:,0]
			U[:,1] = U_init[:,1]
			U[:,-2] = U_init[:,-2]
			U[:,-1] = U_init[:,-1]
		elif Boundary_type == 1:
			#Neumann
			U[:,0] = U[:,2]
			U[:,1] = U[:,2]
			U[:,-2] = U[:,N+1]
			U[:,-1] = U[:,N+1]

		t += delta_t

		if abs(t - t_plot) < 0.5*delta_t:
			print("Plot at time t = ", t)
			rho[:] = U[0,:]
			vx[:] = U[1,:] / U[0,:]
			vy[:] = U[2,:] / U[0,:]
			vz[:] = U[3,:] / U[0,:]
			E[:] = U[4,:]
			By[:] = U[5,:]
			Bz[:] = U[6,:]
			P[:] = (gamma - 1.) * (E[:] - 0.5 * rho[:] * (vx[:]*vx[:] + vy[:]*vy[:] + vz[:]*vz[:]) - 0.5 * (Bx[:]*Bx[:] + By[:]*By[:] + Bz[:]*Bz[:]) )
			
			x_plot = np.empty(N,float)
			for i in range(N):
				x_plot[i] = x_min + (i+0.5) * delta_x

			# add plot
			fig0 = plt.figure()
			ax0=fig0.add_subplot(1,2,1)
			ax0.plot(x_plot[:],rho_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax0.plot(x_plot[:],rho[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax0.set_xlabel(r"$x$",fontsize=24)
			ax0.set_ylabel(r"$\rho$",fontsize=24)
			ax0.set_xlim([x_min,x_max])
			ax0.set_ylim([min(rho[2:N+2])-0.1, max(rho[2:N+2])+0.1])
			ax0.legend()
		

			ax1=fig0.add_subplot(1,2,2)
			ax1.plot(x_plot[:],P_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax1.plot(x_plot[:],P[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax1.set_xlabel(r"$x$",fontsize=24)
			ax1.set_ylabel(r"$P$",fontsize=24)
			ax1.set_xlim([x_min,x_max])
			ax1.set_ylim([min(P[2:N+2])-0.5, max(P[2:N+2])+0.5])
			ax1.legend()

			# ax7=fig0.add_subplot(1,3,3)
			# ax7.plot(x_plot[:],E_iv[2:N+2],"r--",linewidth=2)
			# ax7.plot(x_plot[:],E[2:N+2],"b",linewidth=2)
			# ax7.set_xlabel(r"$x$",fontsize=15)
			# ax7.set_ylabel(r"$E$",fontsize=13)
			# ax7.set_xlim([x_min,x_max])
			# ax7.set_ylim([min(E[2:N+2])-0.1, max(E[2:N+2])+0.1])

			fig1 = plt.figure()
			ax2=fig1.add_subplot(3,2,1)
			ax2.plot(x_plot[:],vx_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax2.plot(x_plot[:],vx[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax2.set_xlabel(r"$x$",fontsize=17)
			ax2.set_ylabel(r"$v_x$",fontsize=17)
			ax2.set_xlim([x_min,x_max])
			ax2.set_ylim([min(vx[2:N+2])-0.1, max(vx[2:N+2])+0.1])
			ax2.legend()

			ax4=fig1.add_subplot(3,2,2)
			ax4.plot(x_plot[:],vy_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax4.plot(x_plot[:],vy[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax4.set_xlabel(r"$x$",fontsize=17)
			ax4.set_ylabel(r"$v_y$",fontsize=17)
			ax4.set_xlim([x_min,x_max])
			ax4.set_ylim([min(vy[2:N+2])-0.1, max(vy[2:N+2])+0.1])
			ax4.legend()

			ax8=fig1.add_subplot(3,2,3)
			ax8.plot(x_plot[:],vz_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax8.plot(x_plot[:],vz[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax8.set_xlabel(r"$x$",fontsize=17)
			ax8.set_ylabel(r"$v_z$",fontsize=17)
			ax8.set_xlim([x_min,x_max])
			ax8.set_ylim([min(vz[2:N+2])-0.1, max(vz[2:N+2])+0.1])
			ax8.legend()

			ax5=fig1.add_subplot(3,2,4)
			ax5.plot(x_plot[:],Bx_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax5.plot(x_plot[:],Bx[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax5.set_xlabel(r"$x$",fontsize=17)
			ax5.set_ylabel(r"$B_x$",fontsize=17)
			ax5.set_xlim([x_min,x_max])
			ax5.set_ylim([min(Bx[2:N+2])-0.1, max(Bx[2:N+2])+0.1])
			ax5.legend()

			ax3=fig1.add_subplot(3,2,5)
			ax3.plot(x_plot[:],By_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax3.plot(x_plot[:],By[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax3.set_xlabel(r"$x$",fontsize=17)
			ax3.set_ylabel(r"$B_y$",fontsize=17)
			ax3.set_xlim([x_min,x_max])
			ax3.set_ylim([min(By[2:N+2])-0.1, max(By[2:N+2])+0.1])
			ax3.legend()

			# fig1 = plt.figure()
			# ax4=fig1.add_subplot(2,2,1)
			# ax4.plot(x_plot[:],vy_iv[2:N+2],"r--",linewidth=2)
			# ax4.plot(x_plot[:],vy[2:N+2],"b",linewidth=2)
			# ax4.set_xlabel(r"$x$",fontsize=24)
			# ax4.set_ylabel(r"$v_y$",fontsize=24)
			# ax4.set_xlim([x_min,x_max])
			# ax4.set_ylim([min(vy[2:N+2])-0.1, max(vy[2:N+2])+0.1])

			# ax5=fig1.add_subplot(2,2,2)
			# ax5.plot(x_plot[:],Bx_iv[2:N+2],"r--",linewidth=2)
			# ax5.plot(x_plot[:],Bx[2:N+2],"b",linewidth=2)
			# ax5.set_xlabel(r"$x$",fontsize=24)
			# ax5.set_ylabel(r"$B_x$",fontsize=24)
			# ax5.set_xlim([x_min,x_max])
			# ax5.set_ylim([min(Bx[2:N+2])-0.1, max(Bx[2:N+2])+0.1])

			ax6=fig1.add_subplot(3,2,6)
			ax6.plot(x_plot[:],Bz_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax6.plot(x_plot[:],Bz[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax6.set_xlabel(r"$x$",fontsize=17)
			ax6.set_ylabel(r"$B_z$",fontsize=17)
			ax6.set_xlim([x_min,x_max])
			ax6.set_ylim([min(Bz[2:N+2])-0.1, max(Bz[2:N+2])+0.1])
			ax6.legend()

			# ax7=fig1.add_subplot(2,2,4)
			# ax7.plot(x_plot[:],E_iv[2:N+2],"r--",linewidth=2)
			# ax7.plot(x_plot[:],E[2:N+2],"b",linewidth=2)
			# ax7.set_xlabel(r"$x$",fontsize=24)
			# ax7.set_ylabel(r"$E$",fontsize=24)
			# ax7.set_xlim([x_min,x_max])
			# ax7.set_ylim([min(E[2:N+2])-0.1, max(E[2:N+2])+0.1])

	print("-------------------Fast Rarefaction (FR) waves test DONE!------------------------")
elif Test_type == "SR":
	"""
	Grid (cell) initilization
	"""
	x_min = -0.5
	x_max = 1.5
	N = 400 			#N = 800				#number of REAL cells
	delta_x = (x_max - x_min) / N 	#width of each cell
	########################################################
	"""
	Fast switch-on Shock test********************************************************
	"""
	print("-------------------Slow Rarefaction (SR) waves test------------------------")
	"""
	Basic parameters and constants
	"""
	gamma = 5./3.
	CFL_num = 0.3
	#######################################################
	"""
	Time initialization
	"""
	t_init = 0.		#initial time
	t_final = 0.31	#final time

	t_plot = 0.3	#time to produce the plot
	#########################################################
	"""
	Initial conditions list
	"""
	t = t_init
	#left half region
	rho_L = 1.
	P_L = 2.
	vx_L = 0.
	vy_L = 0.
	vz_L = 0.

	Bx_const = 1.

	By_L = 0.
	Bz_L = 0.

	#right half region
	rho_R = 0.2
	P_R = 0.1368
	vx_R = 1.186
	vy_R = 2.967
	vz_R = 0.

	Bx_const = 1.

	By_R = 1.6405
	Bz_R = 0.
	#############################################################
	"""
	Vector initialization and initial condition setting
	"""
	rho = np.zeros(N+4,float)			#rho is the array of the density
	vx = np.zeros(N+4,float)			#vx is the array of the x-comp velocity
	vy = np.zeros(N+4,float)			#vy is the array of the y-comp velocity
	vz = np.zeros(N+4,float)			#vz is the array of the z-comp velocity
	P = np.zeros(N+4,float)				#P is the array of the pressure
	Bx = np.zeros(N+4,float)			#Bx is the array of x-comp of magnetic field
	By = np.zeros(N+4,float)			#By is the array of y-comp of magnetic field
	Bz = np.zeros(N+4,float)			#Bz is the array of z-comp of magnetic field
	E = np.empty(N+4,float)				#E is the array of the Energy density

	#initial values
	rho_iv = np.zeros(N+4,float)
	vx_iv = np.zeros(N+4,float)
	vy_iv = np.zeros(N+4,float)
	vz_iv = np.zeros(N+4,float)
	P_iv = np.zeros(N+4,float)
	Bx_iv = np.zeros(N+4,float)
	By_iv = np.zeros(N+4,float)
	Bz_iv = np.zeros(N+4,float)
	E_iv = np.zeros(N+4,float)

	#initial condtion setting for real cells
	rho[2 : N//2+2] = rho_L
	rho[N//2+2 : N+2] = rho_R

	vx[2 : N//2+2] = vx_L
	vx[N//2+2 : N+2] = vx_R
	vy[2 : N//2+2] = vy_L
	vy[N//2+2 : N+2] = vy_R
	vz[2 : N//2+2] = vz_L
	vz[N//2+2 : N+2] = vz_R

	P[2 : N//2+2] = P_L
	P[N//2+2 : N+2] = P_R

	Bx[:] = Bx_const

	By[2 : N//2+2] = By_L
	By[N//2+2 : N+2] = By_R
	Bz[2 : N//2+2] = Bz_L
	Bz[N//2+2 : N+2] = Bz_R

	E[2 : N+2] = P[2 : N+2]/(gamma - 1.) + 0.5 * rho[2 : N+2] * (vx[2 : N+2]*vx[2 : N+2] + vy[2 : N+2]*vy[2 : N+2] + vz[2 : N+2]*vz[2 : N+2]) + 0.5 * (Bx[2 : N+2]*Bx[2 : N+2] + By[2 : N+2]*By[2 : N+2] + Bz[2 : N+2]*Bz[2 : N+2])

	#initial condition setting for ghost cells
	rho[0:2] = rho[2]
	rho[-2:] = rho[N+1]

	vx[0:2] = vx[2]
	vx[-2:] = vx[N+1]

	vy[0:2] = vy[2]
	vy[-2:] = vy[N+1]

	vz[0:2] = vz[2]
	vz[-2:] = vz[N+1]

	P[0:2] = P[2]
	P[-2:] = P[N+1]

	By[0:2] = By[2]
	By[-2:] = By[N+1]

	Bz[0:2] = Bz[2]
	Bz[-2:] = Bz[N+1]

	E[0:2] = E[2]
	E[-2:] = E[N+1]

	rho_iv[:] = rho[:]
	vx_iv[:] = vx[:]
	vy_iv[:] = vy[:]
	vz_iv[:] = vz[:]
	P_iv[:] = P[:]
	Bx_iv[:] = Bx[:]
	By_iv[:] = By[:]
	Bz_iv[:] = Bz[:]
	E_iv[:] = E[:]
	################################################################
	"""
	Convert all physical quantities into the vector of conserved variables U
	U is a 7 * (N+4) matrix
	"""
	U = np.zeros([7,N+4],float)
	U[0,:] = rho[:]
	U[1,:] = rho[:] * vx[:]
	U[2,:] = rho[:] * vy[:]
	U[3,:] = rho[:] * vz[:]
	U[4,:] = E[:]
	U[5,:] = By[:]
	U[6,:] = Bz[:]

	"""
	make a copy of U for applying Dirichlet B.C. 
	"""
	U_init = np.zeros([7,N+4],float)
	U_init = np.copy(U)

	"""
	Temporary U for updating U using 3rd Runge-Kutta scheme (Shu & Osher)
	"""
	U_temp_1 = np.zeros([7,N+4],float)
	U_temp_2 = np.zeros([7,N+4],float)
	##################################################################

	"""
	Choice of the type of boundary condtion:
	0. Dirichlet
	1. Neumann
	"""
	Boundary_type = float(input("Please choose the type of boundary condition (0 for Dirichlet B.C., 1 for Neumann B.C.): "))
	if Boundary_type == 0:
		print("The Boundary type is Dirichlet B.C.")
	elif Boundary_type == 1:
		print("The Boundary type is Neumann B.C.")
	else:
		print("Please choose '0' or '1'!")

	while t<t_final:
		delta_t = MHD1dSS.GetDt(gamma,N,delta_x,CFL_num,U,Bx)
		#REAL cells are updated using Lu and 3rd-order Runge-Kutta
		#GHOST cells are updated using boundary condition (Dirichlet or Neumann)
		U_temp_1 = U + delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U,Bx)
		U_temp_2 = 3./4. * U + 1./4. * U_temp_1 + 1./4. * delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U_temp_1,Bx)
		U = 1./3. * U + 2./3. * U_temp_2 + 2./3. * delta_t * MHD1dSS.Lu_PLM_HLLD(gamma,N,delta_x,U_temp_2,Bx) 
		#adjust ghost cells
		if Boundary_type == 0:
			#Dirichlet
			U[:,0] = U_init[:,0]
			U[:,1] = U_init[:,1]
			U[:,-2] = U_init[:,-2]
			U[:,-1] = U_init[:,-1]
		elif Boundary_type == 1:
			#Neumann
			U[:,0] = U[:,2]
			U[:,1] = U[:,2]
			U[:,-2] = U[:,N+1]
			U[:,-1] = U[:,N+1]

		t += delta_t

		if abs(t - t_plot) < 0.5*delta_t:
			print("Plot at time t = ", t)
			rho[:] = U[0,:]
			vx[:] = U[1,:] / U[0,:]
			vy[:] = U[2,:] / U[0,:]
			vz[:] = U[3,:] / U[0,:]
			E[:] = U[4,:]
			By[:] = U[5,:]
			Bz[:] = U[6,:]
			P[:] = (gamma - 1.) * (E[:] - 0.5 * rho[:] * (vx[:]*vx[:] + vy[:]*vy[:] + vz[:]*vz[:]) - 0.5 * (Bx[:]*Bx[:] + By[:]*By[:] + Bz[:]*Bz[:]) )
			
			x_plot = np.empty(N,float)
			for i in range(N):
				x_plot[i] = x_min + (i+0.5) * delta_x

			# add plot
			fig0 = plt.figure()
			ax0=fig0.add_subplot(1,2,1)
			ax0.plot(x_plot[:],rho_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax0.plot(x_plot[:],rho[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax0.set_xlabel(r"$x$",fontsize=24)
			ax0.set_ylabel(r"$\rho$",fontsize=24)
			ax0.set_xlim([x_min,x_max])
			ax0.set_ylim([min(rho[2:N+2])-0.1, max(rho[2:N+2])+0.1])
			ax0.legend()
		

			ax1=fig0.add_subplot(1,2,2)
			ax1.plot(x_plot[:],P_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax1.plot(x_plot[:],P[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax1.set_xlabel(r"$x$",fontsize=24)
			ax1.set_ylabel(r"$P$",fontsize=24)
			ax1.set_xlim([x_min,x_max])
			ax1.set_ylim([min(P[2:N+2])-0.5, max(P[2:N+2])+0.5])
			ax1.legend()

			# ax7=fig0.add_subplot(1,3,3)
			# ax7.plot(x_plot[:],E_iv[2:N+2],"r--",linewidth=2)
			# ax7.plot(x_plot[:],E[2:N+2],"b",linewidth=2)
			# ax7.set_xlabel(r"$x$",fontsize=15)
			# ax7.set_ylabel(r"$E$",fontsize=13)
			# ax7.set_xlim([x_min,x_max])
			# ax7.set_ylim([min(E[2:N+2])-0.1, max(E[2:N+2])+0.1])

			fig1 = plt.figure()
			ax2=fig1.add_subplot(3,2,1)
			ax2.plot(x_plot[:],vx_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax2.plot(x_plot[:],vx[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax2.set_xlabel(r"$x$",fontsize=17)
			ax2.set_ylabel(r"$v_x$",fontsize=17)
			ax2.set_xlim([x_min,x_max])
			ax2.set_ylim([min(vx[2:N+2])-0.1, max(vx[2:N+2])+0.1])
			ax2.legend(loc="upper left")

			ax4=fig1.add_subplot(3,2,2)
			ax4.plot(x_plot[:],vy_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax4.plot(x_plot[:],vy[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax4.set_xlabel(r"$x$",fontsize=17)
			ax4.set_ylabel(r"$v_y$",fontsize=17)
			ax4.set_xlim([x_min,x_max])
			ax4.set_ylim([min(vy[2:N+2])-0.1, max(vy[2:N+2])+0.1])
			ax4.legend(loc="upper left")

			ax8=fig1.add_subplot(3,2,3)
			ax8.plot(x_plot[:],vz_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax8.plot(x_plot[:],vz[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax8.set_xlabel(r"$x$",fontsize=17)
			ax8.set_ylabel(r"$v_z$",fontsize=17)
			ax8.set_xlim([x_min,x_max])
			ax8.set_ylim([min(vz[2:N+2])-0.1, max(vz[2:N+2])+0.1])
			ax8.legend()

			ax5=fig1.add_subplot(3,2,4)
			ax5.plot(x_plot[:],Bx_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax5.plot(x_plot[:],Bx[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax5.set_xlabel(r"$x$",fontsize=17)
			ax5.set_ylabel(r"$B_x$",fontsize=17)
			ax5.set_xlim([x_min,x_max])
			ax5.set_ylim([min(Bx[2:N+2])-0.1, max(Bx[2:N+2])+0.1])
			ax5.legend()

			ax3=fig1.add_subplot(3,2,5)
			ax3.plot(x_plot[:],By_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax3.plot(x_plot[:],By[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax3.set_xlabel(r"$x$",fontsize=17)
			ax3.set_ylabel(r"$B_y$",fontsize=17)
			ax3.set_xlim([x_min,x_max])
			ax3.set_ylim([min(By[2:N+2])-0.1, max(By[2:N+2])+0.1])
			ax3.legend(loc="upper left")

			# fig1 = plt.figure()
			# ax4=fig1.add_subplot(2,2,1)
			# ax4.plot(x_plot[:],vy_iv[2:N+2],"r--",linewidth=2)
			# ax4.plot(x_plot[:],vy[2:N+2],"b",linewidth=2)
			# ax4.set_xlabel(r"$x$",fontsize=24)
			# ax4.set_ylabel(r"$v_y$",fontsize=24)
			# ax4.set_xlim([x_min,x_max])
			# ax4.set_ylim([min(vy[2:N+2])-0.1, max(vy[2:N+2])+0.1])

			# ax5=fig1.add_subplot(2,2,2)
			# ax5.plot(x_plot[:],Bx_iv[2:N+2],"r--",linewidth=2)
			# ax5.plot(x_plot[:],Bx[2:N+2],"b",linewidth=2)
			# ax5.set_xlabel(r"$x$",fontsize=24)
			# ax5.set_ylabel(r"$B_x$",fontsize=24)
			# ax5.set_xlim([x_min,x_max])
			# ax5.set_ylim([min(Bx[2:N+2])-0.1, max(Bx[2:N+2])+0.1])

			ax6=fig1.add_subplot(3,2,6)
			ax6.plot(x_plot[:],Bz_iv[2:N+2],"r--",linewidth=2,label="t = 0")
			ax6.plot(x_plot[:],Bz[2:N+2],"b",linewidth=2,label="t = {:.2f}".format(t))
			ax6.set_xlabel(r"$x$",fontsize=17)
			ax6.set_ylabel(r"$B_z$",fontsize=17)
			ax6.set_xlim([x_min,x_max])
			ax6.set_ylim([min(Bz[2:N+2])-0.1, max(Bz[2:N+2])+0.1])
			ax6.legend()

			# ax7=fig1.add_subplot(2,2,4)
			# ax7.plot(x_plot[:],E_iv[2:N+2],"r--",linewidth=2)
			# ax7.plot(x_plot[:],E[2:N+2],"b",linewidth=2)
			# ax7.set_xlabel(r"$x$",fontsize=24)
			# ax7.set_ylabel(r"$E$",fontsize=24)
			# ax7.set_xlim([x_min,x_max])
			# ax7.set_ylim([min(E[2:N+2])-0.1, max(E[2:N+2])+0.1])

	print("-------------------Slow Rarefaction (SR) waves test DONE!------------------------")
plt.show()



























