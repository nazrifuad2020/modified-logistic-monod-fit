# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:09:46 2020

@author: nazri
"""

from casadi import MX, vertcat, Function, collocation_points , nlpsol, integrator

import numpy as np

from pyDOE import lhs

import matplotlib.pyplot as plt

tspan = [0,18.25,40.5,51.75,64.75,76,89,100.5,113,124.5,136,148,164.5,191.75] # in hr
Xv_exp = np.array([0.592361493,0.777786656,1.120982789,1.318994188,1.60900909,
                   1.982518497,2.39706606,2.94009058,3.438177052,3.595593515,
                   3.738473677,3.529648804,3.393770758,2.696653628]) # viable cell concentration, 10^6 cells/mL
Xd_exp = np.array([0.015599827,0.034745077,0.04804039,0.056017558,0.073390086,
                   0.067717473,0.099980712,0.128299713,0.112478296,0.156175454,
                   0.192072789,0.302335143,0.544575572,1.064067523]) # dead cell concentration, 10^6 cells/mL
S_exp = np.array([0.030925683,0.030021789,0.024895399,0.025254246,0.023357482,
                  0.018948787,0.015975481,0.013566077,0.007670729,0.001570325,
                  8.37E-05,8.37E-05,8.37E-05,8.37E-05,]) # growth limiting subs. conc. (M), here Glc
P1_exp = np.array([0,9.04,23.5995,31.80833333,45.22,51.47,67.19,70.47,91.32,
                   103.8,107.46,127.22,128.47,115.6]) # AAT concentration (M)

#model parameters in log10 transformed values
logmumax = MX.sym('logmumax') 
logKs = MX.sym('logKs')
logalpha = MX.sym('logalpha')
logKd = MX.sym('logKd')
logk = MX.sym('logk')
logbheta = MX.sym('logbheta')
logKlys = MX.sym('logKlys')
logYsx = MX.sym('logYsx')
logYp1x = MX.sym('logYp1x')
pars = [logmumax,logKs,logalpha,logKd,logk,logbheta,logKlys,logYsx,logYp1x]
npars = len(pars)

#model states
Xv = MX.sym('Xv')
Xd = MX.sym('Xd')
S = MX.sym('S')
P1 = MX.sym('P1')
y = vertcat(Xv,Xd,S,P1)
nstates = y.size()[0]

#create the model using casadi's symbolic expression
mumax = 10**logmumax
Ks = 10**logKs
alpha = 10**logalpha # alpha = 1/Xvmax
Kd = 10**logKd
mu = mumax*S/(S+Ks)*(1-alpha*Xv)
dXv = (mu-Kd)*Xv

k = 10**logk
bheta = 10**logbheta # bheta=1/Xmax
Klys = 10**logKlys
Xt = Xv+Xd
dXd = k*Xv*(1-bheta*Xt)-dXv-Klys*Xd

Ysx = 10**logYsx #Ysx = 1/Yxs
dS = -Ysx*mu*Xv

Yp1x = 10**logYp1x
dP1 = Yp1x*mu*Xv

dy = vertcat(dXv,dXd,dS,dP1)

f_ode = Function('f_ode',[y],[dy],['y'],['dy'])

# Start with empty NLP
w=[]
lbw = []
ubw = []
g=[]
lbg = []
ubg = []

wX = []
x0 = []
lbwX = []
ubwX = []

#Define states at collocation points in log10 transformed values: Xk=log10(yk)
Xk = MX.sym('X0', nstates)
wX += [Xk]    
lbstates = [np.floor(np.log10(min(Xv_exp))), np.floor(np.log10(min(Xd_exp))), 
            np.floor(np.log10(min(S_exp))), -24]
ubstates = [np.ceil(np.log10(max(Xv_exp))), np.ceil(np.log10(max(Xd_exp))), 
            np.ceil(np.log10(max(S_exp))), np.ceil(np.log10(max(P1_exp)))]
meanstates = [np.log10(np.mean(Xv_exp)), np.log10(np.mean(Xd_exp)), 
              np.log10(np.mean(S_exp)), np.log10(np.mean(P1_exp))]
lbwX += lbstates 
ubwX += ubstates
x0 += meanstates

# revert the states at collocation points back to normal values and calculate SSE
SSE = (((Xv_exp[0]-10**Xk[0])/max(Xv_exp))**2 
       + ((Xd_exp[0]-10**Xk[1])/max(Xd_exp))**2 
       + ((S_exp[0]-10**Xk[2])/max(S_exp))**2
       + ((P1_exp[0]-10**Xk[3])/max(P1_exp))**2)

# Degree of interpolating polynomial
d = 3
# Get collocation points
tau_root = np.append(0, collocation_points(d, 'legendre'))
# Coefficients of the collocation equation
C = np.zeros((d+1,d+1))
# Coefficients of the continuity equation
D = np.zeros(d+1)
# Coefficients of the quadrature function
B = np.zeros(d+1)
# Construct polynomial basis
for j in range(d+1):
    # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    p = np.poly1d([1])
    for r in range(d+1):
        if r != j:
            p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])
    # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    D[j] = p(1.0)
    # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    pder = np.polyder(p)
    for r in range(d+1):
        C[j,r] = pder(tau_root[r])
    # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
    pint = np.polyint(p)
    B[j] = pint(1.0)
    
N = len(tspan)
for k in range(1,N):
    # State at collocation points
    Xc = []
    for j in range(d):
        Xkj = MX.sym('X_'+str(k)+'_'+str(j), nstates)
        Xc += [Xkj]
        wX += [Xkj]
        lbwX += lbstates
        ubwX += ubstates
        x0 += meanstates
        
    # Loop over collocation points
    Xk_end = D[0]*10**Xk
    h = tspan[k]-tspan[k-1]
    for j in range(1,d+1):
        # Expression for the state derivative at the collocation point
        xp = C[0,j]*10**Xk
        for r in range(d): 
            xp = xp + C[r+1,j]*10**Xc[r]

        # Append collocation equations
        fj = f_ode(10**Xc[j-1])
        g += [h*fj - xp]
        lbg += [0]*nstates
        ubg += [0]*nstates

        # Add contribution to the end state
        Xk_end = Xk_end + D[j]*10**Xc[j-1]
            
    # New NLP variable for state at end of interval
    Xk = MX.sym('X_' + str(k), nstates)
    wX += [Xk]
    lbwX += lbstates
    ubwX += ubstates
    x0 += meanstates
        
    # Add equality constraint
    g += [Xk_end-10**Xk]
    lbg += [0]*nstates
    ubg += [0]*nstates  
    
    # calculate SSE for states at end of interval
    SSE += (((Xv_exp[k]-10**Xk[0])/max(Xv_exp))**2 
       + ((Xd_exp[k]-10**Xk[1])/max(Xd_exp))**2 
       + ((S_exp[k]-10**Xk[2])/max(S_exp))**2
       + ((P1_exp[k]-10**Xk[3])/max(P1_exp))**2)
    
# Create an NLP solver for LSE   
w = pars + wX
lbw = [-12]*npars + lbwX
ubw = [2]*npars + ubwX
prob = {'f': SSE, 'x': vertcat(*w), 'g': vertcat(*g)}
#nlpopt = {'ipopt': {'max_iter': 120}}
solver = nlpsol('solver','ipopt',prob)

# generate 20 random trials for NLP runs
init_pars = lhs(npars, samples=20, criterion='maximin')*(2+12)-12
SSEs = []
solxs = []
for tr in range(20):
    w0 = init_pars[tr,:].tolist() + x0
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    
    if solver.stats()['success']:
        SSEs += [float(sol['f'])]
        solxs += [sol['x'].full().flatten()]
        
# find the result of the best run
best_index = SSEs.index(min(SSEs))
w_opt = solxs[best_index]
best_SSE = SSEs[best_index]
DF = (N-1)*nstates-npars
MSE = best_SSE/DF
pars_opt = w_opt[:npars]
Xcol = w_opt[npars:]
X0 = 10**Xcol[:nstates]
Xv_col = 10**Xcol[0::nstates][0::d+1]
Xd_col = 10**Xcol[1::nstates][0::d+1]
S_col = 10**Xcol[2::nstates][0::d+1]
P1_col = 10**Xcol[3::nstates][0::d+1]

# statistical inferencing
G = [np.zeros((nstates,npars))] # sensitivity matrix at each time point
Cfac = [1/max(Xv_exp),1/max(Xd_exp),1/max(S_exp),1/max(P1_exp)]
Cfac = np.diag(Cfac)

Xv_pred = np.zeros(N)
Xd_pred = np.zeros(N)
S_pred = np.zeros(N)
P1_pred = np.zeros(N)

CovarMeas = MSE*np.eye(nstates)
Qmat = np.linalg.inv(CovarMeas)
Amat = G[0].T@Cfac.T@Qmat@Cfac@G[0]

Xv_pred[0] = X0[0]
Xd_pred[0] = X0[1]
S_pred[0] = X0[2]
P1_pred[0] = X0[3]

ode_model = {'x':y, 'p':vertcat(*pars), 'ode':dy}
for k in range(1,N):
    opts = {'tf':tspan[k]-tspan[0]}
    I = integrator('I', 'cvodes', ode_model, opts)
    Ik = I(x0=X0, p=pars_opt)
    Xk = Ik['xf'].full().flatten()
    
    Xv_pred[k] = Xk[0]
    Xd_pred[k] = Xk[1]
    S_pred[k] = Xk[2]
    P1_pred[k] = Xk[3]
    
    I_jac = I.factory("I_jac", ["x0", "p"], ["jac:xf:p"])
    G += [I_jac(X0, pars_opt).full()]
    Amat += G[-1].T@Cfac.T@Qmat@Cfac@G[-1]
    
# parameter covariance matrix, the square root of its main diagonal gives
# parameter estimation errors
CovPar = np.linalg.inv(Amat)
pars_sig = np.sqrt(np.diag(CovPar))

from scipy.special import stdtrit
alpha = 0.05
tv = stdtrit(DF, 1-alpha/2)

pars_ub = pars_opt + tv*pars_sig
pars_lb = pars_opt - tv*pars_sig

pars_opt_ori = 10**pars_opt
pars_ub_ori = 10**pars_ub
pars_lb_ori = 10**pars_lb

import pandas as pd

pars_list = []
for par in pars:
    pars_list += [par.str()[3:]]
par_est_rslt = pd.DataFrame({'Parameter': pars_list,
 'Value': pars_opt_ori,
 'Lower bound': pars_lb_ori,
 'Upper bound': pars_ub_ori})

print('\n')
print(par_est_rslt)

# plot the model simulation
fig, axs = plt.subplots(nrows=3, ncols=2, constrained_layout=True)

axs[0,0].set_xlabel('Time (hr)')
axs[0,0].set_ylabel('10^6 cells/mL')
axs[0,0].set_title('Viable cells')
axs[0,0].scatter(tspan,Xv_exp,label='Data',color='red')
axs[0,0].plot(tspan,Xv_pred,label='Model sim.',color='blue')
axs[0,0].legend()

axs[0,1].set_xlabel('Time (hr)')
axs[0,1].set_ylabel('10^6 cells/mL')
axs[0,1].set_title('Dead cells')
axs[0,1].scatter(tspan,Xd_exp,label='Data',color='red')
axs[0,1].plot(tspan,Xd_pred,label='Model sim.',color='blue')
axs[0,1].legend()

axs[1,0].set_xlabel('Time (hr)')
axs[1,0].set_ylabel('mmol/mL')
axs[1,0].set_title('Glucose')
axs[1,0].plot(tspan,S_pred,label='Model sim.',color='blue')
axs[1,0].scatter(tspan,S_exp,label='Data',color='red')
axs[1,0].legend()

axs[1,1].set_xlabel('Time (hr)')
axs[1,1].set_ylabel('mmol/mL')
axs[1,1].set_title('AAT')
axs[1,1].plot(tspan,P1_pred,label='Model sim.',color='blue')
axs[1,1].scatter(tspan,P1_exp,label='Data',color='red')
axs[1,1].legend()

axs[2,0].set_xlabel('Time (hr)')
axs[2,0].set_ylabel('%')
axs[2,0].set_title('Cell viability')
axs[2,0].plot(tspan,Xv_pred/(Xv_pred+Xd_pred)*100,label='Model sim.',color='blue')
axs[2,0].scatter(tspan,Xv_exp/(Xv_exp+Xd_exp)*100,label='Data',color='red')
axs[2,0].legend()

fig2, axs2 = plt.subplots(constrained_layout=True)
axs2.plot(tspan,Xv_pred+Xd_pred,label='Model sim.',color='blue')
axs2.scatter(tspan,Xv_exp+Xd_exp,label='Data',color='red')
axs2.set_xlabel('Time (hr)')
axs2.set_ylabel('10^6 cells/mL')
axs2.set_title('Total cells (live + dead)')
axs2.legend()

pars_ori = []
pars = []

mumax = MX.sym('mumax')
pars += [mumax]
pars_ori += [pars_opt_ori[0]]
Ks = MX.sym('Ks')
pars += [Ks]
pars_ori += [pars_opt_ori[1]]
Xvmax = MX.sym('Xvmax')
pars += [Xvmax]
pars_ori += [1/pars_opt_ori[2]]
Kd = MX.sym('Kd')
pars += [Kd]
pars_ori += [pars_opt_ori[3]]
mu = mumax*S/(S+Ks)*(1-Xv/Xvmax)
dXv = (mu-Kd)*Xv

k = MX.sym('k')
pars += [k]
pars_ori += [pars_opt_ori[4]]
Xtmax = MX.sym('Xtmax')
pars += [Xtmax]
pars_ori += [1/pars_opt_ori[5]]
Klys = MX.sym('Klys')
pars += [Klys]
pars_ori += [pars_opt_ori[6]]
Xt = Xv+Xd
dXd = k*Xv*(1-Xt/Xtmax)-dXv-Klys*Xd

Yxs = MX.sym('Yxs')
pars += [Yxs]
pars_ori += [1/pars_opt_ori[7]]
dS = -1/Yxs*mu*Xv

Yp1x = MX.sym('Yp1x')
pars += [Yp1x]
pars_ori += [pars_opt_ori[8]]
dP1 = Yp1x*mu*Xv

dy = vertcat(dXv,dXd,dS,dP1)
growth_model = {'x':y, 'p':vertcat(*pars), 'ode':dy}

for k in range(1,N):
    opts = {'tf':tspan[k]-tspan[0]}
    I = integrator('I', 'cvodes', growth_model, opts)
    
    I_jac = I.factory("I_jac", ["x0", "p"], ["jac:xf:p"])
    G[k] = I_jac(X0, pars_ori).full()

mumax = pars_opt_ori[0]
Ks = pars_opt_ori[1]
Xvmax = 1/pars_opt_ori[2]
Kd = pars_opt_ori[3]
mu = mumax*S/(S+Ks)*(1-Xv/Xvmax)

k = pars_opt_ori[4]
Xtmax = 1/pars_opt_ori[5]

Kdpseudo = k*(1-Xt/Xtmax)-(mu-Kd)
f_Kdpseudo = Function('f_Kdpseudo',[Xv,Xd,S],[Kdpseudo],['Xv','Xd','S'],['Kdpseudo'])
f_muhyb = Function('f_muhyb',[Xv,S],[mu])
f_klogit = Function('f_klogit',[Xv,Xd],[k*(1-Xt/Xtmax)])

tspan_new = np.linspace(tspan[0],tspan[-1])
kdpsudo = np.zeros(len(tspan_new))
Xd_pred = np.zeros(len(tspan_new))
muhyb = np.zeros(len(tspan_new))
klogit = np.zeros(len(tspan_new))
for k in range(len(tspan_new)):
    opts = {'tf':tspan_new[k]-tspan_new[0]}
    I = integrator('I', 'cvodes', growth_model, opts)
    
    Ik = I(x0=X0, p=pars_ori)
    Xk = Ik['xf'].full().flatten()
    
    Xd_pred[k] = Xk[1]
    
    kdpsudo[k] = f_Kdpseudo(Xk[0],Xk[1],Xk[2])
    muhyb[k] = f_muhyb(Xk[0],Xk[2])
    klogit[k] = f_klogit(Xk[0],Xk[1])
    
from scipy import optimize

kT = pars_ori[4]
Xtmax = pars_ori[5]
mumax = pars_ori[0]
Ks = pars_ori[1]
Xvmax = pars_ori[2]
Kd = pars_ori[3]
Klys = pars_ori[6]
Yxvs = pars_ori[7]
Ypxv = pars_ori[8]
    
def fun(y, D=0.):   
    Xv = y[0]
    Xd = y[1]
    S = y[2]
    
    Xt = Xv+Xd
    
    mu = (mumax*S)/(Ks+S)*(1-Xv/Xvmax)
    
    Sf=X0[2]
    
    return [mu-Kd-D,
            kT*Xv*(1-Xt/Xtmax)-mu*Xv+Kd*Xv-Klys*Xd-D*Xd,
            -mu*(1/Yxvs)*Xv+D*(Sf-S)]

Dval = np.linspace(0.,min(k,mumax),101)
Xv_ss = np.zeros(len(Dval))
Xd_ss = np.zeros(len(Dval))
S_ss = np.zeros(len(Dval))
kdps_ss = np.zeros(len(Dval))
PD_ss = np.zeros(len(Dval))
x0 = X0[:3]
for i, dval in enumerate(Dval[::-1]):
    sol = optimize.root(fun, x0, args=(dval))
    if sol.success:
        Xv_ss[i] = sol.x[0]
        Xd_ss[i] = sol.x[1]
        S_ss[i] = sol.x[2]
        x0 = sol.x
        
        mu = (mumax*S_ss[i])/(Ks+S_ss[i])*(1-Xv_ss[i]/Xvmax)
        kdps_ss[i] = f_Kdpseudo(Xv_ss[i],Xd_ss[i],S_ss[i])
        
        PD_ss[i] = Ypxv*mu*Xv_ss[i]
        
Xv_ss = Xv_ss[::-1]
Xd_ss = Xd_ss[::-1]
S_ss = S_ss[::-1]
kdps_ss = kdps_ss[::-1]
PD_ss = PD_ss[::-1]
    
xind= np.argwhere(Xv_ss>0.)