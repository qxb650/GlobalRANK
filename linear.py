import numpy as np
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import random
import time
import matplotlib.pyplot as plt
import os
from flax.core import FrozenDict
from jax.scipy.stats import norm
from jax.scipy.interpolate import RegularGridInterpolator
from matplotlib.ticker import MaxNLocator
from aux_ import draw_shocks

from model_funcs import euler_error, NKPC_error, taylor_rule

##################################
# GENERAL LINERIZATION FUNCTIONS #
##################################

def solve_for_P(A, B, C, K):

    # 1. define matrices in Sylvester Equation
    A_syl = jnp.linalg.solve(A, B)
    B_syl = K
    C_syl = -jnp.linalg.solve(A, C)

    # 2. call Sylvester solver
    P = jax.scipy.linalg.solve_sylvester(A_syl, B_syl, C_syl, method='eigen')

    return P

def compute_linear_policy(par, ZLB_regime=False):

    # 1. unpack DSS
    Y_DSS, pi_DSS, i_DSS = par["Y_DSS"], par["pi_DSS"], par["i_DSS"]
    u_DSS, z_DSS, ln_Gamma_DSS = par["u_DSS"], par["z_DSS"], par["ln_Gamma_DSS"]

    # 2. compute gradients of Euler Error
    def ee_wrapper(Y, Y_p, pi, pi_p, u, z, ln_Gamma, ZLB_regime):

        u = jnp.array([u])
        z = jnp.array([z])
        eps_z = jnp.zeros(1)
        eps_Gamma = jnp.zeros(1)
        w = jnp.ones(1)

        if ZLB_regime:
            i = 0.0
        else:
            i = taylor_rule(par, Y, pi, u, z, ln_Gamma, eps_z, eps_Gamma, -100, w)
            euler_error(par, Y, Y_p, pi, pi_p, i, u, z, eps_z, w)
            
        return euler_error(par, Y, Y_p, pi, pi_p, i, u, z, eps_z, w)[0]
    
    ee_grad_func = jax.grad(ee_wrapper, argnums=(0,1,2,3,4,5,6))
    ee_grads = ee_grad_func(Y_DSS, Y_DSS, pi_DSS, pi_DSS, u_DSS, z_DSS, ln_Gamma_DSS, ZLB_regime)

    ee_Y, ee_Y_p = ee_grads[0], ee_grads[1]
    ee_pi, ee_pi_p = ee_grads[2], ee_grads[3]
    ee_u, ee_z, ee_ln_Gamma = ee_grads[4], ee_grads[5], ee_grads[6]

    # 3. compute Euler Error residiual in point of approximation (only relevant in ZLB-regime)
    ee = ee_wrapper(Y_DSS, Y_DSS, pi_DSS, pi_DSS, u_DSS, z_DSS, ln_Gamma_DSS, ZLB_regime)

    # 4. compute gradients of NKPC Error
    def nkpce_wrapper(Y, Y_p, pi, pi_p, u, z, ln_Gamma):

        u = jnp.array([u])
        z = jnp.array([z])
        ln_Gamma = jnp.array([ln_Gamma])
        eps_z = jnp.zeros(1)
        eps_Gamma = jnp.zeros(1)
        w = jnp.ones(1)

        if ZLB_regime:
            i = 0.0
        else:
            i = taylor_rule(par, Y, pi, u, z, ln_Gamma, eps_z, eps_Gamma, -100, w)
        return NKPC_error(par, Y, Y_p, pi, pi_p, i, u, ln_Gamma, w)[0]

    nkpce_grad_func = jax.grad(nkpce_wrapper, argnums=(0,1,2,3,4,5,6))
    nkpce_grads = nkpce_grad_func(Y_DSS, Y_DSS, pi_DSS, pi_DSS, u_DSS, z_DSS, ln_Gamma_DSS)

    nkpce_Y, nkpce_Y_p = nkpce_grads[0], nkpce_grads[1]
    nkpce_pi, nkpce_pi_p = nkpce_grads[2], nkpce_grads[3]
    nkpce_u, nkpce_z, nkpce_ln_Gamma = nkpce_grads[4], nkpce_grads[5], nkpce_grads[6]

    # 5. compute NKPC Error residual in point of approximation (only relevant in ZLB-regime)
    nkpce = nkpce_wrapper(Y_DSS, Y_DSS, pi_DSS, pi_DSS, u_DSS, z_DSS, ln_Gamma_DSS)
 
    # 6. contruct structural matrices: Rows are [Y, pi], columns are [u, z, ln(Gamma)]
    A = jnp.array([ # (2,2)
        [ee_Y_p, ee_pi_p],
        [nkpce_Y_p, nkpce_pi_p]
    ])

    B = jnp.array([ # (2, 2)
        [ee_Y, ee_pi],
        [nkpce_Y, nkpce_pi]
    ])

    C = jnp.array([ # (2, 3)
        [ee_u, ee_z, ee_ln_Gamma],
        [nkpce_u, nkpce_z, nkpce_ln_Gamma]
    ])

    D = jnp.array([ # (2, 1)
        [ee],
        [nkpce]
    ])

    K = jnp.diag(jnp.array([par["rho_u"], par["rho_z"], par["rho_Gamma"]])) # (3, 3)

    # 7. compute policy
    P = solve_for_P(A, B, C, K)

    return P, A, B, C, D, K

################
# SETUP LINEAR #
################

def setup_linear(model, T_OccBin, n_grid=50, shock_interp=0.03):

    par = FrozenDict(model.par)

    linear = dict()

    # policy matrices of non-ZLB model, structural matrices of no-ZLB regime
    P, A, B, C, D, K = compute_linear_policy(par, ZLB_regime=False)

    linear["P"] = P
    linear["A"] = A
    linear["B"] = B
    linear["C"] = C
    linear["D"] = D
    linear["K"] = K

    # structural matrices of ZLB regime
    P_ZLB, A_ZLB, B_ZLB, C_ZLB, D_ZLB, K_ZLB = compute_linear_policy(par, ZLB_regime=True)

    linear["P_ZLB"] = P_ZLB
    linear["A_ZLB"] = A_ZLB
    linear["B_ZLB"] = B_ZLB
    linear["C_ZLB"] = C_ZLB
    linear["D_ZLB"] = D_ZLB
    linear["K_ZLB"] = K_ZLB

    # OccBin policy matrices 
    P_ZLB_hist, d_ZLB_hist = compute_P_star(P, A_ZLB, B_ZLB, C_ZLB, D_ZLB, K_ZLB, T_OccBin)

    linear["P_ZLB_hist"] = P_ZLB_hist
    linear["d_ZLB_hist"] = d_ZLB_hist

    # aux
    linear["T_OccBin"] = T_OccBin

    # OccBin interpolators
    out_OccBin = compute_OccBin_interp(par, linear, n_grid, shock_interp)
    linear["Y_interp_OccBin"], linear["pi_interp_OccBin"], linear["time_to_ZLB_slack_interp"], linear["max_expected_ZLB"] = out_OccBin

    model.linear = linear

def compute_P_star(P, A_ZLB, B_ZLB, C_ZLB, D_ZLB, K_ZLB, T):

    # 1. define function for scan: find policy if ZLB is expected slack T periods out, T-1 periods out, ...
    def scan_fun(carry, _):

        P_curr, d_curr = carry

        # implied policy from equilibrium equations
        P_next = -jnp.linalg.solve(B_ZLB, A_ZLB @ P_curr @ K_ZLB + C_ZLB)
        d_next = -jnp.linalg.solve(B_ZLB, A_ZLB @ d_curr + D_ZLB)

        return (P_next, d_next), (P_next, d_next)

    # 2. define initial carry
    init_carry = (P, jnp.zeros((2,1)))

    # 3. backwards induction: tau = T, T-1, T-2, ..., 1
    _, (P_hist, d_hist) = jax.lax.scan(scan_fun, init_carry, None, length=T)

    return P_hist, d_hist

def compute_OccBin_interp(par, linear, n_grid, states_sigma):

    n_grid = n_grid + 1 # +1 for 0.5 being contained in the grid, ppf(0.5) = 0.0
    unit_grid = jnp.linspace(0.000001,0.999999,n_grid)
    u_grid = norm.ppf(unit_grid, loc=0, scale=states_sigma)
    z_grid = norm.ppf(unit_grid, loc=0, scale=states_sigma)
    ln_Gamma_grid = norm.ppf(unit_grid, loc=0, scale=states_sigma)

    states_grid = jnp.stack(jnp.meshgrid(u_grid, z_grid, ln_Gamma_grid, indexing="ij"), axis=-1).reshape(-1,3)

    X_sol_grid, time_to_ZLB_slack = OccBin(par, linear, states_grid)
    Y_sol_grid = X_sol_grid[:, 0].reshape(n_grid, n_grid, n_grid)
    pi_sol_grid = X_sol_grid[:, 1].reshape(n_grid, n_grid, n_grid)
    time_to_ZLB_slack = time_to_ZLB_slack[:, 0].reshape(n_grid, n_grid, n_grid)
    max_expected_ZLB = time_to_ZLB_slack.max()

    Y_interp = RegularGridInterpolator((u_grid, z_grid, ln_Gamma_grid), Y_sol_grid, fill_value=None)
    pi_interp = RegularGridInterpolator((u_grid, z_grid, ln_Gamma_grid), pi_sol_grid, fill_value=None)
    time_to_ZLB_slack_interp = RegularGridInterpolator((u_grid, z_grid, ln_Gamma_grid), time_to_ZLB_slack, fill_value=None, method = "nearest")

    return Y_interp, pi_interp, time_to_ZLB_slack_interp, max_expected_ZLB

####################
# OccBin ALGORITHM #
####################

def compute_policy_and_ZLB(par, states, P, d):

    X = states @ P.T + d.T # (N, 3) x (3, 2) -> (N, 2)
    pi = X[:, 1]  # (N,)
    Y = X[:, 0] + par["Y_DSS"] # REMEBER TAYLOR RULE IS MADE FOR LEVEL VARIABLES
    u = states[:, 0] # (N,)
    z = states[:, 1]
    ln_Gamma = states[:, 2]
    i_shadow = taylor_rule(par, Y, pi, u, z, ln_Gamma, jnp.zeros(1), jnp.zeros(1), -100, jnp.ones(1)) # (N,)
    ZLB_binds = i_shadow < par["ZLB"]

    return X, ZLB_binds

def OccBin(par, linear, states):

    # infer N
    N = states.shape[0]
    d = jnp.zeros((2,1))

    P = linear["P"]
    A_ZLB = linear["A_ZLB"]
    B_ZLB = linear["B_ZLB"]
    C_ZLB = linear["C_ZLB"]
    D_ZLB = linear["D_ZLB"]
    K_ZLB = linear["K_ZLB"]

    P_hist = linear["P_ZLB_hist"]
    d_hist = linear["d_ZLB_hist"]

    T_max = linear["T_OccBin"]

    # allocate solutions
    X_sol = jnp.zeros((N, 2)) + jnp.nan
    time_to_ZLB_slack = jnp.zeros((N,1))

    # 1. Guess: T = 0, does ZLB not hold?
    X_sol_init, ZLB_binds_init = compute_policy_and_ZLB(par, states, P, d)
    
    # 1.a. if ZLB does not hold in period, fill in standard linear solution
    solved_init = ~ZLB_binds_init
    X_sol = jnp.where(solved_init[:, None], X_sol_init, X_sol)
    
    # 1.b. fill in that there are 0 periods to ZLB does not hold anymore
    time_to_ZLB_slack_init = jnp.where(solved_init[:, None], 0.0, time_to_ZLB_slack)

    # 1.c. expected state transition
    states_T_init = states

    # 2. Guess: T = 1, 2, 3, ..., T_max
    
    # 2.a. define carry in lax.scan
    carry_init = (X_sol, solved_init, states_T_init, time_to_ZLB_slack_init)

    # 2.b. define scan function
    def scan_fun(carry, T):

        # a. unpack carry, denoted i
        X_sol_i, solved_i, states_T_iminus, time_to_ZLB_slack_i = carry

        # b. state transition in expectations
        states_T_i = states_T_iminus @ K_ZLB.T

        # c. unpack policy for t for when ZLB is slack in period t+T: pi_t, Y_t
        X, ZLB_binds = compute_policy_and_ZLB(par, states, P_hist[T-1], d_hist[T-1])

        # d. compute t+T policy: check if ZLB is expected to be slack
        _, ZLB_binds_T = compute_policy_and_ZLB(par, states_T_i, P, d)
        ZLB_slack_T = ~ZLB_binds_T

        # e. ZLB expected slack in t+T, ZLB binds in t, not solved before ? -> fill in 
        fill_in_mask = ZLB_slack_T & (~solved_i) #& ZLB_binds 
        X_sol_next = jnp.where(fill_in_mask[:, None], X, X_sol_i)
        time_to_ZLB_slack_next = jnp.where(fill_in_mask[:, None], T, time_to_ZLB_slack_i)
        solved_next = solved_i | fill_in_mask

        return (X_sol_next, solved_next, states_T_i, time_to_ZLB_slack_next), None

    (X_sol_final, solved_final, _, time_to_ZLB_slack_final), _ = jax.lax.scan(scan_fun, carry_init, jnp.arange(1,T_max))

    assert solved_final.sum() == N

    return X_sol_final, time_to_ZLB_slack_final

def simulate_OccBin(model, N, states_sigma, key_=42, return_linear_wo_OccBin=False):

    par = FrozenDict(model.par)
    linear = FrozenDict(model.linear)
    
    # 2. set key and draw states
    key = jax.random.PRNGKey(key_)
    states = states_sigma*jax.random.normal(key, shape=(N, 3))

    # 3. run OccBin
    X_sol = OccBin(par, linear, states)

    if return_linear_wo_OccBin:
        P = linear["P"]
        X_sol_no_OccBin = states @ P.T

        return states, X_sol, X_sol_no_OccBin

    else:
        return states, X_sol

def eval_OccBin(linear, states):

    Y_interp, pi_interp = linear["Y_interp_OccBin"], linear["pi_interp_OccBin"]

    if len(states.shape) == 2:
        Y = Y_interp(states)
        pi = pi_interp(states)

    else:
        N = states.shape[0]
        states = states.reshape(-1, 3)
        Y = Y_interp(states).reshape(N, -1, 2)
        pi = pi_interp(states).reshape(N, -1, 2)

    return Y, pi

########
# IRFs #
########

def compute_linear_IRFs(model, shock, T, rtol=50):

    par = model.par
    linear = model.linear

    P = linear["P"]

    rho_u = par["rho_u"]
    rho_z = par["rho_z"]
    rho_Gamma = par["rho_Gamma"]

    T_u = int(10*jnp.ceil(-jnp.log(rtol)/(10*jnp.log(rho_u))).item())
    T_z = int(10*jnp.ceil(-jnp.log(rtol)/(10*jnp.log(rho_z))).item())
    T_Gamma = int(10*jnp.ceil(-jnp.log(rtol)/(10*jnp.log(rho_Gamma))).item())

    # MP shock
    zeros = jnp.zeros((T_u, 1))
    u_shock = shock * rho_u**jnp.arange(T_u)
    u_shock_states = jnp.concat([u_shock[:, None], zeros, zeros], axis=-1)
    out_u_OccBin, exp_T_u_OccBin = OccBin(par, linear, u_shock_states)
    out_u_lin = u_shock_states @ P.T

    IRF_Y_u_OccBin = out_u_OccBin[:, 0]
    IRF_pi_u_OccBin = out_u_OccBin[:, 1]

    IRF_Y_u_lin = out_u_lin[:, 0]
    IRF_pi_u_lin = out_u_lin[:, 1]

    # preference shock
    zeros = jnp.zeros((T_z, 1))
    z_shock = shock * rho_z**jnp.arange(T_z)
    z_shock_states = jnp.concat([zeros, z_shock[:, None], zeros], axis=-1)
    out_z_OccBin, exp_T_z_OccBin = OccBin(par, linear, z_shock_states)
    out_z_lin = z_shock_states @ P.T

    IRF_Y_z_OccBin = out_z_OccBin[:, 0]
    IRF_pi_z_OccBin = out_z_OccBin[:, 1]

    IRF_Y_z_lin = out_z_lin[:, 0]
    IRF_pi_z_lin = out_z_lin[:, 1]

    # productivity shock
    zeros = jnp.zeros((T_Gamma, 1))
    ln_Gamma_shock = shock * rho_Gamma**jnp.arange(T_Gamma)
    ln_Gamma_shock_states = jnp.concat([zeros, zeros, ln_Gamma_shock[:, None]], axis=-1)
    out_ln_Gamma_OccBin, exp_T_ln_Gamma_OccBin = OccBin(par, linear, ln_Gamma_shock_states)
    out_ln_Gamma_lin = ln_Gamma_shock_states @ P.T

    IRF_Y_ln_Gamma_OccBin = out_ln_Gamma_OccBin[:, 0]
    IRF_pi_ln_Gamma_OccBin = out_ln_Gamma_OccBin[:, 1]

    IRF_Y_ln_Gamma_lin = out_ln_Gamma_lin[:, 0]
    IRF_pi_ln_Gamma_lin = out_ln_Gamma_lin[:, 1]

    if hasattr(model, "IRF"):
        IRF = model.IRF

    else:
        IRF = SimpleNamespace()

        IRF.u = u_shock
        IRF.z = z_shock
        IRF.ln_Gamma = ln_Gamma_shock

        IRF.T_u = T_u
        IRF.T_z = T_z
        IRF.T_Gamma = T_Gamma

    IRF.Y_u_OccBin = IRF_Y_u_OccBin
    IRF.pi_u_OccBin = IRF_pi_u_OccBin
    IRF.exp_T_u_OccBin = exp_T_u_OccBin
    IRF.Y_u_lin = IRF_Y_u_lin
    IRF.pi_u_lin = IRF_pi_u_lin

    IRF.Y_z_OccBin = IRF_Y_z_OccBin
    IRF.pi_z_OccBin = IRF_pi_z_OccBin
    IRF.exp_T_z_OccBin = exp_T_z_OccBin
    IRF.Y_z_lin = IRF_Y_z_lin
    IRF.pi_z_lin = IRF_pi_z_lin

    IRF.Y_ln_Gamma_OccBin = IRF_Y_ln_Gamma_OccBin
    IRF.pi_ln_Gamma_OccBin = IRF_pi_ln_Gamma_OccBin
    IRF.exp_T_ln_Gamma_OccBin = exp_T_ln_Gamma_OccBin
    IRF.Y_ln_Gamma_lin = IRF_Y_ln_Gamma_lin
    IRF.pi_ln_Gamma_lin = IRF_pi_ln_Gamma_lin

    if not hasattr(model, "IRF"): model.IRF = IRF

def plot_linear_IRFs(model, plot_exp_T = False):

    IRF = model.IRF

    T_u = IRF.T_u
    T_z = IRF.T_z
    T_Gamma = IRF.T_Gamma

    f, ax = plt.subplots(3+int(plot_exp_T), 3, figsize=(12, 12))

    # shocks (du antager de ligger i IRF)
    ax[0,0].plot(jnp.arange(T_u), IRF.u)
    ax[0,1].plot(jnp.arange(T_z), IRF.z)
    ax[0,2].plot(jnp.arange(T_Gamma), IRF.ln_Gamma)

    ax[0,0].set_title(r'$u_t$')
    ax[0,1].set_title(r'$z_t$')
    ax[0,2].set_title(r'$\ln(\Gamma_t)$')

    # OUTPUT (OccBin vs lin)
    ax[1,0].plot(jnp.arange(T_u), IRF.Y_u_OccBin, label='OccBin')
    ax[1,0].plot(jnp.arange(T_u), IRF.Y_u_lin, label='linear')

    ax[1,1].plot(jnp.arange(T_z), IRF.Y_z_OccBin, label='OccBin')
    ax[1,1].plot(jnp.arange(T_z), IRF.Y_z_lin, label='linear')

    ax[1,2].plot(jnp.arange(T_Gamma), IRF.Y_ln_Gamma_OccBin, label='OccBin')
    ax[1,2].plot(jnp.arange(T_Gamma), IRF.Y_ln_Gamma_lin, label='linear')

    for i in range(3):
        ax[1,i].set_title('Output')

    # INFLATION (OccBin vs lin)
    ax[2,0].plot(jnp.arange(T_u), IRF.pi_u_OccBin, label='OccBin')
    ax[2,0].plot(jnp.arange(T_u), IRF.pi_u_lin, label='linear')

    ax[2,1].plot(jnp.arange(T_z), IRF.pi_z_OccBin, label='OccBin')
    ax[2,1].plot(jnp.arange(T_z), IRF.pi_z_lin, label='linear')

    ax[2,2].plot(jnp.arange(T_Gamma), IRF.pi_ln_Gamma_OccBin, label='OccBin')
    ax[2,2].plot(jnp.arange(T_Gamma), IRF.pi_ln_Gamma_lin, label='linear')

    for i in range(3):
        ax[2,i].set_title('Inflation')

    for i in range(3):
        for j in range(3):
            ax[i,j].legend()

    if plot_exp_T:
        ax[3,0].step(jnp.arange(T_u), IRF.exp_T_u_OccBin, where='post')
        ax[3,1].step(jnp.arange(T_z), IRF.exp_T_z_OccBin, where='post')
        ax[3,2].step(jnp.arange(T_Gamma), IRF.exp_T_ln_Gamma_OccBin, where='post')
        ax[3,0].yaxis.set_major_locator(MaxNLocator(integer=True))
        ax[3,1].yaxis.set_major_locator(MaxNLocator(integer=True))
        ax[3,2].yaxis.set_major_locator(MaxNLocator(integer=True))

        for i in range(3):
            ax[3,i].set_title('Expected Duration of ZLB')

    f.tight_layout()


##############
# SIMULATION #
##############

def simulate_linear(model, sigmas, T, N=1, known_states=None, key_=42, plot=False):

    par = model.par
    linear = model.linear
    dtype = model.dtype

    P = linear["P"]
    K = linear["K"]
    key = jax.random.PRNGKey(key_)

    sigma_sim_eps_u = sigmas["sigma_eps_u"]
    sigma_sim_eps_z = sigmas["sigma_eps_z"]
    sigma_sim_eps_Gamma = sigmas["sigma_eps_Gamma"]

    states = jnp.zeros((T,N,3)) + jnp.nan
    
    # (N,3)

    # allocate for solutions
    Y_lin = jnp.zeros((T,N)) + jnp.nan
    pi_lin = jnp.zeros((T,N)) + jnp.nan
    i_lin = jnp.zeros((T,N)) + jnp.nan

    Y_OccBin = jnp.zeros((T,N)) + jnp.nan
    pi_OccBin = jnp.zeros((T,N)) + jnp.nan
    i_OccBin = jnp.zeros((T,N)) + jnp.nan

    for t in range(T):

        if known_states is not None:
            states_t = known_states[t]

        else:
            if t == 0:
                states_t = jnp.zeros((N,3))
            else:
                key, subkey = jax.random.split(key)
                eps = draw_shocks(subkey, dtype, N, sigma_sim_eps_u, sigma_sim_eps_z, sigma_sim_eps_Gamma)
                states_t = states_t @ K.T + eps
        
        states = states.at[t, :, :].set(states_t)

        u, z, ln_Gamma = states_t[:, 0], states_t[:, 1], states_t[:, 2]

        out_lin_t = states_t @ P.T
        
        Y_lin = Y_lin.at[t, :].set(out_lin_t[:, 0] + par["Y_DSS"])
        pi_lin = pi_lin.at[t, :].set(out_lin_t[:, 1]) 
        i_lin = i_lin.at[t, :].set(taylor_rule(par, out_lin_t[:, 0]+ par["Y_DSS"], out_lin_t[:, 1], u, z, ln_Gamma, jnp.zeros((1,1)), jnp.zeros((1,1)), -100, jnp.ones((1,1))))

        out_OccBin_t, _ = OccBin(par, linear, states_t)

        Y_OccBin = Y_OccBin.at[t, :].set(out_OccBin_t[:, 0]+ par["Y_DSS"])
        pi_OccBin = pi_OccBin.at[t, :].set(out_OccBin_t[:, 1]) 
        i_OccBin = i_OccBin.at[t, :].set(taylor_rule(par, out_OccBin_t[:, 0]+ par["Y_DSS"], out_OccBin_t[:, 1], u, z, ln_Gamma, jnp.zeros((1,1)), jnp.zeros((1,1)), 0.00, jnp.ones((1,1))))

    if hasattr(model, "sim"):
        sim = model.sim

    else:
        sim = SimpleNamespace()

    sim.states = states

    sim.Y_lin = Y_lin
    sim.pi_lin = pi_lin
    sim.i_lin = i_lin

    sim.Y_OccBin = Y_OccBin
    sim.pi_OccBin = pi_OccBin
    sim.i_OccBin = i_OccBin

    if not hasattr(model, "sim"): model.sim = sim

    if plot:
        f, ax = plt.subplots(1,3, figsize=(15,5))

        ax[0].plot(Y_lin, label='Linear')
        ax[0].plot(Y_OccBin, label='OccBin')

        ax[1].plot(pi_lin, label='Linear')
        ax[1].plot(pi_OccBin, label='OccBin')

        ax[2].plot(i_lin, label='Linear')
        ax[2].plot(i_OccBin, label='OccBin')

        for i in range(3): ax[i].legend()

###################
# ERROR FUNCTIONS #
###################

