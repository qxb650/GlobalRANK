import numpy as np
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax.core import FrozenDict
from jax.scipy.stats import norm
from jax.scipy.interpolate import RegularGridInterpolator
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
from aux_ import draw_shocks, draw_states_directly
from copy import copy
from neural_nets import eval_nn
from aux_ import next_states_quad
import scipy.io as sio

from model_funcs import euler_error, NKPC_error, taylor_rule_lin, taylor_rule

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

    # unpack DSS
    Y_DSS, pi_DSS, i_DSS = par["Y_DSS"], par["pi_DSS"], par["i_DSS"]
    u_DSS, z_DSS, ln_Gamma_DSS = par["u_DSS"], par["z_DSS"], par["ln_Gamma_DSS"]

    # unpack pars and compute aux pars
    alpha = par["alpha"]
    beta = par["beta"]
    sigma = par["sigma"]
    kappa = par["kappa"]
    epsilon = par["epsilon"]
    varphi = par["varphi"]
    ZLB = par["ZLB"]

    rho_u = par["rho_u"]
    rho_z = par["rho_z"]
    rho_Gamma = par["rho_Gamma"]

    mu = epsilon/(epsilon-1)
    kappa_lin = (kappa/mu)*((varphi+alpha+sigma-alpha*sigma)/(1-alpha))

    # handle ZLB regime
    if ZLB_regime:
        phi_y = 0.0
        phi_pi = 0.0
        ee_error = -jnp.log(1+ZLB) - jnp.log(beta)

    else:
        phi_y = par["phi_y"]
        phi_pi = par["phi_pi"]
        ee_error = 0.0

    # compute structural matrices
    A = jnp.array([ # (2,2)
        [sigma, 1.0],
        [0, -beta]
    ])

    B = jnp.array([ # (2, 2)
        [-(phi_y+sigma), -phi_pi],
        [-kappa_lin, 1]
    ])

    C = jnp.array([ # (2, 3)
        [-1, 1-rho_z, phi_y*((1+varphi)/(varphi+alpha+sigma-alpha*sigma))],
        [0, 0, (kappa/mu)*((1+varphi)/(1-alpha))]
    ])

    D = jnp.array([ # (2, 1)
        [ee_error],
        [0.00]
    ])

    K = jnp.diag(jnp.array([par["rho_u"], par["rho_z"], par["rho_Gamma"]])) # (3, 3)

    # solve for P in Sylvester equation
    P = solve_for_P(A, B, C, K)

    return P, A, B, C, D, K

################
# SETUP LINEAR #
################

def setup_linear(model, T_OccBin):

    par = model.par
    train = model.train

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

    linear["T_OccBin"] = train["T_OccBin"]

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

####################
# OccBin ALGORITHM #
####################

def compute_policy_and_ZLB(par, states, P, d):

    X = states @ P.T + d.T # (N, 3) x (3, 2) -> (N, 2)
    pi = X[:, 1]
    Y = X[:, 0]
    u = states[:, 0] # (N,)
    z = states[:, 1]
    ln_Gamma = states[:, 2]
    i_dev_shadow = taylor_rule_lin(par, Y, pi, u, ln_Gamma)
    ZLB_binds = par["i_DSS"] + i_dev_shadow <= par["ZLB"]  + 1e-5

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

################
# GET POLICIES #
################

def compute_Y(par, Y_dev):

    Y_DSS = par["Y_DSS"]
    
    Y = Y_DSS*jnp.exp(Y_dev)

    return Y

def compute_Y_per(par, Y_dev):
    
    Y_DSS = par["Y_DSS"]
    
    Y_per = jnp.exp(Y_dev)-1

    return Y_per

def eval_lin(model, states, return_dev, return_i=False):

    par = model.par
    linear = model.linear

    P = linear["P"].T # (3, 2)

    out_lin = states @ P

    Y_raw = out_lin[..., 0]
    pi_raw = out_lin[..., 1]

    if return_dev:
        Y_dev = compute_Y_per(par, Y_raw)
        pi_dev = pi_raw

        if return_i:
            i_dev = taylor_rule_lin(par, Y_raw, pi_raw, states[..., 0], states[..., 2])

            return Y_dev, pi_dev, i_dev
            
        else:
            return Y_dev, pi_dev

    else:
        Y = compute_Y(par, Y_raw)
        pi = pi_raw
        
        if return_i:
            i_dev = taylor_rule_lin(par, Y_raw, pi_raw, states[..., 0], states[..., 2])
            i = par["i_DSS"] + i_dev
            return Y, pi, i
        else:
            return Y, pi

def eval_lin_womodel(par, linear, states, return_dev, return_i=False):

    P = linear["P"].T # (3, 2)

    out_lin = states @ P

    Y_raw = out_lin[..., 0]
    pi_raw = out_lin[..., 1]

    if return_dev:
        Y_dev = compute_Y_per(par, Y_raw)
        pi_dev = pi_raw

        if return_i:
            i_dev = taylor_rule_lin(par, Y_raw, pi_raw, states[..., 0], states[..., 2])

            return Y_dev, pi_dev, i_dev
            
        else:
            return Y_dev, pi_dev

    else:
        Y = compute_Y(par, Y_raw)
        pi = pi_raw
        
        if return_i:
            i_dev = taylor_rule_lin(par, Y_raw, pi_raw, states[..., 0], states[..., 2])
            i = par["i_DSS"] + i_dev
            return Y, pi, i
        else:
            return Y, pi

# def eval_lin_nn(par, linear, states, return_dev):

#     P = linear["P"].T # (3, 2)

#     out_lin = states @ P

#     Y_raw = out_lin[..., 0]
#     pi_raw = out_lin[..., 1]

#     if return_dev:
#         Y_dev = compute_Y_per(par, Y_raw)
#         pi_dev = pi_raw

#         return Y_dev, pi_dev

#     else:
#         Y = compute_Y(par, Y_raw)
#         pi = pi_raw

#         return Y, pi


def eval_OccBin(model, states, return_dev=False, return_i=False):

    par = model.par
    linear = model.linear

    shape = states.shape
    final_shape = shape[:-1]

    states = states.reshape(-1, 3)

    out_OccBin, _ = OccBin(par, linear, states)

    Y_raw = out_OccBin[:, 0].reshape(final_shape)
    pi_raw = out_OccBin[:, 1].reshape(final_shape)
    states = states.reshape(shape)

    if return_dev:
        Y_dev = compute_Y_per(par, Y_raw)
        pi_dev = pi_raw

        if return_i:
            i_dev = taylor_rule_lin(par, Y_raw, pi_raw, states[..., 0], states[..., 2])
            i_dev = jnp.maximum(i_dev, par["ZLB"] - par["i_DSS"])

            return Y_dev, pi_dev, i_dev
        else:
            return Y_dev, pi_dev

    else:
        Y = compute_Y(par, Y_raw)
        pi = pi_raw

        if return_i:
        
            i_dev = taylor_rule_lin(par, Y_raw, pi_raw, states[..., 0], states[..., 2])
            i_dev = jnp.maximum(i_dev, par["ZLB"] - par["i_DSS"])
            i = par["i_DSS"] + i_dev

            return Y, pi, i

        else:
            return Y, pi

def eval_OccBin_womodel(par, linear, states, return_dev=False, return_i=False):

    shape = states.shape
    final_shape = shape[:-1]

    states = states.reshape(-1, 3)

    out_OccBin, _ = OccBin(par, linear, states)

    Y_raw = out_OccBin[:, 0].reshape(final_shape)
    pi_raw = out_OccBin[:, 1].reshape(final_shape)
    states = states.reshape(shape)

    if return_dev:
        Y_dev = compute_Y_per(par, Y_raw)
        pi_dev = pi_raw

        if return_i:
            i_dev = taylor_rule_lin(par, Y_raw, pi_raw, states[..., 0], states[..., 2])
            i_dev = jnp.maximum(i_dev, par["ZLB"] - par["i_DSS"])

            return Y_dev, pi_dev, i_dev
        else:
            return Y_dev, pi_dev

    else:
        Y = compute_Y(par, Y_raw)
        pi = pi_raw

        if return_i:
        
            i_dev = taylor_rule_lin(par, Y_raw, pi_raw, states[..., 0], states[..., 2])
            i_dev = jnp.maximum(i_dev, par["ZLB"] - par["i_DSS"])
            i = par["i_DSS"] + i_dev

            return Y, pi, i

        else:
            return Y, pi

########
# IRFs #
########

def compute_lin_OccBin_IRFs(model, sigma_dict, rtol=50, u_neg=False, z_neg=False, ln_Gamma_neg=False, T=None):

    par = model.par
    linear = model.linear

    P = linear["P"]
    rho_u = par["rho_u"]
    rho_z = par["rho_z"]
    rho_Gamma = par["rho_Gamma"]

    # unpack shock values
    u_shock = sigma_dict["sigma_eps_u"]
    z_shock = sigma_dict["sigma_eps_z"]
    ln_Gamma_shock = sigma_dict["sigma_eps_Gamma"]

    if u_neg: u_shock = -u_shock
    if z_neg: z_shock = -z_shock
    if ln_Gamma_neg: ln_Gamma_shock = -ln_Gamma_shock

    # compute when process is rtol times smaller
    T_u = int(10*jnp.ceil(-jnp.log(rtol)/(10*jnp.log(rho_u))).item())
    T_z = int(10*jnp.ceil(-jnp.log(rtol)/(10*jnp.log(rho_z))).item())
    T_Gamma = int(10*jnp.ceil(-jnp.log(rtol)/(10*jnp.log(rho_Gamma))).item())

    if T is None:
        T_u = T_z = T_Gamma = 7
    else:
        T_u = T_z = T_Gamma = T

    # MP shock
    u_shock_states = jnp.concat([(u_shock * rho_u**jnp.arange(T_u))[:, None], jnp.zeros((T_u, 1)), jnp.zeros((T_u, 1))], axis=-1)
    IRF_Y_u_OccBin, IRF_pi_u_OccBin, IRF_i_u_OccBin = eval_OccBin(model, u_shock_states, return_dev=True, return_i=True)
    IRF_Y_u_lin, IRF_pi_u_lin, IRF_i_u_lin = eval_lin(model, u_shock_states, return_dev=True, return_i=True)

    # preference shock
    z_shock_states = jnp.concat([jnp.zeros((T_z, 1)), (z_shock * rho_z**jnp.arange(T_z))[:, None], jnp.zeros((T_z, 1))], axis=-1)
    IRF_Y_z_OccBin, IRF_pi_z_OccBin, IRF_i_z_OccBin = eval_OccBin(model, z_shock_states, return_dev=True, return_i=True)
    IRF_Y_z_lin, IRF_pi_z_lin, IRF_i_z_lin = eval_lin(model, z_shock_states, return_dev=True, return_i=True)

    # productivity shock
    ln_Gamma_shock_states = jnp.concat([jnp.zeros((T_Gamma, 1)), jnp.zeros((T_Gamma, 1)), (ln_Gamma_shock * rho_Gamma**jnp.arange(T_Gamma))[:, None]], axis=-1)
    IRF_Y_ln_Gamma_OccBin, IRF_pi_ln_Gamma_OccBin, IRF_i_ln_Gamma_OccBin = eval_OccBin(model, ln_Gamma_shock_states, return_dev=True, return_i=True)
    IRF_Y_ln_Gamma_lin, IRF_pi_ln_Gamma_lin, IRF_i_ln_Gamma_lin = eval_lin(model, ln_Gamma_shock_states, return_dev=True, return_i=True)

    if hasattr(model, "IRF"):
        IRF = model.IRF

    else:
        IRF = SimpleNamespace()

        IRF.u = u_shock * rho_u**jnp.arange(T_u)
        IRF.z = z_shock * rho_z**jnp.arange(T_z)
        IRF.ln_Gamma = ln_Gamma_shock * rho_Gamma**jnp.arange(T_Gamma)

        IRF.T_u = T_u
        IRF.T_z = T_z
        IRF.T_Gamma = T_Gamma

    IRF.Y_u_OccBin = IRF_Y_u_OccBin
    IRF.pi_u_OccBin = IRF_pi_u_OccBin
    IRF.i_u_OccBin = IRF_i_u_OccBin
    IRF.Y_u_lin = IRF_Y_u_lin
    IRF.pi_u_lin = IRF_pi_u_lin
    IRF.i_u_lin = IRF_i_u_lin

    IRF.Y_z_OccBin = IRF_Y_z_OccBin
    IRF.pi_z_OccBin = IRF_pi_z_OccBin
    IRF.i_z_OccBin = IRF_i_z_OccBin
    IRF.Y_z_lin = IRF_Y_z_lin
    IRF.pi_z_lin = IRF_pi_z_lin
    IRF.i_z_lin = IRF_i_z_lin

    IRF.Y_ln_Gamma_OccBin = IRF_Y_ln_Gamma_OccBin
    IRF.pi_ln_Gamma_OccBin = IRF_pi_ln_Gamma_OccBin
    IRF.i_ln_Gamma_OccBin = IRF_i_ln_Gamma_OccBin
    IRF.Y_ln_Gamma_lin = IRF_Y_ln_Gamma_lin
    IRF.pi_ln_Gamma_lin = IRF_pi_ln_Gamma_lin
    IRF.i_ln_Gamma_lin = IRF_i_ln_Gamma_lin

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

def simulate_linear(model, sigmas, T, N=1, known_states=None, key_=42, plot=False, do_save=False):

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
        pi_lin = pi_lin.at[t, :].set(out_lin_t[:, 1]+ par["pi_DSS"]) 
        i_lin = i_lin.at[t, :].set(taylor_rule(par, out_lin_t[:, 0]+ par["Y_DSS"], out_lin_t[:, 1]+ par["pi_DSS"], u, z, ln_Gamma, jnp.zeros((1,1)), jnp.zeros((1,1)), -100, jnp.ones((1,1))))

        out_OccBin_t, _ = OccBin(par, linear, states_t)

        Y_OccBin = Y_OccBin.at[t, :].set(out_OccBin_t[:, 0]+ par["Y_DSS"])
        pi_OccBin = pi_OccBin.at[t, :].set(out_OccBin_t[:, 1]+ par["pi_DSS"])
        i_OccBin = i_OccBin.at[t, :].set(taylor_rule(par, out_OccBin_t[:, 0]+ par["Y_DSS"], out_OccBin_t[:, 1]+ par["pi_DSS"], u, z, ln_Gamma, jnp.zeros((1,1)), jnp.zeros((1,1)), par["ZLB"], jnp.ones((1,1))))

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

        f, ax = plt.subplots(2,3, figsize=(15,10))

        ax[0,0].plot(Y_lin, label='Linear')
        ax[0,0].plot(Y_OccBin, label='OccBin', ls='--')

        ax[0,1].plot(pi_lin, label='Linear')
        ax[0,1].plot(pi_OccBin, label='OccBin', ls='--')

        ax[0,2].plot(i_lin, label='Linear')
        ax[0,2].plot(i_OccBin, label='OccBin', ls='--')

        ax[0,0].set_title(r'Output: $Y_t$')
        ax[0,1].set_title(r'Inflation: $\pi_t$')
        ax[0,2].set_title(r'Nominal Interest Rate: $i_t$')

        ax[0,0].legend()

        ax[1,0].plot(states[:,:,0])
        ax[1,0].set_title(r'$u_t$')

        ax[1,1].plot(states[:,:,1])
        ax[1,1].set_title(r'$z_t$')

        ax[1,2].plot(states[:,:,2])
        ax[1,2].set_title(r'$\ln(\Gamma_t)$')

        bounds = jnp.abs(states).max() + 0.01

        for i in range(3): ax[1,i].set_ylim([-bounds, bounds])

        f.tight_layout()

        if do_save: f.savefig('plots/Simulation_example_lin_OccBin.png')

###################
# ERROR FUNCTIONS #
###################



def compute_errors(model, sigma_dict, N=50_000, compare_nn=False, key_number=42):

    from solve import construct_gh_nodes

    nn = model.nn
    par = model.par
    train = model.train
    linear = model.linear
    dtype = model.dtype

    ZLB = par["ZLB"]
    gh_n_per_shock = train["gh_n_per_shock"]
    gh_x, gh_w = construct_gh_nodes(dtype, gh_n_per_shock, sigma_dict)

    # draw states
    key = jax.random.PRNGKey(key_number)
    _, test_key = jax.random.split(key)
    states = draw_states_directly(test_key, par, dtype, N, sigma_dict["sigma_eps_u"], sigma_dict["sigma_eps_z"], sigma_dict["sigma_eps_Gamma"])

    # 1. quad nodes and weights
    eps_z = gh_x[:,1]
    eps_Gamma = gh_x[:,2]

    # 2. unpack state space: shape (N,)
    u = states[:, 0]
    z = states[:, 1]
    ln_Gamma = states[:, 2]

    # 3. compute next-period states: shape (N, gh_n)
    states_p = next_states_quad(par, dtype, states, gh_x)

    # 4. call policies
    Y, pi = eval_nn(par, train, linear, nn, states, N)
    i = taylor_rule(par, Y, pi, u, z, ln_Gamma, eps_z, eps_Gamma, ZLB, gh_w)
    Y_OccBin, pi_OccBin, i_OccBin = eval_OccBin(model, states, return_i=True)
    Y_lin, pi_lin = eval_lin(model, states, False, return_i=False)
    i_lin = taylor_rule(par, Y_lin, pi_lin, u, z, ln_Gamma, eps_z, eps_Gamma, ZLB, gh_w)

    # 5. call policies
    Y_p, pi_p = eval_nn(par, train, linear, nn, states_p, N)
    Y_p_OccBin, pi_p_OccBin, _ = eval_OccBin(model, states_p, return_i=True)
    Y_p_lin, pi_p_lin, _ = eval_lin(model, states_p, False, return_i=True)

    # 6. evaluate equilibrium equations

    ee = euler_error(par, Y, Y_p, pi, pi_p, i, u, z, eps_z, gh_w)
    nkpce = NKPC_error(par, Y, Y_p, pi, pi_p, i, u, ln_Gamma, gh_w)
    ee_OccBin = euler_error(par, Y_OccBin, Y_p_OccBin, pi_OccBin, pi_p_OccBin, i_OccBin, u, z, eps_z, gh_w)
    nkpce_OccBin = NKPC_error(par, Y_OccBin, Y_p_OccBin, pi_OccBin, pi_p_OccBin, i_OccBin, u, ln_Gamma, gh_w)
    ee_lin = euler_error(par, Y_lin, Y_p_lin, pi_lin, pi_p_lin, i_lin, u, z, eps_z, gh_w)
    nkpce_lin = NKPC_error(par, Y_lin, Y_p_lin, pi_lin, pi_p_lin, i_lin, u, ln_Gamma, gh_w)

    ZLB_nn = (i <= ZLB + 1e-5)
    ZLB_OccBin = (i_OccBin <= ZLB + 1e-5)
    ZLB_lin = (i_lin <= ZLB + 1e-5)

    errors = {
            "Baseline":  (ee,         nkpce),
            "OccBin":    (ee_OccBin,  nkpce_OccBin),
            "Linear":    (ee_lin,     nkpce_lin),
        }

    masks = [ZLB_nn, ZLB_OccBin, ZLB_lin]

    print(f"{'':12} {'Baseline':>12} {'OccBin':>12} {'Linear':>12}")
    print("-" * 52)
    for label, vals in [("Euler eq.", [v[0] for v in errors.values()]),
                        ("NKPC",      [v[1] for v in errors.values()])]:
        maes = [jnp.mean(jnp.abs(v)) for v in vals]
        print(f"{label:12} {maes[0]:>12.6f} {maes[1]:>12.6f} {maes[2]:>12.6f}")

    # ZLB binding
    print(f"\n{'ZLB binding':12} {'Baseline':>12} {'OccBin':>12} {'Linear':>12}")
    print("-" * 52)
    for label, vals, in [("Euler eq.", [v[0] for v in errors.values()]),
                         ("NKPC",      [v[1] for v in errors.values()])]:
        maes = [jnp.mean(jnp.abs(v[m])) for v, m in zip(vals, masks)]
        print(f"{label:12} {maes[0]:>12.6f} {maes[1]:>12.6f} {maes[2]:>12.6f}")

################
# DYNARE CHECK #
################

def check_dynare(model, matlab_path, sigma_IRFs):

    # print sigmas
    print(f"Loading MatLab Results with:")
    print(sigma_IRFs)
    print("\n")

    # load and print MatLab results
    matlab_results = sio.loadmat(matlab_path)
    print(matlab_results)

    # unpack Dynare results (see order in MatLab file for indexing: last index refers to shock, 2nd index to variables: 0=Y, 1=pi)
    Dynare_OccBin = matlab_results["all_results"]
    Dynare_Linear = matlab_results["all_linear_results"]

    Dynare_Y_u_pos_OccBin = Dynare_OccBin[:, 0, 0]
    Dynare_Y_u_pos_lin = Dynare_Linear[:, 0, 0]

    Dynare_Y_z_neg_OccBin = Dynare_OccBin[:, 0, 3]
    Dynare_Y_z_neg_lin = Dynare_Linear[:, 0, 3]

    Dynare_Y_ln_Gamma_pos_OccBin = Dynare_OccBin[:, 0, 4]
    Dynare_Y_ln_Gamma_pos_lin = Dynare_Linear[:, 0, 4]

    Dynare_pi_u_pos_OccBin = Dynare_OccBin[:, 1, 0]
    Dynare_pi_u_pos_lin = Dynare_Linear[:, 1, 0]

    Dynare_pi_z_neg_OccBin = Dynare_OccBin[:, 1, 3]
    Dynare_pi_z_neg_lin = Dynare_Linear[:, 1, 3]

    Dynare_pi_ln_Gamma_pos_OccBin = Dynare_OccBin[:, 1, 4]
    Dynare_pi_ln_Gamma_pos_lin = Dynare_Linear[:, 1, 4]

    # compute IRFs based on Python implementation
    model.compute_IRF(sigma_IRFs, T=20)

    # plot
    fig = plt.figure(figsize=(12, 11))
    fig.text(0.5, 0.99, 'Log-Linear', ha='center', fontsize=13, fontweight='bold')
    fig.text(0.5, 0.49, 'OccBin',     ha='center', fontsize=13, fontweight='bold')

    ax = fig.subplots(4, 3)

    ax[0,0].set_title(f'Output IRF to $u_0={sigma_IRFs["sigma_eps_u"]:.4f}$')
    ax[0,0].plot(Dynare_Y_u_pos_lin, label='Dynare')
    ax[0,0].plot(jnp.log(1+model.IRF.Y_u_lin), ls='--', label='Python-Implementation')

    ax[0,1].set_title(f'Output IRF to $z_0={sigma_IRFs["sigma_eps_z"]:.4f}$')
    ax[0,1].plot(Dynare_Y_z_neg_lin, label='Dynare')
    ax[0,1].plot(jnp.log(1+model.IRF.Y_z_lin), ls='--', label='Python-Implementation')

    ax[0,2].set_title(f'Output IRF to $\\ln(\\Gamma_0)={sigma_IRFs["sigma_eps_Gamma"]:.4f}$')
    ax[0,2].plot(Dynare_Y_ln_Gamma_pos_lin, label='Dynare')
    ax[0,2].plot(jnp.log(1+model.IRF.Y_ln_Gamma_lin), ls='--', label='Python-Implementation')

    ax[1,0].set_title(f'Inflation IRF to $u_0={sigma_IRFs["sigma_eps_u"]:.4f}$')
    ax[1,0].plot(Dynare_pi_u_pos_lin, label='Dynare')
    ax[1,0].plot(model.IRF.pi_u_lin, ls='--', label='Python-Implementation')

    ax[1,1].set_title(f'Inflation IRF to $z_0={sigma_IRFs["sigma_eps_z"]:.4f}$')
    ax[1,1].plot(Dynare_pi_z_neg_lin, label='Dynare')
    ax[1,1].plot(model.IRF.pi_z_lin, ls='--', label='Python-Implementation')

    ax[1,2].set_title(f'Inflation IRF to $\\ln(\\Gamma_0)={sigma_IRFs["sigma_eps_Gamma"]:.4f}$')
    ax[1,2].plot(Dynare_pi_ln_Gamma_pos_lin, label='Dynare')
    ax[1,2].plot(model.IRF.pi_ln_Gamma_lin, ls='--', label='Python-Implementation')

    ax[2,0].set_title(f'Output IRF to $u_0={sigma_IRFs["sigma_eps_u"]:.4f}$')
    ax[2,0].plot(Dynare_Y_u_pos_OccBin, label='Dynare')
    ax[2,0].plot(jnp.log(1+model.IRF.Y_u_OccBin), ls='--', label='Python-Implementation')

    ax[2,1].set_title(f'Output IRF to $z_0={sigma_IRFs["sigma_eps_z"]:.4f}$')
    ax[2,1].plot(Dynare_Y_z_neg_OccBin, label='Dynare')
    ax[2,1].plot(jnp.log(1+model.IRF.Y_z_OccBin), ls='--', label='Python-Implementation')

    ax[2,2].set_title(f'Output IRF to $\\ln(\\Gamma_0)={sigma_IRFs["sigma_eps_Gamma"]:.4f}$')
    ax[2,2].plot(Dynare_Y_ln_Gamma_pos_OccBin, label='Dynare')
    ax[2,2].plot(jnp.log(1+model.IRF.Y_ln_Gamma_OccBin), ls='--', label='Python-Implementation')

    ax[3,0].set_title(f'Inflation IRF to $u_0={sigma_IRFs["sigma_eps_u"]:.4f}$')
    ax[3,0].plot(Dynare_pi_u_pos_OccBin, label='Dynare')
    ax[3,0].plot(model.IRF.pi_u_OccBin, ls='--', label='Python-Implementation')

    ax[3,1].set_title(f'Inflation IRF to $z_0={sigma_IRFs["sigma_eps_z"]:.4f}$')
    ax[3,1].plot(Dynare_pi_z_neg_OccBin, label='Dynare')
    ax[3,1].plot(model.IRF.pi_z_OccBin, ls='--', label='Python-Implementation')

    ax[3,2].set_title(f'Inflation IRF to $\\ln(\\Gamma_0)={sigma_IRFs["sigma_eps_Gamma"]:.4f}$')
    ax[3,2].plot(Dynare_pi_ln_Gamma_pos_OccBin, label='Dynare')
    ax[3,2].plot(model.IRF.pi_ln_Gamma_OccBin, ls='--', label='Python-Implementation')

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(hspace=0.6)

    pairs = [
        (ax[0,0], Dynare_Y_u_pos_lin,         Dynare_Y_u_pos_OccBin),
        (ax[0,1], Dynare_Y_z_neg_lin,         Dynare_Y_z_neg_OccBin),
        (ax[0,2], Dynare_Y_ln_Gamma_pos_lin,  Dynare_Y_ln_Gamma_pos_OccBin),
        (ax[1,0], Dynare_pi_u_pos_lin,        Dynare_pi_u_pos_OccBin),
        (ax[1,1], Dynare_pi_z_neg_lin,        Dynare_pi_z_neg_OccBin),
        (ax[1,2], Dynare_pi_ln_Gamma_pos_lin, Dynare_pi_ln_Gamma_pos_OccBin),
        (ax[2,0], jnp.log(1+model.IRF.Y_u_lin),         jnp.log(1+model.IRF.Y_u_OccBin)),
        (ax[2,1], jnp.log(1+model.IRF.Y_z_lin),         jnp.log(1+model.IRF.Y_z_OccBin)),
        (ax[2,2], jnp.log(1+model.IRF.Y_ln_Gamma_lin),  jnp.log(1+model.IRF.Y_ln_Gamma_OccBin)),
        (ax[3,0], model.IRF.pi_u_lin,         model.IRF.pi_u_OccBin),
        (ax[3,1], model.IRF.pi_z_lin,         model.IRF.pi_z_OccBin),
        (ax[3,2], model.IRF.pi_ln_Gamma_lin,  model.IRF.pi_ln_Gamma_OccBin),
    ]

    shade_patch = mpatches.Patch(color='grey', alpha=0.3, label='ZLB binding for OccBin')

    for a, lin, occ in pairs:
        diff = np.abs(np.array(lin) - np.array(occ))
        mask = diff > 1e-6
        x = np.arange(len(lin))
        ymin, ymax = a.get_ylim()
        a.fill_between(x, ymin, ymax, where=mask, color='grey', alpha=0.3, zorder=0)
        handles, labels = a.get_legend_handles_labels()
        a.legend(handles=handles + [shade_patch], fontsize=7)

    plt.show()