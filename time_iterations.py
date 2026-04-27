import numpy as np
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import random
import time
import os
import matplotlib.pyplot as plt

from neural_nets import Policy


from scipy import optimize
from scipy import interpolate
from scipy.stats import norm
import copy
import model_funcs

# interpolate next-period policies given quadrature nodes
def pol_plus(par, train, n_grid, u_grid, ln_beta_grid, ln_Gamma_grid, x):

    gh_n = train['gh_n']

    Y_pol = x[:n_grid**3].reshape(n_grid, n_grid, n_grid)
    pi_pol = x[n_grid**3:].reshape(n_grid, n_grid, n_grid)

    Y_inp = interpolate.RegularGridInterpolator((u_grid,ln_beta_grid,ln_Gamma_grid), Y_pol, bounds_error=False, fill_value=None)
    pi_inp = interpolate.RegularGridInterpolator((u_grid,ln_beta_grid,ln_Gamma_grid), pi_pol, bounds_error=False, fill_value=None)

    u_grid_ = u_grid[:, None, None, None, None, None]
    ln_beta_grid_ = ln_beta_grid[None, :, None, None, None, None]
    ln_Gamma_grid_ = ln_Gamma_grid[None, None, :, None, None, None]
    eps_i_gh = train["gh_nodes_eps_i"][None, None, None, :, None, None]
    eps_beta_gh = train["gh_nodes_eps_beta"][None, None, None, None, :, None]
    eps_Gamma_gh = train["gh_nodes_eps_Gamma"][None, None, None, None, None, :]

    u_grid_next = par['rho_i']*u_grid_ + eps_i_gh
    ln_beta_grid_next = par["rho_beta"]*ln_beta_grid_ + eps_beta_gh
    ln_Gamma_grid_next = par["rho_Gamma"]*ln_Gamma_grid_ + eps_Gamma_gh

    pts = np.stack(np.broadcast_arrays(u_grid_next, ln_beta_grid_next, ln_Gamma_grid_next),axis=-1)

    Y_p = Y_inp(pts).reshape(n_grid, n_grid, n_grid, gh_n, gh_n, gh_n)
    pi_p = pi_inp(pts).reshape(n_grid, n_grid, n_grid, gh_n, gh_n, gh_n)

    return Y_p, pi_p

def errors(par, DSS, train, u_grid, ln_beta_grid, ln_Gamma_grid, Y_pol, pi_pol, Y_pol_p, pi_pol_p):

    gh_w = train['gh_weights']
    gh_weights = gh_w[None, None, None, :, None, None] * gh_w[None, None, None, None, :, None] * gh_w[None, None, None, None, None, :]

    euler_errors = model_funcs.euler_error(par, DSS, Y_pol, Y_pol_p, pi_pol, pi_pol_p, u_grid, ln_beta_grid, gh_weights)
    nkpc_errors = model_funcs.NKPC_error(par, DSS, pi_pol, pi_pol_p, Y_pol, Y_pol_p, u_grid, ln_beta_grid, ln_Gamma_grid, gh_weights)

    return np.concatenate([euler_errors.flatten(), nkpc_errors.flatten()])

def compute_SSS(u_grid, n_grid, x):

    Y_pol = x[0]
    pi_pol = x[1]
    
    Y_inp = interpolate.RegularGridInterpolator((u_grid,), Y_pol, bounds_error=False, fill_value=None)
    pi_inp = interpolate.RegularGridInterpolator((u_grid,), pi_pol, bounds_error=False, fill_value=None)
    
    return Y_inp([0]), pi_inp([0])

def unpack(x, n_grid):
    Y_pol = x[:n_grid**3].reshape(n_grid, n_grid, n_grid)
    pi_pol = x[n_grid**3:].reshape(n_grid, n_grid, n_grid)
    return Y_pol, pi_pol

def solve(model, k, do_print=True):

    par = model.par
    train = model.train
    DSS = model.DSS

    # construct grids
    n_grid = 10
    unit_grid = np.linspace(0.0001,0.9999,n_grid)
    u_grid = norm.ppf(unit_grid, loc=0, scale=par['sigma_eps_i']/(np.sqrt(1-par['rho_i']**2)))
    ln_beta_grid = norm.ppf(unit_grid, loc=np.log(par['beta_SSS']), scale=par['sigma_beta']/(np.sqrt(1-par['rho_beta']**2)))
    ln_Gamma_grid = norm.ppf(unit_grid, loc=np.log(par['beta_SSS']), scale=par['sigma_Gamma']/(np.sqrt(1-par['rho_Gamma']**2)))

    # aggregate policy guesses
    Y_pol = np.ones((n_grid,n_grid,n_grid)) # (u, beta, Gamma)
    pi_pol = np.zeros((n_grid,n_grid,n_grid))
    x0 = np.concatenate([Y_pol.flatten(), pi_pol.flatten()]) # (variable, u, beta, Gamma)

    # time iterate
    for k in range(k):

        if k > 0: x0 = copy.copy(x_new)

        Y_pol_p, pi_pol_p = pol_plus(par, train, n_grid, u_grid, ln_beta_grid, ln_Gamma_grid, x0)
        obj = lambda x: errors(par, DSS, train, u_grid, ln_beta_grid, ln_Gamma_grid, *unpack(x, n_grid), Y_pol_p, pi_pol_p)

        res = optimize.root(obj, x0)
        x_new = res.x
        print(x_new)
        if do_print: print(f'Iteration {k}:\t max change = {max(abs(x_new-x0)):.8f}\t root-finding: {res.success}')

    # root finder
    obj = lambda x: errors(par, DSS, train, u_grid, x[:n_grid], x[n_grid:], *(pol_plus(par, train, n_grid, u_grid, x)))
    x_new = res.x
    res = optimize.root(obj, x_new, tol=1e-12)
    Y_pol, pi_pol = res.x[:n_grid], res.x[n_grid:]
    Y_SSS, pi_SSS = compute_SSS(u_grid, n_grid, res.x)
    
    if do_print:
        print(f'\nFinal root-finder: {res.success}')
        print(f'\nSSS:\t Y = {Y_SSS[0]:.4f}\t pi = {pi_SSS[0]:.4f}')
    
    time_iterations = SimpleNamespace()
    time_iterations.u_grid = u_grid
    time_iterations.n_grid = n_grid
    time_iterations.Y_pol = Y_pol
    time_iterations.pi_pol = pi_pol

    model.time_iterations = time_iterations

def plot_agg_policies(model):

    time_iterations = model.time_iterations
    u_grid = time_iterations.u_grid

    f, ax = plt.subplots(1,2, figsize=(12,5))

    ax[0].set_title('$Y$')
    ax[0].plot(u_grid, time_iterations.Y_pol)
    
    ax[1].set_title('$\pi$')
    ax[1].plot(u_grid, time_iterations.pi_pol)
    
    for i in range(2): ax[i].set_xlabel('$u$')