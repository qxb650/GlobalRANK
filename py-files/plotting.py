import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax
from flax.core import FrozenDict
import matplotlib.ticker as mticker
import pandas as pd
from scipy.stats import chi2
from matplotlib.patches import Ellipse

import solve
import aux_ as aux
import RANK
from model_funcs import taylor_rule
from neural_nets import eval_nn
from linear import eval_OccBin_womodel, eval_lin_womodel

def plot_policies(model, sigma_dict, N_grid=50, N_mc=10000, std_low=-3, std_high=3, key_number=42):

    par = FrozenDict(model.par)
    linear = FrozenDict(model.linear)
    train = FrozenDict(model.train)
    nn = model.nn

    ZLB = par["ZLB"]

    # prep for grids and compute state ergodic std
    s_prep = jnp.linspace(std_low, std_high, N_grid)[:, None]
    std_u = (sigma_dict["sigma_eps_u"]/jnp.sqrt(1-par["rho_u"]**2))
    std_z = (sigma_dict["sigma_eps_z"]/jnp.sqrt(1-par["rho_z"]**2))
    std_ln_Gamma = (sigma_dict["sigma_eps_Gamma"]/jnp.sqrt(1-par["rho_Gamma"]**2))

    # pure 1D grids
    u_grid = std_u*s_prep
    z_grid = std_z*s_prep
    ln_Gamma_grid = std_ln_Gamma*s_prep

    # 3D states grid for non-integrated policies
    s_u = jnp.concat([u_grid, jnp.zeros((N_grid,2))], axis=-1)
    s_z = jnp.concat([jnp.zeros((N_grid,1)), z_grid, jnp.zeros((N_grid,1))], axis=-1)
    s_ln_Gamma = jnp.concat([jnp.zeros((N_grid,2)), ln_Gamma_grid], axis=-1)

    # policies when no mc on grids
    Y_u, pi_u = eval_nn(par, train, linear, nn, s_u, N_grid)
    i_u = taylor_rule(par, Y_u, pi_u, u_grid.flatten(), 0.00, 0.00, 0.00, 0.00, ZLB, 0.00)
    
    Y_z, pi_z = eval_nn(par, train, linear, nn, s_z, N_grid)
    i_z = taylor_rule(par, Y_z, pi_z, 0.00, s_z, 0.00, 0.00, 0.00, ZLB, 0.00)
    
    Y_ln_Gamma, pi_ln_Gamma = eval_nn(par, train, linear, nn, s_ln_Gamma, N_grid)
    i_ln_Gamma = taylor_rule(par, Y_ln_Gamma, pi_ln_Gamma, 0.00, 0.00, ln_Gamma_grid[:, 0], 0.00, 0.00, ZLB, 0.00)

    # drawing mc
    key = jax.random.key(key_number)
    draws = jax.random.normal(key, shape=(N_mc*N_grid, 3))
    u_mc = std_u*draws[:, 0, None] # (N_mc*N_grid, 1)
    z_mc = std_z*draws[:, 1, None]
    ln_Gamma_mc = std_ln_Gamma*draws[:, 2, None]

    # 3D state grids with mc in non-policy dimensions
    s_u_mc = jnp.concat([jnp.tile(u_grid, (N_mc, 1)), z_mc, ln_Gamma_mc], axis=-1) # (N_mc*N_grid, 3)
    s_z_mc = jnp.concat([u_mc, jnp.tile(z_grid, (N_mc, 1)), ln_Gamma_mc], axis=-1)
    s_ln_Gamma_mc = jnp.concat([u_mc, z_mc, jnp.tile(ln_Gamma_grid, (N_mc, 1))], axis=-1)

    # MP shock
    Y_u_mc, pi_u_mc = eval_nn(par, train, linear, nn, s_u_mc, N_grid*N_mc)
    i_u_mc = taylor_rule(par, Y_u_mc, pi_u_mc, s_u_mc[..., 0], s_u_mc[..., 1], s_u_mc[..., 2], 0.00, 0.00, -0.005, 0.00)
    Y_u_mc, pi_u_mc, i_u_mc = Y_u_mc.reshape(N_mc, N_grid), pi_u_mc.reshape(N_mc, N_grid), i_u_mc.reshape(N_mc, N_grid)

    Y_u_mc_mean = jnp.mean(Y_u_mc, axis=0)
    pi_u_mc_mean = jnp.mean(pi_u_mc, axis=0)
    i_u_mc_mean = jnp.mean(i_u_mc, axis=0)
    Y_u_mc_lower = jnp.percentile(Y_u_mc, 2.5, axis=0)
    Y_u_mc_upper = jnp.percentile(Y_u_mc, 97.5, axis=0)
    i_u_mc_lower = jnp.percentile(i_u_mc, 2.5, axis=0)
    i_u_mc_upper = jnp.percentile(i_u_mc, 97.5, axis=0)

    # z
    Y_z_mc, pi_z_mc = eval_nn(par, train, linear, nn, s_z_mc, N_grid*N_mc)
    i_z_mc = taylor_rule(par, Y_z_mc, pi_z_mc, s_z_mc[..., 0], s_z_mc[..., 1], s_z_mc[..., 2], 0.00, 0.00, -0.005, 0.00)
    Y_z_mc, pi_z_mc, i_z_mc = Y_z_mc.reshape(N_mc, N_grid), pi_z_mc.reshape(N_mc, N_grid), i_z_mc.reshape(N_mc, N_grid)
    Y_z_mc_mean = jnp.mean(Y_z_mc, axis=0)
    pi_z_mc_mean = jnp.mean(pi_z_mc, axis=0)
    i_z_mc_mean = jnp.mean(i_z_mc, axis=0)
    i_z_mc_lower = jnp.percentile(i_z_mc, 2.5, axis=0)
    i_z_mc_upper = jnp.percentile(i_z_mc, 97.5, axis=0)

    # ln Gamma
    Y_ln_Gamma_mc, pi_ln_Gamma_mc = eval_nn(par, train, linear, nn, s_ln_Gamma_mc, N_grid*N_mc)
    i_ln_Gamma_mc = taylor_rule(par, Y_ln_Gamma_mc, pi_ln_Gamma_mc, s_ln_Gamma_mc[..., 0], s_ln_Gamma_mc[..., 1], s_ln_Gamma_mc[..., 2], 0.00, 0.00, -0.005, 0.00)
    Y_ln_Gamma_mc, pi_ln_Gamma_mc, i_ln_Gamma_mc = Y_ln_Gamma_mc.reshape(N_mc, N_grid), pi_ln_Gamma_mc.reshape(N_mc, N_grid), i_ln_Gamma_mc.reshape(N_mc, N_grid)
    Y_ln_Gamma_mc_mean = jnp.mean(Y_ln_Gamma_mc, axis=0)
    pi_ln_Gamma_mc_mean = jnp.mean(pi_ln_Gamma_mc, axis=0)
    i_ln_Gamma_mc_mean = jnp.mean(i_ln_Gamma_mc, axis=0)

    i_ln_Gamma_mc_lower = jnp.percentile(i_ln_Gamma_mc, 2.5, axis=0)
    i_ln_Gamma_mc_upper = jnp.percentile(i_ln_Gamma_mc, 97.5, axis=0)

    f, ax = plt.subplots(3,3, figsize=(15,10))

    ax[0,0].plot(s_u[..., 0], Y_u)
    ax[0,0].plot(s_u[..., 0], Y_u_mc_mean)
    ax[0,1].plot(s_z[..., 1], Y_z)
    ax[0,1].plot(s_z[..., 1], Y_z_mc_mean)
    ax[0,2].plot(s_ln_Gamma[..., 2], Y_ln_Gamma)
    ax[0,2].plot(s_ln_Gamma[..., 2], Y_ln_Gamma_mc_mean)

    ax[1,0].plot(s_u[..., 0], pi_u)
    ax[1,0].plot(s_u[..., 0], pi_u_mc_mean)
    ax[1,1].plot(s_z[..., 1], pi_z)
    ax[1,1].plot(s_z[..., 1], pi_z_mc_mean)
    ax[1,2].plot(s_ln_Gamma[..., 2], pi_ln_Gamma)
    ax[1,2].plot(s_ln_Gamma[..., 2], pi_ln_Gamma_mc_mean)

    ax[2,0].plot(s_u[..., 0], i_u_mc_mean)
    ax[2,0].fill_between(s_u[..., 0], i_u_mc_lower, i_u_mc_upper, color='purple', alpha=0.15)
    ax[2,0].plot(u_grid, i_u)
    ax[2,1].plot(s_z[..., 1], i_z_mc_mean)
    ax[2,1].fill_between(s_z[..., 1], i_z_mc_lower, i_z_mc_upper, color='purple', alpha=0.15)
    ax[2,1].plot(z_grid, i_z)
    ax[2,2].plot(s_ln_Gamma[..., 2], i_ln_Gamma_mc_mean)
    ax[2,2].plot(ln_Gamma_grid, i_ln_Gamma)
    ax[2,2].fill_between(s_ln_Gamma[..., 2], i_ln_Gamma_mc_lower, i_ln_Gamma_mc_upper, color='purple', alpha=0.15)

def plot_i(model, sigma_dict, N_grid=50, N_mc=10000, std_low=-3, std_high=3, key_number=42, save_path=None):

    par = FrozenDict(model.par)
    linear = FrozenDict(model.linear)
    train = FrozenDict(model.train)
    nn = model.nn

    plt.rcParams.update({'font.size': 15})

    ZLB = par["ZLB"]

    # prep for grids and compute state ergodic std
    s_prep = jnp.linspace(std_low, std_high, N_grid)[:, None]
    std_u = (sigma_dict["sigma_eps_u"]/jnp.sqrt(1-par["rho_u"]**2))
    std_z = (sigma_dict["sigma_eps_z"]/jnp.sqrt(1-par["rho_z"]**2))
    std_ln_Gamma = (sigma_dict["sigma_eps_Gamma"]/jnp.sqrt(1-par["rho_Gamma"]**2))

    # pure 1D grids
    u_grid = std_u*s_prep
    z_grid = std_z*s_prep
    ln_Gamma_grid = std_ln_Gamma*s_prep

    # 3D states grid for non-integrated policies
    s_z = jnp.concat([jnp.zeros((N_grid,1)), z_grid, jnp.zeros((N_grid,1))], axis=-1)
    s_ln_Gamma = jnp.concat([jnp.zeros((N_grid,2)), ln_Gamma_grid], axis=-1)

    # policies when no mc on grids
    Y_z, pi_z = eval_nn(par, train, linear, nn, s_z, N_grid)
    i_z = taylor_rule(par, Y_z, pi_z, 0.00, s_z, 0.00, 0.00, 0.00, ZLB, 0.00)
    Y_ln_Gamma, pi_ln_Gamma = eval_nn(par, train, linear, nn, s_ln_Gamma, N_grid)
    i_ln_Gamma = taylor_rule(par, Y_ln_Gamma, pi_ln_Gamma, 0.00, 0.00, ln_Gamma_grid[:, 0], 0.00, 0.00, ZLB, 0.00)

    # drawing mc
    key = jax.random.key(key_number)
    draws = jax.random.normal(key, shape=(N_mc*N_grid, 3))
    u_mc = std_u*draws[:, 0, None] # (N_mc*N_grid, 1)
    z_mc = std_z*draws[:, 1, None]
    ln_Gamma_mc = std_ln_Gamma*draws[:, 2, None]

    # 3D state grids with mc in non-policy dimensions
    s_z_mc = jnp.concat([u_mc, jnp.tile(z_grid, (N_mc, 1)), ln_Gamma_mc], axis=-1)
    s_ln_Gamma_mc = jnp.concat([u_mc, z_mc, jnp.tile(ln_Gamma_grid, (N_mc, 1))], axis=-1)

    # z
    Y_z_mc, pi_z_mc = eval_nn(par, train, linear, nn, s_z_mc, N_grid*N_mc)
    i_z_mc = taylor_rule(par, Y_z_mc, pi_z_mc, s_z_mc[..., 0], s_z_mc[..., 1], s_z_mc[..., 2], 0.00, 0.00, -0.005, 0.00)
    Y_z_mc, pi_z_mc, i_z_mc = Y_z_mc.reshape(N_mc, N_grid), pi_z_mc.reshape(N_mc, N_grid), i_z_mc.reshape(N_mc, N_grid)
    Y_z_mc_mean = jnp.mean(Y_z_mc, axis=0)
    pi_z_mc_mean = jnp.mean(pi_z_mc, axis=0)
    i_z_mc_mean = jnp.mean(i_z_mc, axis=0)
    i_z_mc_lower = jnp.percentile(i_z_mc, 2.5, axis=0)
    i_z_mc_upper = jnp.percentile(i_z_mc, 97.5, axis=0)

    _, _, i_z_mc_OccBin = eval_OccBin_womodel(par, linear, s_z_mc, return_i=True)
    i_z_mc_OccBin = i_z_mc_OccBin.reshape(N_mc, N_grid)
    i_z_mc_OccBin_mean = jnp.mean(i_z_mc_OccBin, axis=0)
    i_z_mc_OccBin_lower = jnp.percentile(i_z_mc_OccBin, 2.5, axis=0)
    i_z_mc_OccBin_upper = jnp.percentile(i_z_mc_OccBin, 97.5, axis=0)

    # ln Gamma
    Y_ln_Gamma_mc, pi_ln_Gamma_mc = eval_nn(par, train, linear, nn, s_ln_Gamma_mc, N_grid*N_mc)
    i_ln_Gamma_mc = taylor_rule(par, Y_ln_Gamma_mc, pi_ln_Gamma_mc, s_ln_Gamma_mc[..., 0], s_ln_Gamma_mc[..., 1], s_ln_Gamma_mc[..., 2], 0.00, 0.00, -0.005, 0.00)
    Y_ln_Gamma_mc, pi_ln_Gamma_mc, i_ln_Gamma_mc = Y_ln_Gamma_mc.reshape(N_mc, N_grid), pi_ln_Gamma_mc.reshape(N_mc, N_grid), i_ln_Gamma_mc.reshape(N_mc, N_grid)
    Y_ln_Gamma_mc_mean = jnp.mean(Y_ln_Gamma_mc, axis=0)
    pi_ln_Gamma_mc_mean = jnp.mean(pi_ln_Gamma_mc, axis=0)
    i_ln_Gamma_mc_mean = jnp.mean(i_ln_Gamma_mc, axis=0)

    i_ln_Gamma_mc_lower = jnp.percentile(i_ln_Gamma_mc, 2.5, axis=0)
    i_ln_Gamma_mc_upper = jnp.percentile(i_ln_Gamma_mc, 97.5, axis=0)

    _, _, i_ln_Gamma_mc_OccBin = eval_OccBin_womodel(par, linear, s_ln_Gamma_mc, return_i=True)
    i_ln_Gamma_mc_OccBin = i_ln_Gamma_mc_OccBin.reshape(N_mc, N_grid)
    i_ln_Gamma_mc_OccBin_mean = jnp.mean(i_ln_Gamma_mc_OccBin, axis=0)
    i_ln_Gamma_mc_OccBin_lower = jnp.percentile(i_ln_Gamma_mc_OccBin, 2.5, axis=0)
    i_ln_Gamma_mc_OccBin_upper = jnp.percentile(i_ln_Gamma_mc_OccBin, 97.5, axis=0)

    # boundary plot
    z_grid_meshed, ln_Gamma_grid_meshed = jnp.meshgrid(z_grid.flatten(), ln_Gamma_grid.flatten(), indexing='ij') # (N_grid, N_grid)
    keys_grid = jax.random.split(key, num=N_grid * N_grid)

    z_flat = z_grid_meshed.flatten() # (N_grid*N_grid, )
    ln_Gamma_flat = ln_Gamma_grid_meshed.flatten()

    def scan_body(carry, xs):
        
        # unpack
        z, ln_Gamma, key_ = xs
        
        # draw MP shocks for mc
        u_mc_ = (sigma_dict["sigma_eps_u"]/jnp.sqrt(1-par["rho_u"]**2))*jax.random.normal(key_, shape=(N_mc, 1)) # (N_mc, )

        s = jnp.concat([
            u_mc_,
            jnp.full((N_mc, 1), z),
            jnp.full((N_mc, 1), ln_Gamma)
        ], axis=-1) # (N_mc, 3)

        Y, pi = eval_nn(par, train, linear, nn, s, N_mc) # (N_mc, )
        _, _, i_OccBin = eval_OccBin_womodel(par, linear, s, return_i=True)
        # eval_lin_womodel(par, linear, s, False, return_i=True) #
        z_vec = jnp.full((N_mc,), z)
        ln_Gamma_vec = jnp.full((N_mc,), ln_Gamma)
        
        i = taylor_rule(par, Y, pi, u_mc_.flatten(), z_vec, ln_Gamma_vec, 0.00, 0.00, ZLB, 0.00)
        i_mean = jnp.mean(i)
        i_lower = jnp.percentile(i, 2.5)
        i_upper = jnp.percentile(i, 97.5)

        i_OccBin_mean = jnp.mean(i_OccBin)

        return None, (i_mean, i_lower, i_upper, i_OccBin_mean)

    _, out_flat = jax.lax.scan(scan_body, None, (z_flat, ln_Gamma_flat, keys_grid))

    s_z_ln_Gamma = jnp.concat([
        jnp.zeros((N_grid*N_grid, 1)),
        z_flat[:, None],
        ln_Gamma_flat[:, None]
    ], axis=-1)

    Y, pi = eval_nn(par, train, linear, nn, s_z_ln_Gamma, N_mc)
    i = taylor_rule(par, Y, pi, s_z_ln_Gamma[..., 0], s_z_ln_Gamma[..., 1], s_z_ln_Gamma[..., 2], 0.00, 0.00, ZLB, 0.00)

    _, _, i_OccBin = eval_OccBin_womodel(par, linear, s_z_ln_Gamma, return_i=True)

    # unpack and reshape to grids
    i = i.reshape(N_grid, N_grid)
    i_OccBin = i_OccBin.reshape(N_grid, N_grid)
    i_mean = out_flat[0].reshape(N_grid, N_grid)
    i_lower = out_flat[1].reshape(N_grid, N_grid)
    i_upper = out_flat[2].reshape(N_grid, N_grid)
    i_OccBin_mean = out_flat[3].reshape(N_grid, N_grid)

    zlb_grid_no_mc = (i <= ZLB + 1e-5).astype(float) #jnp.isclose(i, ZLB)
    zlb_grid = (i_mean <= ZLB + 1e-5).astype(float) #jnp.isclose(i_mean, ZLB)
    zlb_grid_lower = (i_lower <= ZLB + 1e-5).astype(float) #jnp.isclose(i_lower, ZLB)
    zlb_grid_upper = (i_upper <= ZLB + 1e-5).astype(float) #jnp.isclose(i_upper, ZLB)
    zlb_grid_OccBin = (i_OccBin_mean <= ZLB + 1e-5).astype(float)
    zlb_grid_OccBin_no_mc = (i_OccBin <= ZLB + 1e-5).astype(float)

    def zlb_threshold(zlb, grid, axis, take_min=True):
        n = zlb.shape[1-axis]
        thresholds = []
        for k in range(n):
            slc = zlb[k, :] if axis == 1 else zlb[:, k]
            g   = grid[k, :] if axis == 1 else grid[:, k]
            idx = np.where(slc == 1)[0]
            thresholds.append(g[idx.min() if take_min else idx.max()] if len(idx) > 0 else np.nan)
        return np.array(thresholds)

    thresh_deqn_lG   = zlb_threshold(zlb_grid,        ln_Gamma_grid_meshed, axis=1, take_min=True)
    thresh_occbin_lG = zlb_threshold(zlb_grid_OccBin, ln_Gamma_grid_meshed, axis=1, take_min=True)

    thresh_deqn_z    = zlb_threshold(zlb_grid,        z_grid_meshed, axis=0, take_min=False)
    thresh_occbin_z  = zlb_threshold(zlb_grid_OccBin, z_grid_meshed, axis=0, take_min=False)

    print('ln_Gamma: ', pd.Series(thresh_deqn_lG   - thresh_occbin_lG).mean())
    print('z: ', pd.Series(thresh_deqn_z    - thresh_occbin_z).mean())

    f, ax = plt.subplots(1,4,figsize=(15,5))

    ax[0].contourf(z_grid_meshed, ln_Gamma_grid_meshed, zlb_grid, levels=[-0.5, 0.5, 1.5], colors=['skyblue', 'green'])
    ax[0].contour(z_grid_meshed, ln_Gamma_grid_meshed, zlb_grid, levels=[0.5], colors='red', linewidths=2)
    #ax[0].contour(z_grid_meshed, ln_Gamma_grid_meshed, zlb_grid_no_mc, levels=[0.5], colors='red', linewidths=2, linestyles='dashed')
    ax[0].contour(z_grid_meshed, ln_Gamma_grid_meshed, zlb_grid_OccBin, levels=[0.5], colors='purple', linewidths=2, linestyles='dashed')
    #ax[0].contour(z_grid_meshed, ln_Gamma_grid_meshed, zlb_grid_OccBin_no_mc, levels=[0.5], colors='purple', linewidths=2, linestyles='dashdot')
    ax[0].scatter(0.00, 0.00, color='blue') # , label='DSS & SSS'
    ax[0].text(0.005, -0.005, 'DSS', fontweight='bold', color='blue', fontsize=12)

    # proxy lines for plotting in the contour plot
    ax[0].plot([], [], color='red', label='DEQN', linewidth=2)
    #ax[0].plot([], [], color='red', label='Non-Int. DEQN', linewidth=2, ls='dashed')
    ax[0].plot([], [], color='purple', label='OccBin', linewidth=2, ls='dashed')
    #ax[0].plot([], [], color='purple', label='Non-Int. OccBin', linewidth=2, ls='dashdot')

    #ax[0].contour(z_grid_meshed, ln_Gamma_grid_meshed, zlb_grid_lower, levels=[0.5], colors='orange', linewidths=2)
    #ax[0].contour(z_grid_meshed, ln_Gamma_grid_meshed, zlb_grid_upper, levels=[0.5], colors='orange', linewidths=2)

    ax[0].text(-0.085, 0.04, 'ZLB binding', fontweight='bold', fontsize=12)
    ax[0].text(0.02, 0.04, 'ZLB slack', fontweight='bold', fontsize=12)

    ax[0].set_xlabel(r'$z_t$')
    ax[0].set_ylabel(r'$\ln(\Gamma_t)$')

    ax[1].plot(s_z[..., 1], 100*i_z_mc_mean, color='red') # label='DEQN'
    ax[1].fill_between(s_z[..., 1], 100*i_z_mc_lower, 100*i_z_mc_upper, color='red', alpha=0.15, label='DEQN: 95 per.')
    #ax[1].plot(z_grid, i_z, ls='--', label='DEQN')
    ax[1].hlines(100*ZLB, z_grid.min(), z_grid.max(), color='gray', ls='--')
    ax[1].hlines(100*model.par["i_DSS"], z_grid.min(), z_grid.max(), color='blue', ls='--')
    ax[1].text(0.05, 100*(-0.009), r'$i=ZLB$', color='gray', fontsize=14)
    ax[1].text(0.05, 100*0.005, r'$i^{DSS}$', color='blue', fontsize=18)

    ax[1].plot(s_z[..., 1], 100*i_z_mc_OccBin_mean, color="purple", ls='--') # , label='OccBin'
    ax[1].fill_between(s_z[..., 1], 100*i_z_mc_OccBin_lower, 100*i_z_mc_OccBin_upper, color='purple', alpha=0.15, label='OccBin: 95 per.')

    ax[1].set_xlabel(r'$z_t$')

    ax[2].plot(s_ln_Gamma[..., 2], 100*i_ln_Gamma_mc_mean, color='red') # , label='DEQN'
    #ax[2].plot(ln_Gamma_grid, i_ln_Gamma, ls='--', label='DEQN')
    ax[2].fill_between(s_ln_Gamma[..., 2], 100*i_ln_Gamma_mc_lower, 100*i_ln_Gamma_mc_upper, color='red', alpha=0.15) # , label='DEQN: 95 per.'
    ax[2].hlines(100*ZLB, ln_Gamma_grid.min(), ln_Gamma_grid.max(), color='gray', ls='--')
    ax[2].hlines(100*model.par["i_DSS"], ln_Gamma_grid.min(), ln_Gamma_grid.max(), color='blue', ls='--')
    ax[2].text(0.03, 100*(-0.009), r'$i=ZLB$', color='gray', fontsize=14)
    ax[2].text(0.03, 100*0.005, r'$i^{DSS}$', color='blue', fontsize=18)

    ax[2].plot(s_ln_Gamma[..., 2], 100*i_ln_Gamma_mc_OccBin_mean, color="purple", ls='--') # , label='OccBin'
    ax[2].fill_between(s_ln_Gamma[..., 2], 100*i_ln_Gamma_mc_OccBin_lower, 100*i_ln_Gamma_mc_OccBin_upper, color="purple", alpha=0.15) # , label='OccBin: 95 per.'

    ax[2].set_xlabel(r'$\ln(\Gamma_t)$')

    ax[0].sharex(ax[1])
    #ax[0].legend(loc='lower right')

    std_z        = 0.03271
    std_ln_Gamma = 0.01608

    for pct, alpha, color in zip([0.55, 0.80, 0.95], [1, 1, 1], ['yellow', 'brown', 'magenta']):
        chi2_val = np.sqrt(chi2.ppf(pct, df=2))
        ellipse = Ellipse(xy=(0, 0),
                        width=2*chi2_val*std_z,
                        height=2*chi2_val*std_ln_Gamma,
                        edgecolor=color, facecolor='none',
                        linestyle='--', linewidth=1, alpha=alpha)
        ax[0].add_patch(ellipse)

    ax[2].plot([], [], color='yellow', linestyle='--', label =  r'55 per. band for $(z_t,\ln(\Gamma_t))$')
    ax[2].plot([], [], color='brown', linestyle='--', label = r'80 per. band for $(z_t,\ln(\Gamma_t))$')
    ax[2].plot([], [], color='magenta', linestyle='--', label = r'95 per. band for $(z_t,\ln(\Gamma_t))$')

    for i in range(3):
        ax[i].xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax[i].tick_params(axis='x', labelrotation=45)

    for i in range(1,3):
        ax[i].set_ylim([100*(-0.011), 100*0.05])
        ax[i].set_ylabel('%')

    ax[0].set_title(r'$\mathbb{E}[i(u_t,z_t,\ln(\Gamma_t)) \vert z_t, \ln(\Gamma_t)]=\text{ZLB}$')
    ax[1].set_title(r'$i(u_t,z_t,\ln(\Gamma_t)) \vert z_t$')
    ax[2].set_title(r'$i(u_t,z_t,\ln(\Gamma_t)) \vert \ln(\Gamma_t)$')

    f.legend(loc='lower center', ncols=4, bbox_to_anchor=(0.5, -0.15))

    f.tight_layout()

    if save_path is not None:
            f.savefig(save_path, bbox_inches='tight')



def plot_boundaries(model, sigma_dict, N_grid=50, N_mc=10000, std_low=-3, std_high=3, key_number=42, save_path=None):

    par = FrozenDict(model.par)
    linear = FrozenDict(model.linear)
    train = FrozenDict(model.train)
    nn = model.nn

    plt.rcParams.update({'font.size': 15})

    ZLB = par["ZLB"]

    # prep for grids and compute state ergodic std
    s_prep = jnp.linspace(std_low, std_high, N_grid)
    std_u = (sigma_dict["sigma_eps_u"]/jnp.sqrt(1-par["rho_u"]**2))
    std_z = (sigma_dict["sigma_eps_z"]/jnp.sqrt(1-par["rho_z"]**2))
    std_ln_Gamma = (sigma_dict["sigma_eps_Gamma"]/jnp.sqrt(1-par["rho_Gamma"]**2))

    # pure 1D grids
    u_grid = std_u*s_prep
    z_grid = std_z*s_prep
    ln_Gamma_grid = std_ln_Gamma*s_prep

    # no u
    z_nou_mesh, ln_Gamma_nou_mesh = jnp.meshgrid(z_grid, ln_Gamma_grid)
    states_nou = jnp.concat([z_nou_mesh[..., None], ln_Gamma_nou_mesh[..., None]], axis=-1) # (N_grid, N_grid, 2)
    states_nou = states_nou.reshape(-1, 2) # (N_grid*N_grid, 2)
    states_nou = jnp.concat([jnp.zeros((N_grid*N_grid, 1)), states_nou], axis=-1)

    Y_nou, pi_nou = eval_nn(par, train, linear, nn, states_nou, N_grid*N_grid)
    Y_nou, pi_nou = Y_nou.reshape(N_grid, N_grid), pi_nou.reshape(N_grid, N_grid)
    states_nou = states_nou.reshape(N_grid, N_grid, 3)
    i_nou = taylor_rule(par, Y_nou, pi_nou, states_nou[..., 0], states_nou[..., 1], states_nou[..., 2], 0.00, 0.00, ZLB, 0.00)
    ZLB_nou = (i_nou <= ZLB + 1e-5).astype(float)

    Y_OccBin_nou, pi_OccBin_nou, i_OccBin_nou = eval_OccBin_womodel(par, linear, states_nou, return_i=True)
    ZLB_OccBin_nou = (i_OccBin_nou <= ZLB + 1e-5).astype(float)

    f, ax = plt.subplots(1,3,figsize=(15,5))

    ax[0].contourf(z_nou_mesh, ln_Gamma_nou_mesh, ZLB_nou, levels=[-0.5, 0.5, 1.5], colors=['skyblue', 'green'])
    ax[0].contour(z_nou_mesh, ln_Gamma_nou_mesh, ZLB_nou, levels=[0.5], colors='red', linewidths=2)
    ax[0].contour(z_nou_mesh, ln_Gamma_nou_mesh, ZLB_OccBin_nou, levels=[0.5], colors='purple', linewidths=2, linestyles='dashed')

    # no z
    u_noz_mesh, ln_Gamma_noz_mesh = jnp.meshgrid(u_grid, ln_Gamma_grid)
    states_noz = jnp.concat([u_noz_mesh[..., None], ln_Gamma_noz_mesh[..., None]], axis=-1) # (N_grid, N_grid, 2)
    states_noz = states_noz.reshape(-1, 2) # (N_grid*N_grid, 2)
    states_noz = jnp.concat([states_noz[..., 0, None], jnp.zeros((N_grid*N_grid, 1)), states_noz[..., 1, None]], axis=-1)

    Y_noz, pi_noz = eval_nn(par, train, linear, nn, states_noz, N_grid*N_grid)
    Y_noz, pi_noz = Y_noz.reshape(N_grid, N_grid), pi_noz.reshape(N_grid, N_grid)
    states_noz = states_noz.reshape(N_grid, N_grid, 3)
    i_noz = taylor_rule(par, Y_noz, pi_noz, states_noz[..., 0], states_noz[..., 1], states_noz[..., 2], 0.00, 0.00, ZLB, 0.00)
    ZLB_noz = (i_noz <= ZLB + 1e-5).astype(float)

    Y_OccBin_noz, pi_OccBin_noz, i_OccBin_noz = eval_OccBin_womodel(par, linear, states_noz, return_i=True)
    ZLB_OccBin_noz = (i_OccBin_noz <= ZLB + 1e-5).astype(float)

    ax[1].contourf(u_noz_mesh, ln_Gamma_noz_mesh, ZLB_noz, levels=[-0.5, 0.5, 1.5], colors=['skyblue', 'green'])
    ax[1].contour(u_noz_mesh, ln_Gamma_noz_mesh, ZLB_noz, levels=[0.5], colors='red', linewidths=2)
    ax[1].contour(u_noz_mesh, ln_Gamma_noz_mesh, ZLB_OccBin_noz, levels=[0.5], colors='purple', linewidths=2, linestyles='dashed')

    # no ln_Gamma
    u_noln_Gamma_mesh, z_noln_Gamma_mesh = jnp.meshgrid(u_grid, z_grid)
    states_noln_Gamma = jnp.concat([u_noln_Gamma_mesh[..., None], z_noln_Gamma_mesh[..., None]], axis=-1) # (N_grid, N_grid, 2)
    states_noln_Gamma = states_noln_Gamma.reshape(-1, 2) # (N_grid*N_grid, 2)
    states_noln_Gamma = jnp.concat([states_noln_Gamma, jnp.zeros((N_grid*N_grid, 1))], axis=-1)

    Y_noln_Gamma, pi_noln_Gamma = eval_nn(par, train, linear, nn, states_noln_Gamma, N_grid*N_grid)
    Y_noln_Gamma, pi_noln_Gamma = Y_noln_Gamma.reshape(N_grid, N_grid), pi_noln_Gamma.reshape(N_grid, N_grid)
    states_noln_Gamma = states_noln_Gamma.reshape(N_grid, N_grid, 3)
    i_noln_Gamma = taylor_rule(par, Y_noln_Gamma, pi_noln_Gamma, states_noln_Gamma[..., 0], states_noln_Gamma[..., 1], states_noln_Gamma[..., 2], 0.00, 0.00, ZLB, 0.00)
    ZLB_noln_Gamma = (i_noln_Gamma <= ZLB).astype(float)

    Y_OccBin_noln_Gamma, pi_OccBin_noln_Gamma, i_OccBin_noln_Gamma = eval_OccBin_womodel(par, linear, states_noln_Gamma, return_i=True)
    ZLB_OccBin_noln_Gamma = (i_OccBin_noln_Gamma <= ZLB).astype(float)

    ax[2].contourf(u_noln_Gamma_mesh, z_noln_Gamma_mesh, ZLB_noln_Gamma, levels=[-0.5, 0.5, 1.5], colors=['skyblue', 'green'])
    ax[2].contour(u_noln_Gamma_mesh, z_noln_Gamma_mesh, ZLB_noln_Gamma, levels=[0.5], colors='red', linewidths=2)
    ax[2].contour(u_noln_Gamma_mesh, z_noln_Gamma_mesh, ZLB_OccBin_noln_Gamma, levels=[0.5], colors='purple', linewidths=2, linestyles='dashed')

    # extra
    for i in range(3): ax[i].scatter(0.00, 0.00, color='blue') # , label='DSS & SSS'

    ax[0].text(0.005, -0.005, 'SSS', fontweight='bold', color='blue', fontsize=12)
    ax[1].text(0.0005, -0.005, 'SSS', fontweight='bold', color='blue', fontsize=12)
    ax[2].text(0.0005, -0.008, 'SSS', fontweight='bold', color='blue', fontsize=12)

    def set_std_ticks(ax, std_val, symbol, sigma_label, axis='x', decimals=3):
        ticks = [-2*std_val, -std_val, 0, std_val, 2*std_val]
        labels = [
            rf'$-2\sigma_{{{sigma_label}}}$',
            rf'$-\sigma_{{{sigma_label}}}$',
            '0',
            rf'$\sigma_{{{sigma_label}}}$',
            rf'$2\sigma_{{{sigma_label}}}$'
        ]
        if axis == 'x':
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_xlabel(rf'${symbol}$ $(\sigma_{{{sigma_label}}}={std_val:.4f})$')
        else:
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
            ax.set_ylabel(rf'${symbol}$ $(\sigma_{{{sigma_label}}}={std_val:.4f})$')

    std_u_val        = float(std_u)
    std_z_val        = float(std_z)
    std_ln_Gamma_val = float(std_ln_Gamma)

    set_std_ticks(ax[0], std_z_val,        'z_t',            'z',        axis='x')
    set_std_ticks(ax[0], std_ln_Gamma_val, r'\ln(\Gamma_t)', r'\Gamma',  axis='y')

    set_std_ticks(ax[1], std_u_val,        'u_t',            'u',        axis='x')
    set_std_ticks(ax[1], std_ln_Gamma_val, r'\ln(\Gamma_t)', r'\Gamma',  axis='y')

    set_std_ticks(ax[2], std_u_val,        'u_t',            'u',        axis='x')
    set_std_ticks(ax[2], std_z_val,        'z_t',            'z',        axis='y')

    alphas = [1, 0.7, 0.4]

    for pct, alpha, color in zip([0.45, 0.70, 0.9], alphas, ['black', 'black', 'black']):
        chi2_val = np.sqrt(chi2.ppf(pct, df=2))
        ellipse = Ellipse(xy=(0, 0),
                        width=2*chi2_val*std_z,
                        height=2*chi2_val*std_ln_Gamma,
                        edgecolor=color, facecolor='none',
                        linestyle='dotted', linewidth=2, alpha=alpha)
        ax[0].add_patch(ellipse)

    for pct, alpha, color in zip([0.45, 0.70, 0.9], alphas, ['black', 'black', 'black']):
        chi2_val = np.sqrt(chi2.ppf(pct, df=2))
        ellipse = Ellipse(xy=(0, 0),
                        width=2*chi2_val*std_u,
                        height=2*chi2_val*std_ln_Gamma,
                        edgecolor=color, facecolor='none',
                        linestyle='dotted', linewidth=2, alpha=alpha)
        ax[1].add_patch(ellipse)

    for pct, alpha, color in zip([0.45, 0.70, 0.9], alphas, ['black', 'black', 'black']):
        chi2_val = np.sqrt(chi2.ppf(pct, df=2))
        ellipse = Ellipse(xy=(0, 0),
                        width=2*chi2_val*std_u,
                        height=2*chi2_val*std_z,
                        edgecolor=color, facecolor='none',
                        linestyle='dotted', linewidth=2, alpha=alpha)
        ax[2].add_patch(ellipse)

    ax[0].fill_between((0,0), (0,0), color='green', linestyle='solid', label = r'ZLB binds: $i=\text{ZLB}$')

    ax[1].plot([], [], color='red', linestyle='solid', label = r'DEQN', linewidth=2)

    ax[0].plot([], [], color='black', linestyle='dotted', label = r'45 per.', alpha=0.8)
    ax[0].fill_between((0,0), (0,0), color='skyblue', linestyle='solid', label = r'ZLB does not bind: $i>\text{ZLB}$')
    ax[0].plot([], [], color='black', linestyle='dotted', label = r'70 per.', alpha=0.5)
    ax[2].plot([], [], color='black', linestyle='dotted', label =  r'90 per.', alpha=0.2)

    ax[2].plot([], [], color='purple', linestyle='--', label = r'OccBin', linewidth=2)

    f.tight_layout()

    f.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.15))

    f.tight_layout()

    if save_path is not None:
            f.savefig(save_path, bbox_inches='tight')

def plot_GIRF(model, save_path=None):

    IRF = model.IRF
    GIRF = model.GIRF

    plt.rcParams.update({'font.size': 12})

    f, ax = plt.subplots(5, 3, figsize=(10, 15))

    T = GIRF.T

    ax[0,0].plot(jnp.arange(T), IRF.u)
    ax[0,1].plot(jnp.arange(T), IRF.z)
    ax[0,2].plot(jnp.arange(T), IRF.ln_Gamma)

    ax[0,0].set_title(r'Monetary Policy Shock: $u_t$')
    ax[0,1].set_title(r'Preference Shifter: $z_t$')
    ax[0,2].set_title(r'Productivity: $\ln(\Gamma_t)$')

    # output
    ax[1,0].plot(jnp.arange(T), 100*GIRF.Y_u, color='C0', label='DEQN (GIRF)')
    ax[1,0].plot(jnp.arange(T), 100*GIRF.Y_u_OccBin, color='C1', marker='D', ms=2.5, ls='--', label='OccBin (GIRF)')
    ax[1,0].plot(jnp.arange(T), 100*IRF.Y_u_lin, color='C2', label='Log-Linear (IRF)')

    ax[1,1].plot(jnp.arange(T), 100*GIRF.Y_z, color='C0')
    ax[1,1].plot(jnp.arange(T), 100*GIRF.Y_z_OccBin, color='C1', marker='D', ms=2.5, ls='--')
    ax[1,1].plot(jnp.arange(T), 100*IRF.Y_z_lin, color='C2')

    ax[1,2].plot(jnp.arange(T), 100*GIRF.Y_ln_Gamma, color='C0')
    ax[1,2].plot(jnp.arange(T), 100*GIRF.Y_ln_Gamma_OccBin, color='C1', marker='D', ms=2.5, ls='--')
    ax[1,2].plot(jnp.arange(T), 100*IRF.Y_ln_Gamma_lin, color='C2')

    # inflation
    ax[2,0].plot(jnp.arange(T), 100*GIRF.pi_u, color='C0')
    ax[2,0].plot(jnp.arange(T), 100*GIRF.pi_u_OccBin, color='C1', marker='D', ms=2.5, ls='--')
    ax[2,0].plot(jnp.arange(T), 100*IRF.pi_u_lin, color='C2')

    ax[2,1].plot(jnp.arange(T), 100*GIRF.pi_z, color='C0')
    ax[2,1].plot(jnp.arange(T), 100*GIRF.pi_z_OccBin, color='C1', marker='D', ms=2.5, ls='--')
    ax[2,1].plot(jnp.arange(T), 100*IRF.pi_z_lin, color='C2')

    ax[2,2].plot(jnp.arange(T), 100*GIRF.pi_ln_Gamma, color='C0')
    ax[2,2].plot(jnp.arange(T), 100*GIRF.pi_ln_Gamma_OccBin, color='C1', marker='D', ms=2.5, ls='--')
    ax[2,2].plot(jnp.arange(T), 100*IRF.pi_ln_Gamma_lin, color='C2')

    # nominal interest rate
    ax[3,0].plot(jnp.arange(T), 100*GIRF.i_u, color='C0')
    ax[3,0].plot(jnp.arange(T), 100*GIRF.i_u_OccBin, color='C1', marker='D', ms=2.5, ls='--')
    ax[3,0].plot(jnp.arange(T), 100*IRF.i_u_lin, color='C2')

    ax[3,1].plot(jnp.arange(T), 100*GIRF.i_z, color='C0')
    ax[3,1].plot(jnp.arange(T), 100*GIRF.i_z_OccBin, color='C1', marker='D', ms=2.5, ls='--')
    ax[3,1].plot(jnp.arange(T), 100*IRF.i_z_lin, color='C2')

    ax[3,2].plot(jnp.arange(T), 100*GIRF.i_ln_Gamma, color='C0')
    ax[3,2].plot(jnp.arange(T), 100*GIRF.i_ln_Gamma_OccBin, color='C1', marker='D', ms=2.5, ls='--')
    ax[3,2].plot(jnp.arange(T), 100*IRF.i_ln_Gamma_lin, color='C2')

    # ZLB frequency
    ax[4,0].plot(jnp.arange(T), 100*GIRF.i_u_ZLB, color='C0')
    ax[4,0].plot(jnp.arange(T), 100*GIRF.i_u_ZLB_OccBin, color='C1', marker='D', ms=2.5, ls='--')

    ax[4,1].plot(jnp.arange(T), 100*GIRF.i_z_ZLB, color='C0')
    ax[4,1].plot(jnp.arange(T), 100*GIRF.i_z_ZLB_OccBin, color='C1', marker='D', ms=2.5, ls='--')

    ax[4,2].plot(jnp.arange(T), 100*GIRF.i_ln_Gamma_ZLB, color='C0')
    ax[4,2].plot(jnp.arange(T), 100*GIRF.i_ln_Gamma_ZLB_OccBin, color='C1', marker='D', ms=2.5, ls='--')

    # pretify
    ax[0,0].set_ylabel('Abs. deviation')
    for i_ in [1,4]: ax[i_,0].set_ylabel('pct.')
    for i_ in [2,3]: ax[i_,0].set_ylabel('p.p.')
    for i_ in range(3): ax[1, i_].set_title(r'Output: $Y_t$')
    for i_ in range(3): ax[2, i_].set_title(r'Inflation: $\pi_t$')
    for i_ in range(3): ax[3, i_].set_title(r'Nominal Interest Rate: $i_t$')
    for i_ in range(3): ax[4, i_].set_title(r'ZLB frequency')

    for a in ax.flat:
        a.set_xticks([0, 2, 4, 8, 12])
        a.set_xticklabels(['Impact', 'Q2', 'Q4', 'Q8', 'Q12'], rotation=45, ha='right', rotation_mode='anchor')
        a.grid(True, alpha=0.3)

    f.legend(loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3)
    

    f.tight_layout()

    if save_path is not None:
        f.savefig(save_path, bbox_inches='tight')