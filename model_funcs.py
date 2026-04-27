import numpy as np
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import random
import time
import os

def marg_util(par, C):

    sigma = par["sigma"]

    return C**(-sigma)

def inv_marg_util(par, marg_u):

    sigma = par["sigma"]

    return marg_u**(-(1/sigma))

def market_clearing_C(par, Y, pi):

    theta = par["theta"]
    pi_target = par["pi_target"]

    pi_gab = pi - pi_target

    return Y*(1-0.5*theta*pi_gab**2)

def compute_Y_star(par, ln_Gamma):

    sigma = par["sigma"]
    varphi = par["varphi"]
    mu = par["mu"]

    num_frac = (1+varphi)/(sigma+varphi)
    denom_frac = 1/(sigma+varphi)

    Gamma = jnp.exp(ln_Gamma)

    return (Gamma**(num_frac))/(mu**(denom_frac))

def compute_i_star(par, z, ln_Gamma, eps_z, eps_Gamma, weights):

    beta = par["beta"]
    sigma = par["sigma"]
    varphi = par["varphi"]
    rho_Gamma = par["rho_Gamma"]
    rho_z = par["rho_z"]

    frac_ = sigma*((1+varphi)/(sigma+varphi))
    num = jnp.exp((1-rho_z)*z + frac_*(rho_Gamma-1)*ln_Gamma) # (Nparallel,)

    denom = jnp.exp(eps_z[None, :]+frac_*eps_Gamma[None, :])
    expected_denom = jnp.sum(weights[None, :]*denom, axis=-1, keepdims=False)

    return num/(beta*expected_denom) - 1

def taylor_rule(par, Y, pi, u, z, ln_Gamma, eps_z, eps_Gamma, ZLB, weights, return_shadow=False):
    
    beta = par["beta"]
    phi_pi = par["phi_pi"]
    phi_y = par["phi_y"]
    pi_target = par["pi_target"]
    lin_taylor_rule = par["do_lin_taylor_rule"]
    do_DSS_as_Ystar = par["do_DSS_as_Ystar"]

    # compute natural output and nominal interest rates
    if do_DSS_as_Ystar:
        Y_star = par["Y_DSS"]
    else:
        Y_star = compute_Y_star(par, ln_Gamma)

    # compute percentage gaps if linear
    if lin_taylor_rule:
        i_star = 1/beta - 1 + pi_target
        output_gab = (Y-Y_star)/Y_star
        pi_gab = pi-pi_target
        i_shadow = i_star + phi_pi*pi_gab + phi_y*output_gab + u

    # compute share gaps if nonlinear and do logs
    else:
        i_star = jnp.log((1+pi_target)/beta)
        output_gab = jnp.log(jnp.maximum(Y, 1e-6)) - jnp.log(Y_star)
        pi_gab = jnp.log(jnp.maximum(1 + pi, 1e-6)) - jnp.log(1+pi_target)
        i_shadow = jnp.exp(i_star + phi_pi*pi_gab + phi_y*output_gab + u) - 1

    i = jnp.maximum(i_shadow, ZLB)

    if return_shadow:
        return i_shadow

    else:
        return i

def euler_error(par, Y, Y_p, pi, pi_p, i, u, z, eps_z, weights):

    beta = par["beta"]
    sigma = par["sigma"]
    rho_z = par["rho_z"]
    theta = par["theta"]
    pi_target = par["pi_target"]

    # Y, pi is (Nparallel,1)
    # Y_p, pi_p is (Nparallel,weigths)

    # period t
    pi_gab = pi - pi_target
    C = market_clearing_C(par, Y, pi)

    # period t+1
    pi_gab_p = pi_p - pi_target
    C_p = market_clearing_C(par, Y_p, pi_p)
    MU_p = marg_util(par, C_p) * jnp.exp(eps_z[None, :]) * (1/(1+pi_p)) 

    EMU_p = beta * (1 + i) * jnp.exp((rho_z-1)*z) * jnp.sum(weights[None, :] * MU_p, axis=-1, keepdims=False)

    return inv_marg_util(par, EMU_p)/C-1

def NKPC_error(par, Y, Y_p, pi, pi_p, i, u, ln_Gamma, weights):

    beta = par["beta"]
    kappa = par["kappa"]
    mu = par["mu"]
    sigma = par["sigma"]
    varphi = par["varphi"]
    theta = par["theta"]
    pi_target = par["pi_target"]

    # Y, pi is (Nparallel,)
    # Y_p, pi_p is (Nparallel,weigths)

    # period t
    pi_gab = pi - pi_target
    Gamma = jnp.exp(ln_Gamma) # (Nparallel,)
    C = market_clearing_C(par, Y, pi) # (Nparallel,)
    frac = (C**(sigma+varphi))/(Gamma**(1+varphi)) # (Nparallel,)
    today = pi_gab * (1 + pi) - kappa * (frac - 1/mu) # (Nparallel,)

    # period t+1
    pi_gab_p = pi_p - pi_target
    tomorrow = Y_p * pi_gab_p * (1 + pi_p)**2 # (Nparallel,weights)
    Etomorrow = jnp.sum(weights[None, :] * tomorrow, axis=-1, keepdims=False)/(Y*(1+i)) # (Nparallel,)

    return Etomorrow - today # (Nparallel,)