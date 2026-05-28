import jax.numpy as jnp

def marg_util(par, C):

    sigma = par["sigma"]

    return C**(-sigma)

def inv_marg_util(par, marg_u):

    sigma = par["sigma"]

    return marg_u**(-(1/sigma))

def market_clearing_C(par, Y, pi):

    return Y

def compute_w(par, Y, pi, ln_Gamma):

    alpha = par["alpha"]
    sigma = par["sigma"]
    varphi = par["varphi"]

    Gamma = jnp.exp(ln_Gamma)

    frac = varphi/(1-alpha)

    w = (Y**(frac+sigma))/(Gamma**frac)

    return w

def compute_mc(par, Y, pi, ln_Gamma):
    
    alpha = par["alpha"]

    Gamma = jnp.exp(ln_Gamma)
    w = compute_w(par, Y, pi, ln_Gamma)

    num = w * (Y**((alpha)/(1-alpha)))
    denom = (1-alpha)*Gamma**(1/(1-alpha))

    mc = num/denom

    return mc

def compute_Y_star(par, ln_Gamma):

    alpha = par["alpha"]
    sigma = par["sigma"]
    varphi = par["varphi"]
    mu = par["epsilon"]/(par["epsilon"]-1)

    common_term = varphi+alpha+sigma-alpha*sigma

    Gamma = jnp.exp(ln_Gamma)

    Y_star = (((1-alpha)/mu)**((1-alpha)/common_term))*Gamma**((1+varphi)/common_term)

    return Y_star

def taylor_rule(par, Y, pi, u, z, ln_Gamma, eps_z, eps_Gamma, ZLB, weights, return_shadow=False, return_linear=False):
    
    beta = par["beta"]
    phi_pi = par["phi_pi"]
    phi_y = par["phi_y"]

    if return_linear: # in this case variables are in (log-)-DSS-deviations and not levels
        return phi_pi*pi + phi_y*Y-phi_y*(1+par["varphi"])/(par["varphi"]+par["sigma"]+par["alpha"]-par["alpha"]*par["sigma"])*ln_Gamma+u

    # compute natural output and natural (gross) nominal interest rates
    i_star = (1/beta)
    Y_star = compute_Y_star(par, ln_Gamma)

    # compute gross outputgab, avoid division with zero
    output_gab = jnp.maximum(Y, 1e-8)/Y_star

    # compute gross inflation-gab, avoid division with zero
    pi_gab = jnp.maximum(1 + pi, 1e-8)

    # compute shadow rate
    i_shadow = (i_star)*(pi_gab**phi_pi)*(output_gab**phi_y)*jnp.exp(u) - 1

    i = jnp.maximum(i_shadow, ZLB)

    if return_shadow:
        return i_shadow

    else:
        return i

def euler_error(par, Y, Y_p, pi, pi_p, i, u, z, eps_z, weights):

    beta = par["beta"]
    rho_z = par["rho_z"]

    # Y, pi is (Nparallel,)
    # Y_p, pi_p is (Nparallel,weigths)

    # period t
    C = market_clearing_C(par, Y, pi)  # (Nparallel,)

    # period t+1
    C_p = market_clearing_C(par, Y_p, pi_p) # (Nparallel,weights)
    MU_p = marg_util(par, C_p) * jnp.exp(eps_z[None, :]) * (1/(1+pi_p)) # (Nparallel,weights)
    EMU_p = beta * (1 + i) * jnp.exp((rho_z-1)*z) * jnp.sum(weights[None, :] * MU_p, axis=-1, keepdims=False) # (Nparallel,)

    ee = inv_marg_util(par, EMU_p)/C-1 # # (Nparallel,)

    return ee

def NKPC_error(par, Y, Y_p, pi, pi_p, i, u, ln_Gamma, weights):

    beta = par["beta"]
    kappa = par["kappa"]
    mu = par["epsilon"]/(par["epsilon"]-1)

    # Y, pi is (Nparallel,)
    # Y_p, pi_p is (Nparallel,weigths)

    # period t
    mc = compute_mc(par, Y, pi, ln_Gamma) # (Nparallel,)
    today = pi * (1 + pi) - kappa * (mc - 1/mu) # (Nparallel,)

    # period t+1
    tomorrow = Y_p * pi_p * (1 + pi_p) # (Nparallel,weights)
    Etomorrow = beta * (jnp.sum(weights[None, :] * tomorrow, axis=-1, keepdims=False)/Y) # (Nparallel,)

    nkpce = Etomorrow - today # (Nparallel,)

    return nkpce