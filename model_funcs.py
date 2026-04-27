
import jax.numpy as jnp

def marg_util(par, C):

    sigma = par["sigma"]

    return C**(-sigma)

def inv_marg_util(par, marg_u):

    sigma = par["sigma"]

    return marg_u**(-(1/sigma))

def market_clearing_C(par, Y, pi):

    theta = par["theta"]

    return Y*(1-0.5*theta*pi**2)

def compute_w(par, Y, pi, ln_Gamma):

    alpha = par["alpha"]
    sigma = par["sigma"]
    varphi = par["varphi"]
    theta = par["theta"]

    Gamma = jnp.exp(ln_Gamma)

    frac = varphi/(1-alpha)

    w = ((Y**(frac+sigma))*(1-0.5*theta*pi**2)**sigma)/(Gamma**frac)

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
    mu = par["mu"]

    common_term = varphi+alpha+sigma-alpha*sigma

    Gamma = jnp.exp(ln_Gamma)

    Y_star = (((1-alpha)/mu)**((1-alpha)/common_term))*Gamma**((1+varphi)/common_term)

    return Y_star

# def compute_i_star(par, z, ln_Gamma, eps_z, eps_Gamma, weights):

#     beta = par["beta"]
#     sigma = par["sigma"]
#     varphi = par["varphi"]
#     rho_Gamma = par["rho_Gamma"]
#     rho_z = par["rho_z"]

#     frac_ = sigma*((1+varphi)/(sigma+varphi))
#     num = jnp.exp((1-rho_z)*z + frac_*(rho_Gamma-1)*ln_Gamma) # (Nparallel,)

#     denom = jnp.exp(eps_z[None, :]+frac_*eps_Gamma[None, :])
#     expected_denom = jnp.sum(weights[None, :]*denom, axis=-1, keepdims=False)

#     return num/(beta*expected_denom) - 1

def taylor_rule(par, Y, pi, u, z, ln_Gamma, eps_z, eps_Gamma, ZLB, weights, return_shadow=False):
    
    beta = par["beta"]
    phi_pi = par["phi_pi"]
    phi_y = par["phi_y"]
    lin_taylor_rule = par["do_lin_taylor_rule"]
    do_DSS_as_Ystar = par["do_DSS_as_Ystar"]

    # compute natural output and nominal interest rates
    if do_DSS_as_Ystar:
        Y_star = par["Y_DSS"]
    else:
        Y_star = compute_Y_star(par, ln_Gamma)

    # compute percentage gaps if linear
    if lin_taylor_rule:
        i_star = 1/beta - 1
        output_gab = (Y-Y_star)/Y_star
        i_shadow = i_star + phi_pi*pi + phi_y*output_gab + u

    # compute share gaps if nonlinear and do logs
    else:
        i_star = 1/beta - 1
        output_gab = jnp.maximum(Y, 1e-6)/Y_star
        pi_gab = jnp.maximum(1 + pi, 1e-6)
        i_shadow = (1+i_star)*(pi_gab**phi_pi)*(output_gab**phi_y)*(1+u) - 1

    i = jnp.maximum(i_shadow, ZLB)

    if return_shadow:
        return i_shadow

    else:
        return i

def euler_error(par, Y, Y_p, pi, pi_p, i, u, z, eps_z, weights):

    beta = par["beta"]
    rho_z = par["rho_z"]

    # Y, pi is (Nparallel,1)
    # Y_p, pi_p is (Nparallel,weigths)

    # period t
    C = market_clearing_C(par, Y, pi)

    # period t+1
    C_p = market_clearing_C(par, Y_p, pi_p)
    MU_p = marg_util(par, C_p) * jnp.exp(eps_z[None, :]) * (1/(1+pi_p)) 
    EMU_p = beta * (1 + i) * jnp.exp((rho_z-1)*z) * jnp.sum(weights[None, :] * MU_p, axis=-1, keepdims=False)

    ee = inv_marg_util(par, EMU_p)/C-1

    return ee

def NKPC_error(par, Y, Y_p, pi, pi_p, i, u, ln_Gamma, weights):

    beta = par["beta"]
    kappa = par["kappa"]
    mu = par["mu"]

    # Y, pi is (Nparallel,)
    # Y_p, pi_p is (Nparallel,weigths)

    # period t
    mc = compute_mc(par, Y, pi, ln_Gamma) # (Nparallel,)
    today = pi * (1 + pi) - kappa * (mc - 1/mu) # (Nparallel,)

    # period t+1
    tomorrow = Y_p * pi_p * (1 + pi_p) # (Nparallel,weights)
    Etomorrow = beta * (jnp.sum(weights[None, :] * tomorrow, axis=-1, keepdims=False)/Y) # (Nparallel,)

    return Etomorrow - today # (Nparallel,)