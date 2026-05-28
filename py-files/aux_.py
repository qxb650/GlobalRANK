import jax
import jax.numpy as jnp

def choose_gpu():

    print('Avaliable devices:')
    for dev in jax.devices(): print(dev)

    try: # choose gpu if avaliable
        device = jax.devices('gpu')[0]
    except: # otherwise default to cpu
        device = jax.devices('cpu')[0]

    print(f'\nDevice = {device} is choosen')

    return device
    
def draw_shocks(subkey, dtype, N, sigma_sim_eps_u, sigma_sim_eps_z, sigma_sim_eps_Gamma):

    draws = jax.random.normal(subkey, shape=(N, 3), dtype=dtype)

    eps_u = sigma_sim_eps_u * draws[:, 0]
    eps_z = sigma_sim_eps_z * draws[:, 1]
    eps_Gamma = sigma_sim_eps_Gamma * draws[:, 2]

    return jnp.stack([eps_u, eps_z, eps_Gamma], axis=-1)

def next_states_quad(par, dtype, states, gh_x):

    rho_u = par["rho_u"]
    rho_z = par["rho_z"]
    rho_Gamma = par["rho_Gamma"]

    gh_x_u = gh_x[:,0]
    gh_x_z = gh_x[:,1]
    gh_x_Gamma = gh_x[:,2]

    u = states[:, 0]
    z = states[:, 1]
    ln_Gamma = states[:, 2]

    u_p = rho_u*u[:, None] + gh_x_u[None, :]
    z_p = rho_z*z[:, None] + gh_x_z[None, :]
    ln_Gamma_p = rho_Gamma*ln_Gamma[:, None] + gh_x_Gamma[None, :]

    return jnp.stack([u_p, z_p, ln_Gamma_p], axis=-1)

def next_states(par, states, eps):

    rho_u = par["rho_u"]
    rho_z = par["rho_z"]
    rho_Gamma = par["rho_Gamma"]

    u = states[:, 0]
    z = states[:, 1]
    ln_Gamma = states[:, 2]

    eps_u = eps[:, 0]
    eps_z = eps[:, 1]
    eps_Gamma = eps[:, 2]

    u_p = rho_u*u + eps_u
    z_p = rho_z*z + eps_z
    ln_Gamma_p = rho_Gamma*ln_Gamma + eps_Gamma

    return jnp.stack([u_p, z_p, ln_Gamma_p], axis=-1)

def draw_states_directly(subkey, par, dtype, N, sigma_sim_eps_u, sigma_sim_eps_z, sigma_sim_eps_Gamma):

    rho_u = par["rho_u"]
    rho_z = par["rho_z"]
    rho_Gamma = par["rho_Gamma"]

    draws = jax.random.normal(subkey, shape=(N, 3), dtype=dtype)

    sigma_u = sigma_sim_eps_u/(jnp.sqrt(1-rho_u**2))
    sigma_z = sigma_sim_eps_z/(jnp.sqrt(1-rho_z**2))
    sigma_Gamma = sigma_sim_eps_Gamma/(jnp.sqrt(1-rho_Gamma**2))

    u = sigma_u*draws[:, 0]
    z = sigma_z*draws[:, 1]
    ln_Gamma = sigma_Gamma*draws[:, 2]

    return jnp.stack([u, z, ln_Gamma], axis=-1)

def draw_states_mixed(subkey, par, linear, dtype, N, sigmas, zlb_frac=0.0):
    """Draw N states, with zlb_frac forced near/in ZLB region."""
    k1, k2 = jax.random.split(subkey)
    
    N_zlb = int(N * zlb_frac)
    N_erg = N - N_zlb
    
    # ergodic draws
    states_erg = draw_states_directly(k1, par, dtype, N_erg, 
                                       sigmas["sigma_eps_u"], 
                                       sigmas["sigma_eps_z"], 
                                       sigmas["sigma_eps_Gamma"])
    
    # ZLB draws: large negative u (MP shock drives ZLB)
    states_zlb = draw_states_directly(k2, par, dtype, N_zlb,
                                       sigmas["sigma_eps_u"],
                                       sigmas["sigma_eps_z"], 
                                       sigmas["sigma_eps_Gamma"])
                                       
    # push u negative to force ZLB binding
    u_zlb = states_zlb[:, 0] + sigmas["sigma_eps_u"] / jnp.sqrt(1 - par["rho_u"]**2)
    z_zlb = states_zlb[:, 1] - sigmas["sigma_eps_z"] / jnp.sqrt(1 - par["rho_z"]**2)
    Gamma_zlb = states_zlb[:, 2] - sigmas["sigma_eps_Gamma"] / jnp.sqrt(1 - par["rho_Gamma"]**2)
    
    states_zlb = states_zlb.at[:, 0].set(u_zlb)
    states_zlb = states_zlb.at[:, 1].set(z_zlb)
    states_zlb = states_zlb.at[:, 2].set(Gamma_zlb)
    

    return jnp.concat([states_erg, states_zlb], axis=0)