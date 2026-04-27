import jax
import jax.numpy as jnp

def choose_gpu():

    devs = jax.devices()

    if 'gpu' in devs:
        print('choosing gpu')
        return devs['gpu']

    else:
        print('choosing cpu')
        return devs[0]
    
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
