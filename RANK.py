import numpy as np
from types import SimpleNamespace
import jax.numpy as jnp
import optax
from flax import nnx
import os
import matplotlib.pyplot as plt
import pickle

from neural_nets import Policy, eval_nn
import linear
from linear import OccBin
import aux_ as aux
import jax

class RANK_model:

    def __init__(self, device, dtype=jnp.float32):
        
        self.device = device
        self.dtype = dtype

        self.setup()

        self.setup_train()
        self.setup_linear()
        self.setup_nn()

    def setup(self):

        par = {}

        # parameters of baseline model that could be estimated
        par["alpha"] = 0.33
        par["sigma"] = 1.0
        par["beta"] = 0.985 #0.97
        par["epsilon"] = 9.0
        par["varphi"] = 5.0
        par["theta"] = 0.1

        par["kappa"] = par["epsilon"]/90 #par["theta"] FIX THETA
        par["mu"] = (par["epsilon"])/(par["epsilon"]-1)

        # CB parameters to be calibrated
        par["phi_y"] = 0.5
        par["phi_pi"] = 1.5
        par["ZLB"] = 0.00
        par["do_lin_taylor_rule"] = False
        par["do_DSS_as_Ystar"] = False

        # DSS
        par["Y_DSS"] = ((1-par["alpha"])/par["mu"])**((1-par["alpha"])/(par["varphi"]+par["alpha"]+par["sigma"]-par["alpha"]*par["sigma"]))
        par["pi_DSS"] = 0.00
        par["i_DSS"] = 1/par["beta"] - 1
        par["u_DSS"] = 0.0
        par["z_DSS"] = 0.0
        par["ln_Gamma_DSS"] = 0.0

        # shocks
        par["rho_u"] = 0.8753
        par["rho_z"] = 0.9
        par["rho_Gamma"] = 0.7559

        # model
        par["Nshocks"] = 3 # eps_u, eps_z, eps_Gamma
        par["Nstates"] = 3 # u, z, ln_Gamma
        par["Npolicies"] = 2 # Y, pi

        self.par = par

    def setup_train(self):

        train = {}

        train["T"] = 50
        train["neurons"] = (100, 100) # (64, 64) # tuple so it is hashable
        train["Nparallel"] = 200
        train["Nparallel_test"] = 20000
        train["lr"] = 1e-4
        train["gh_n_per_shock"] = 4
        train["do_ZLB_dummy"] = True
        train["do_shadow_taylor_rule"] = False

        train["T_OccBin"] = 150
        train["n_grid_OccBin"] = 150 # + 1 in linear
        train["states_std_interp"] = 0.1

        # quadrature
        # train["gh_x"], train["gh_w"] = construct_gh_nodes(
        #                                         train["gh_n_per_shock"],
        #                                         par["sigma_eps_u"],
        #                                         par["sigma_eps_z"],
        #                                         par["sigma_eps_Gamma"]
        #                                         )

        train["gh_n"] = train["gh_n_per_shock"]**3 # len(train["gh_w"])

        self.train = train

    def setup_linear(self):

        train = self.train

        T_OccBin = train["T_OccBin"]
        n_grid_OccBin = train["n_grid_OccBin"]
        states_std_interp = train["states_std_interp"]

        linear.setup_linear(self, T_OccBin, n_grid=n_grid_OccBin, shock_interp=states_std_interp)
    
    def setup_nn(self):

        par = self.par
        train = self.train
        linear = self.linear
        dtype = self.dtype
        device = self.device

        # 1. compute in- and output dimensions and retrieve list of neurons
        din = par["Nstates"] + 4# 8 #+ 43 #+ 1 #+# 6 #6 + int(linear["max_expected_ZLB"]) + 1 + 2
        dout = par["Npolicies"]
        neurons = train["neurons"]
        lr = train["lr"]

        # 2. call policy class
        nn = Policy(din, dout, neurons, rngs=nnx.Rngs(params=0), dtype=dtype, device=device) # last kwarg set seed=0 for bias and weight initialization

        ### GEMINI SHIT ###
        # 3. setup optimizer
        def make_opt(learning_rate, clipping_value):
            return optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate=learning_rate)
            )

        # 2. Giv denne FUNKTION til inject_hyperparams
        # Bemærk: Vi kalder IKKE make_opt her, vi giver bare navnet videre
        optimizer_with_hparams = optax.inject_hyperparams(make_opt)(learning_rate=lr, clipping_value=1.0)

        # 3. Nu kan NNX bruge den
        opt = nnx.ModelAndOptimizer(nn, optimizer_with_hparams)#, optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr)))
        ### GEMINI SHIT ###

        # 4. save in model
        self.nn = nn
        self.opt = opt

    def compute_IRF(self, shocks, rtol=50, extra_nn=None):

        par = self.par
        train = self.train
        linear = self.linear
        nn = self.nn

        Y_DSS = par["Y_DSS"]

        rho_u = par["rho_u"]
        rho_z = par["rho_z"]
        rho_Gamma = par["rho_Gamma"]

        T_u = int(10*jnp.ceil(-jnp.log(rtol)/(10*jnp.log(rho_u))).item())
        T_z = int(10*jnp.ceil(-jnp.log(rtol)/(10*jnp.log(rho_z))).item())
        T_Gamma = int(10*jnp.ceil(-jnp.log(rtol)/(10*jnp.log(rho_Gamma))).item())
        
        # compute SSS
        Y_SSS, pi_SSS = eval_nn(par, train, linear, nn, jnp.zeros((1, 3)), 1)
        if extra_nn is not None: Y_SSS_x, pi_SSS_x = eval_nn(par, train, linear, extra_nn, jnp.zeros((1, 3)), 1)

        # MP shock
        zeros = jnp.zeros((T_u, 1))
        u_shock = shocks[0] * rho_u**jnp.arange(T_u) #+ #par["mu_u"]
        u_shock_states = jnp.concat([u_shock[:, None], zeros, zeros], axis=-1) # (T, 3)
        out_Y_u, out_pi_u = eval_nn(par, train, linear, nn, u_shock_states, T_u)
        IRF_Y_u = (out_Y_u - Y_SSS)/Y_SSS
        IRF_pi_u = out_pi_u - pi_SSS
        if extra_nn:
            out_Y_x_u, out_pi_x_u = eval_nn(par, train, linear, extra_nn, u_shock_states, T_u)
            IRF_Y_x_u = (out_Y_x_u - Y_SSS_x)/Y_SSS_x
            IRF_pi_x_u = out_pi_x_u - pi_SSS_x

        out_OccBin, _ = OccBin(par, linear, u_shock_states)
        IRF_Y_u_OccBin = out_OccBin[:,0]/Y_DSS
        IRF_pi_u_OccBin = out_OccBin[:,1]

        # preference shock
        zeros = jnp.zeros((T_z, 1))
        z_shock = shocks[1] * rho_z**jnp.arange(T_z) #+ #par["mu_z"]
        z_shock_states = jnp.concat([zeros, z_shock[:, None], zeros], axis=-1)
        out_Y_z, out_pi_z = eval_nn(par, train, linear, nn, z_shock_states, T_z)
        IRF_Y_z = (out_Y_z - Y_SSS)/Y_SSS
        IRF_pi_z = out_pi_z - pi_SSS
        if extra_nn:
            out_Y_x_z, out_pi_x_z = eval_nn(par, train, linear, extra_nn, z_shock_states, T_z)
            IRF_Y_x_z = (out_Y_x_z - Y_SSS_x)/Y_SSS_x
            IRF_pi_x_z = out_pi_x_z - pi_SSS_x

        out_OccBin, _ = OccBin(par, linear, z_shock_states)
        IRF_Y_z_OccBin = out_OccBin[:,0]/Y_DSS
        IRF_pi_z_OccBin = out_OccBin[:,1]

        # productivity shock
        zeros = jnp.zeros((T_Gamma, 1))
        ln_Gamma_shock = shocks[2] * rho_Gamma**jnp.arange(T_Gamma) #+ #par["mu_Gamma"]
        ln_Gamma_shock_states = jnp.concat([zeros, zeros, ln_Gamma_shock[:, None]], axis=-1)
        out_Y_Gamma, out_pi_Gamma = eval_nn(par, train, linear, nn, ln_Gamma_shock_states, T_Gamma)
        IRF_Y_ln_Gamma = (out_Y_Gamma - Y_SSS)/Y_SSS
        IRF_pi_ln_Gamma = out_pi_Gamma - pi_SSS

        if extra_nn:
            out_Y_x_Gamma, out_pi_x_Gamma = eval_nn(par, train, linear, extra_nn, ln_Gamma_shock_states, T_Gamma)
            IRF_Y_x_ln_Gamma = (out_Y_x_Gamma - Y_SSS_x)/Y_SSS_x
            IRF_pi_x_ln_Gamma = out_pi_x_Gamma - pi_SSS_x

        out_OccBin, _ = OccBin(par, linear, ln_Gamma_shock_states)
        IRF_Y_ln_Gamma_OccBin = out_OccBin[:,0]/Y_DSS
        IRF_pi_ln_Gamma_OccBin = out_OccBin[:,1]

        IRF = SimpleNamespace()

        IRF.T_u = T_u
        IRF.T_z = T_z
        IRF.T_Gamma = T_Gamma

        IRF.u = u_shock
        IRF.z = z_shock
        IRF.ln_Gamma = ln_Gamma_shock

        IRF.Y_u = IRF_Y_u
        IRF.Y_z = IRF_Y_z
        IRF.Y_ln_Gamma = IRF_Y_ln_Gamma

        IRF.pi_u = IRF_pi_u
        IRF.pi_z = IRF_pi_z
        IRF.pi_ln_Gamma = IRF_pi_ln_Gamma

        if extra_nn:
            IRF.Y_extra_u = IRF_Y_x_u
            IRF.Y_extra_z = IRF_Y_x_z
            IRF.Y_extra_ln_Gamma = IRF_Y_x_ln_Gamma

            IRF.pi_extra_u = IRF_pi_x_u
            IRF.pi_extra_z = IRF_pi_x_z
            IRF.pi_extra_ln_Gamma = IRF_pi_x_ln_Gamma

        IRF.Y_u_OccBin = IRF_Y_u_OccBin
        IRF.Y_z_OccBin = IRF_Y_z_OccBin
        IRF.Y_ln_Gamma_OccBin = IRF_Y_ln_Gamma_OccBin

        IRF.pi_u_OccBin = IRF_pi_u_OccBin
        IRF.pi_z_OccBin = IRF_pi_z_OccBin
        IRF.pi_ln_Gamma_OccBin = IRF_pi_ln_Gamma_OccBin

        self.IRF = IRF

    def compute_GIRF(self, shocks, N, key_=42, rtol=50, extra_nn=None):

        par = self.par
        train = self.train
        linear = self.linear
        dtype = self.dtype
        nn = self.nn

        Y_DSS = par["Y_DSS"]

        Y_interp_OccBin, pi_interp_OccBin = linear["Y_interp_OccBin"], linear["pi_interp_OccBin"]

        rho_u = par["rho_u"]
        rho_z = par["rho_z"]
        rho_Gamma = par["rho_Gamma"]

        T_u = int(10*jnp.ceil(-jnp.log(rtol)/(10*jnp.log(rho_u))).item())
        T_z = int(10*jnp.ceil(-jnp.log(rtol)/(10*jnp.log(rho_z))).item())
        T_Gamma = int(10*jnp.ceil(-jnp.log(rtol)/(10*jnp.log(rho_Gamma))).item())
        T_max = int(jnp.array([T_u, T_z, T_Gamma]).max())

        # compute control
        key = jax.random.key(key_)
        key, subkey = jax.random.split(key)

        states = jnp.zeros((T_max, N, 3)) + jnp.nan
        states_i = aux.draw_states_directly(subkey, par, dtype, N, *shocks)
        for t in range(T_max):
            states = states.at[t].set(states_i)
            key, subkey = jax.random.split(key)
            eps = aux.draw_shocks(subkey, dtype, N, 0, 0, 0)
            states_i = aux.next_states(par, states_i, eps)

        Y_control, pi_control = eval_nn(par, train, linear, nn, states, T_max)

        out_OccBin, _ = OccBin(par, linear, states.reshape(-1,3))
        out_OccBin = out_OccBin.reshape(T_max, N, 2)
        Y_OccBin_control, pi_OccBin_control = jnp.clip(out_OccBin[...,0], -0.07, 1.00), jnp.clip(out_OccBin[...,1], -0.07, 1.00) #Y_interp_OccBin(states), pi_interp_OccBin(states)

        if extra_nn is not None:
            Y_x_control, pi_x_control = eval_nn(par, train, linear, extra_nn, states, T_max)

        # MP shock
        zeros = jnp.zeros((T_max, 1))
        u_shock = shocks[0] * rho_u**jnp.arange(T_max)
        u_shock_states = jnp.concat([u_shock[:, None], zeros, zeros], axis=-1) # (T, 3)
        u_shock_states = states + u_shock_states[:, None, :] # (T, N, 3)
        out_Y_u, out_pi_u = eval_nn(par, train, linear, nn, u_shock_states, T_max)

        GIRF_Y_u = jnp.mean((out_Y_u[:T_u] - Y_control[:T_u])/Y_control[:T_u], axis=1) # (N, T, 3)
        GIRF_pi_u = jnp.mean(out_pi_u[:T_u] - pi_control[:T_u], axis=1) # (N, T, 3)

        out_OccBin, _ = OccBin(par, linear, u_shock_states.reshape(-1,3))
        out_OccBin = out_OccBin.reshape(T_max, N, 2)

        GIRF_Y_u_OccBin = jnp.mean((jnp.clip(out_OccBin[:T_u, :, 0], -0.07, 1.00)-Y_OccBin_control[:T_u])/(Y_DSS+Y_OccBin_control[:T_u]), axis=1)
        GIRF_pi_u_OccBin = jnp.mean((jnp.clip(out_OccBin[:T_u, :, 1], -0.07, 1.00)-pi_OccBin_control[:T_u]), axis=1)
        
        if extra_nn:
            out_Y_x_u, out_pi_x_u = eval_nn(par, train, linear, nn, u_shock_states, T_max)
            GIRF_Y_x_u =  jnp.mean((out_Y_x_u[:T_u] - Y_x_control[:T_u])/Y_x_control[:T_u], axis=1)
            GIRF_pi_x_u =  jnp.mean(out_pi_x_u[:T_u] - pi_x_control[:T_u], axis=1)

        # preference shock
        zeros = jnp.zeros((T_max, 1))
        z_shock = shocks[1] * rho_z**jnp.arange(T_max) #+ #par["mu_z"]
        z_shock_states = jnp.concat([zeros, z_shock[:, None], zeros], axis=-1)
        z_shock_states = states + z_shock_states[:, None, :]
        out_Y_z, out_pi_z = eval_nn(par, train, linear, nn, z_shock_states, T_max)
        GIRF_Y_z = jnp.mean((out_Y_z[:T_z] - Y_control[:T_z])/Y_control[:T_z], axis=1)
        GIRF_pi_z = jnp.mean(out_pi_z[:T_z] - pi_control[:T_z], axis=1)

        out_OccBin, _ = OccBin(par, linear, z_shock_states.reshape(-1,3))
        out_OccBin = out_OccBin.reshape(T_max, N, 2)

        GIRF_Y_z_OccBin = jnp.mean((jnp.clip(out_OccBin[:T_z, :, 0], -0.07, 1.00)-Y_OccBin_control[:T_z])/(Y_DSS+Y_OccBin_control[:T_z]), axis=1)
        GIRF_pi_z_OccBin = jnp.mean((jnp.clip(out_OccBin[:T_z, :, 1], -0.07, 1.00)-pi_OccBin_control[:T_z]), axis=1)

        if extra_nn:
            out_Y_x_z, out_pi_x_z = eval_nn(par, train, linear, extra_nn, z_shock_states, T_max)
            GIRF_Y_x_z = jnp.mean((out_Y_x_z[:T_z] - Y_x_control[:T_z])/Y_x_control[:T_z], axis=1)
            GIRF_pi_x_z = jnp.mean(out_pi_x_z[:T_z] - pi_x_control[:T_z], axis=1)

        # productivity shock
        zeros = jnp.zeros((T_max, 1))
        ln_Gamma_shock = shocks[2] * rho_Gamma**jnp.arange(T_max) #+ #par["mu_Gamma"]
        ln_Gamma_shock_states = jnp.concat([zeros, zeros, ln_Gamma_shock[:, None]], axis=-1)
        ln_Gamma_shock_states = states + ln_Gamma_shock_states[:, None, :]
        out_Y_Gamma, out_pi_Gamma = eval_nn(par, train, linear, nn, ln_Gamma_shock_states, T_max)
        GIRF_Y_ln_Gamma = jnp.mean((out_Y_Gamma[:T_Gamma] - Y_control[:T_Gamma])/Y_control[:T_Gamma], axis=1)
        GIRF_pi_ln_Gamma = jnp.mean(out_pi_Gamma[:T_Gamma] - pi_control[:T_Gamma], axis=1)

        out_OccBin, _ = OccBin(par, linear, ln_Gamma_shock_states.reshape(-1,3))
        out_OccBin = out_OccBin.reshape(T_max, N, 2)

        GIRF_Y_Gamma_OccBin = jnp.mean((jnp.clip(out_OccBin[:T_Gamma, :, 0], -0.07, 1.00)-Y_OccBin_control[:T_Gamma])/(Y_DSS+Y_OccBin_control[:T_Gamma]), axis=1)
        GIRF_pi_Gamma_OccBin = jnp.mean((np.clip(out_OccBin[:T_Gamma, :, 0], -0.07, 1.00)-pi_OccBin_control[:T_Gamma]), axis=1)

        if extra_nn:
            out_Y_x_Gamma, out_pi_x_Gamma = eval_nn(par, train, linear, nn, ln_Gamma_shock_states, T_max)
            GIRF_Y_x_ln_Gamma = jnp.mean((out_Y_x_Gamma[:T_Gamma] - Y_x_control[:T_Gamma])/Y_x_control[:T_Gamma], axis=1)
            GIRF_pi_x_ln_Gamma = jnp.mean(out_pi_x_Gamma[:T_Gamma] - pi_x_control[:T_Gamma], axis=1)

        GIRF = SimpleNamespace()

        GIRF.T_u = T_u
        GIRF.T_z = T_z
        GIRF.T_Gamma = T_Gamma

        GIRF.u = u_shock[:T_u]
        GIRF.z = z_shock[:T_z]
        GIRF.ln_Gamma = ln_Gamma_shock[:T_Gamma]

        GIRF.Y_u = GIRF_Y_u
        GIRF.Y_z = GIRF_Y_z
        GIRF.Y_ln_Gamma = GIRF_Y_ln_Gamma

        GIRF.pi_u = GIRF_pi_u
        GIRF.pi_z = GIRF_pi_z
        GIRF.pi_ln_Gamma = GIRF_pi_ln_Gamma

        GIRF.Y_u_OccBin = GIRF_Y_u_OccBin
        GIRF.Y_z_OccBin = GIRF_Y_z_OccBin
        GIRF.Y_Gamma_OccBin = GIRF_Y_Gamma_OccBin

        GIRF.pi_u_OccBin = GIRF_pi_u_OccBin
        GIRF.pi_z_OccBin = GIRF_pi_z_OccBin
        GIRF.pi_Gamma_OccBin = GIRF_pi_Gamma_OccBin

        if extra_nn:
            GIRF.Y_extra_u = GIRF_Y_x_u
            GIRF.Y_extra_z = GIRF_Y_x_z
            GIRF.Y_extra_ln_Gamma = GIRF_Y_x_ln_Gamma

            GIRF.pi_extra_u = GIRF_pi_x_u
            GIRF.pi_extra_z = GIRF_pi_x_z
            GIRF.pi_extra_ln_Gamma = GIRF_pi_x_ln_Gamma

        self.GIRF = GIRF

    def plot_IRF(self, save_path=None, plot_extra=False):

        par = self.par
        train = self.train
        linear = self.linear
        IRF = self.IRF

        T_u = IRF.T_u
        T_z = IRF.T_z
        T_Gamma = IRF.T_Gamma
        P = linear["P"]
        Y_DSS = par["Y_DSS"]

        Y_interp_OccBin, pi_interp_OccBin = linear["Y_interp_OccBin"], linear["pi_interp_OccBin"]
        
        f, ax = plt.subplots(3, 3, figsize=(12, 12))

        # shocks (1st row)
        ax[0,0].plot(jnp.arange(T_u), IRF.u)
        ax[0,1].plot(jnp.arange(T_z), IRF.z)
        ax[0,2].plot(jnp.arange(T_Gamma), IRF.ln_Gamma)

        ax[0,0].set_title(r'$u_t$')
        ax[0,1].set_title(r'$z_t$')
        ax[0,2].set_title(r'$\ln(\Gamma_t)$')

        # nn: Y (2nd row)
        ax[1,0].plot(jnp.arange(T_u), IRF.Y_u, label='DEQN', color='red')
        ax[1,1].plot(jnp.arange(T_z), IRF.Y_z, color='red')
        ax[1,2].plot(jnp.arange(T_Gamma), IRF.Y_ln_Gamma, color='red')
        for i in range(3): ax[1,i].set_title('Output')

        # nn: pi (3rd row)
        ax[2,0].plot(jnp.arange(T_u), IRF.pi_u, color='red')
        ax[2,1].plot(jnp.arange(T_z), IRF.pi_z, color='red')
        ax[2,2].plot(jnp.arange(T_Gamma), IRF.pi_ln_Gamma, color='red')
        for i in range(3): ax[2,i].set_title('Inflation')

        if plot_extra:
            
            # nn: Y (2nd row)
            ax[1,0].plot(jnp.arange(T_u), IRF.Y_extra_u, label='DEQN w/o ZLB', color='orange', ls='--')
            ax[1,1].plot(jnp.arange(T_z), IRF.Y_extra_z, color='orange', ls='--')
            ax[1,2].plot(jnp.arange(T_Gamma), IRF.Y_extra_ln_Gamma, color='orange', ls='--')
            
            # nn: pi (3rd row)
            ax[2,0].plot(jnp.arange(T_u), IRF.pi_extra_u, color='orange', ls='--')
            ax[2,1].plot(jnp.arange(T_z), IRF.pi_extra_z, color='orange', ls='--')
            ax[2,2].plot(jnp.arange(T_Gamma), IRF.pi_extra_ln_Gamma, color='orange', ls='--')
            

        # linear: Y (2nd row)
        ax[1,0].plot(jnp.arange(T_u), (P[0,0] * IRF.u)/Y_DSS, label='Linear', color='green')
        ax[1,1].plot(jnp.arange(T_z), (P[0,1] * IRF.z)/Y_DSS, color='green')
        ax[1,2].plot(jnp.arange(T_Gamma), (P[0,2] * IRF.ln_Gamma)/Y_DSS, color='green')

        # linear: pi (3rd row)
        ax[2,0].plot(jnp.arange(T_u), P[1,0] * IRF.u, color='green')
        ax[2,1].plot(jnp.arange(T_z), P[1,1] * IRF.z, color='green')
        ax[2,2].plot(jnp.arange(T_Gamma), P[1,2] * IRF.ln_Gamma, color='green')

        # OccBin: Y (2nd row)
        ax[1,0].plot(jnp.arange(T_u), IRF.Y_u_OccBin, label='OccBin', ls ='--', color='purple')
        ax[1,1].plot(jnp.arange(T_z), IRF.Y_z_OccBin, ls='--', color='purple')
        ax[1,2].plot(jnp.arange(T_Gamma), IRF.Y_ln_Gamma_OccBin, ls='--', color='purple')

        # OccBin: pi (3rd row)
        ax[2,0].plot(jnp.arange(T_u), IRF.pi_u_OccBin , ls ='--', color='purple')
        ax[2,1].plot(jnp.arange(T_z), IRF.pi_z_OccBin , ls ='--', color='purple')
        ax[2,2].plot(jnp.arange(T_Gamma), IRF.pi_ln_Gamma_OccBin , ls ='--', color='purple')

        f.tight_layout(rect=[0, 0.1, 1, 1])

        # 2. Placer nu legenden i det frie område
        f.legend(loc='lower center', 
                bbox_to_anchor=(0.5, 0.05), # 0.02 er lige over bunden i det tomme felt
                ncol=5,          
                frameon=False)

        if save_path is not None:
            f.savefig(save_path)


    def plot_GIRF(self, save_path=None, plot_extra=False, plot_OccBin=True):

        par = self.par
        linear = self.linear
        IRF = self.GIRF

        T_u = IRF.T_u
        T_z = IRF.T_z
        T_Gamma = IRF.T_Gamma
        P = linear["P"]
        Y_DSS = par["Y_DSS"]
        
        f, ax = plt.subplots(3, 3, figsize=(12, 12))

        # shocks (1st row)
        ax[0,0].plot(jnp.arange(T_u), IRF.u)
        ax[0,1].plot(jnp.arange(T_z), IRF.z)
        ax[0,2].plot(jnp.arange(T_Gamma), IRF.ln_Gamma)

        ax[0,0].set_title(r'$u_t$')
        ax[0,1].set_title(r'$z_t$')
        ax[0,2].set_title(r'$\ln(\Gamma_t)$')

        # nn: Y (2nd row)
        ax[1,0].plot(jnp.arange(T_u), IRF.Y_u, label='DEQN', color='red')
        ax[1,1].plot(jnp.arange(T_z), IRF.Y_z, color='red')
        ax[1,2].plot(jnp.arange(T_Gamma), IRF.Y_ln_Gamma, color='red')
        for i in range(3): ax[1,i].set_title('Output')

        # nn: pi (3rd row)
        ax[2,0].plot(jnp.arange(T_u), IRF.pi_u, color='red')
        ax[2,1].plot(jnp.arange(T_z), IRF.pi_z, color='red')
        ax[2,2].plot(jnp.arange(T_Gamma), IRF.pi_ln_Gamma, color='red')
        for i in range(3): ax[2,i].set_title('Inflation')

        if plot_extra:
            
            # nn: Y (2nd row)
            ax[1,0].plot(jnp.arange(T_u), IRF.Y_extra_u, label='DEQN w/o ZLB', color='orange', ls='--')
            ax[1,1].plot(jnp.arange(T_z), IRF.Y_extra_z, color='orange', ls='--')
            ax[1,2].plot(jnp.arange(T_Gamma), IRF.Y_extra_ln_Gamma, color='orange', ls='--')
            
            # nn: pi (3rd row)
            ax[2,0].plot(jnp.arange(T_u), IRF.pi_extra_u, color='orange', ls='--')
            ax[2,1].plot(jnp.arange(T_z), IRF.pi_extra_z, color='orange', ls='--')
            ax[2,2].plot(jnp.arange(T_Gamma), IRF.pi_extra_ln_Gamma, color='orange', ls='--')
            

        # linear: Y (2nd row)
        ax[1,0].plot(jnp.arange(T_u), (P[0,0] * IRF.u)/Y_DSS, label='linear', color='green', ls='dotted')
        ax[1,1].plot(jnp.arange(T_z), (P[0,1] * IRF.z)/Y_DSS, color='green', ls='dotted')
        ax[1,2].plot(jnp.arange(T_Gamma), (P[0,2] * IRF.ln_Gamma)/Y_DSS, color='green', ls='dotted')

        # linear: pi (3rd row)
        ax[2,0].plot(jnp.arange(T_u), P[1,0] * IRF.u, color='green', ls='dotted')
        ax[2,1].plot(jnp.arange(T_z), P[1,1] * IRF.z, color='green', ls='dotted')
        ax[2,2].plot(jnp.arange(T_Gamma), P[1,2] * IRF.ln_Gamma, color='green', ls='dotted')

        if plot_OccBin:
            # OccBin: Y (2nd row)
            ax[1,0].plot(jnp.arange(T_u), IRF.Y_u_OccBin, label='OccBin', ls ='--', color='purple')
            ax[1,1].plot(jnp.arange(T_z), IRF.Y_z_OccBin, ls='--', color='purple')
            ax[1,2].plot(jnp.arange(T_Gamma), IRF.Y_Gamma_OccBin, ls='--', color='purple')

            # OccBin: pi (3rd row)
            ax[2,0].plot(jnp.arange(T_u), IRF.pi_u_OccBin, ls ='--', color='purple')
            ax[2,1].plot(jnp.arange(T_z), IRF.pi_z_OccBin, ls ='--', color='purple')
            ax[2,2].plot(jnp.arange(T_Gamma), IRF.pi_Gamma_OccBin, ls ='--', color='purple')

        f.tight_layout(rect=[0, 0.1, 1, 1])

        # 2. Placer nu legenden i det frie område
        f.legend(loc='lower center', 
                bbox_to_anchor=(0.5, 0.05), # 0.02 er lige over bunden i det tomme felt
                ncol=5,          
                frameon=False)

        if save_path is not None:
            f.savefig(save_path)
    
    def save(self, path, nn, opt=None):

        os.makedirs('output', exist_ok=True)

        _, nn_state = nnx.split(nn)
        serialised_nn = nn_state.to_pure_dict()

        if opt is None:
            save_dict = {
            'nn' : serialised_nn,
            }
            
        else:
            opt_state = opt.opt_state
            save_dict = {
            'nn' : serialised_nn,
            'opt' : opt_state
            }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    def load(self, path, opt_load=True):

        nn = self.nn
        opt = self.opt

        with open(path, 'rb') as f:
            load_dict = pickle.load(f)

        nnx.update(nn, load_dict['nn'])
        if opt_load == True: opt.opt_state = load_dict['opt']

#########
# tools #
#########

def construct_gh_nodes(gh_n, sigma_eps_u, sigma_eps_z, sigma_eps_Gamma):

    # 1. get raw nodes, weights
    x, w = gauss_hermite(gh_n)

    # 2. transformations to shock-specific quadrature
    x_u = jnp.sqrt(2) * sigma_eps_u * x
    x_z = jnp.sqrt(2) * sigma_eps_z * x
    x_Gamma = jnp.sqrt(2) * sigma_eps_Gamma * x
    w = w / jnp.sqrt(jnp.pi)

    # 3. compute combinatorics
    x = jnp.stack(jnp.meshgrid(x_u, x_z, x_Gamma, indexing="ij"), axis=-1)
    W_u, W_z, W_Gamma = jnp.meshgrid(w, w, w, indexing="ij")
    w = (W_u * W_z * W_Gamma).flatten()

    # 4. reshape to (gh_n^3, 3)
    x = x.reshape(-1, 3)

    return x, w

def compute_log_AR_mean(x_DSS, rho_x, sigma_x):
    
    num = (1-rho_x)*sigma_x
    denom = 2*(1-rho_x**2)

    return float((1-rho_x)*x_DSS-num/denom)

def compute_sigma_eps(rho_x, sigma_x):
    
    return float(jnp.sqrt(1-rho_x**2)*sigma_x)

def gauss_hermite(n):

    # a. calculations
    i = jnp.arange(1,n)
    a = jnp.sqrt(i/2)
    CM = jnp.diag(a,1) + jnp.diag(a,-1)
    L,V = jnp.linalg.eigh(CM)
    I = L.argsort()
    V = V[:,I].T

    # b. nodes and weights
    x = L[I]
    w = jnp.sqrt(jnp.pi)*V[:,0]**2

    return x,w