import numpy as np
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import os
import matplotlib.pyplot as plt
import pickle

from neural_nets import Policy, eval_nn, eval_nn_dev
from linear import setup_linear, eval_OccBin, compute_lin_OccBin_IRFs
from model_funcs import taylor_rule
from aux_ import draw_states_directly

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

        # structural parameters
        par["alpha"] = 0.25
        par["sigma"] = 1.0
        par["beta"] = 0.99
        par["epsilon"] = 9.0
        par["kappa"] = 0.099
        par["varphi"] = 1.0

        # CB parameters
        par["phi_y"] = 0.5
        par["phi_pi"] = 1.5
        par["ZLB"] = -0.005

        # implied DSS
        par["Y_DSS"] = ((1-par["alpha"]) * ((par["epsilon"]-1)/(par["epsilon"])))**((1-par["alpha"])/(par["varphi"]+par["alpha"]+par["sigma"]-par["alpha"]*par["sigma"]))
        par["pi_DSS"] = 0.00
        par["i_DSS"] = (1/par["beta"])-1
        par["u_DSS"] = 0.0
        par["z_DSS"] = 0.0
        par["ln_Gamma_DSS"] = 0.0
        
        # sunspot DSS (ZLB-DSS)
        par["Y_ZLB"], par["pi_ZLB"], par["i_ZLB"] = compute_sunspot_DSS(par)

        # shocks
        par["rho_u"] = 0.84000
        par["rho_z"] = 0.74872
        par["rho_Gamma"] = 0.68245

        par["sigma_eps_u"] =  0.001350,
        par["sigma_eps_z"] =  0.020814,
        par["sigma_eps_Gamma"] = 0.012778

        # model
        par["Nshocks"] = 3 # eps_u, eps_z, eps_Gamma
        par["Nstates"] = 3 # u, z, ln_Gamma
        par["Npolicies"] = 2 # Y, pi

        self.par = par

    def setup_train(self):

        train = {}

        train["neurons"] = (100, 100) # tuple so it is hashable
        train["N_test"] = 20000
        train["lr"] = 1e-4
        train["gh_n_per_shock"] = 4
        train["gh_n"] = train["gh_n_per_shock"]**3

        # OccBin
        train["T_OccBin"] = 100 # max "shooting length" in OccBin algo

        self.train = train

    def setup_linear(self):

        train = self.train

        T_OccBin = train["T_OccBin"]

        setup_linear(self, T_OccBin=T_OccBin)
    
    def setup_nn(self):

        par = self.par
        train = self.train
        linear = self.linear
        dtype = self.dtype
        device = self.device

        # 1. compute in- and output dimensions and retrieve list of neurons
        din = par["Nstates"]
        dout = par["Npolicies"]
        neurons = train["neurons"]
        lr = train["lr"]

        # 2. call policy class
        nn = Policy(din, dout, neurons, rngs=nnx.Rngs(params=0), dtype=dtype, device=device) # last kwarg set seed=0 for bias and weight initialization

        # 3. setup optimizer
        def make_opt(learning_rate, clipping_value):
            return optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate=learning_rate)
            )

        # 4. setup function for making optimizer
        optimizer_with_hparams = optax.inject_hyperparams(make_opt)(learning_rate=lr, clipping_value=1.0)

        # 5. inject function into nnx
        opt = nnx.ModelAndOptimizer(nn, optimizer_with_hparams)#, optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr)))

        # 6. save in model
        self.nn = nn
        self.opt = opt

    def compute_IRF(self, sigma_dict, rtol=50, extra_nn=None, u_neg=False, z_neg=False, ln_Gamma_neg=False, T=None):

        par = self.par
        train = self.train
        linear = self.linear
        nn = self.nn

        Y_DSS = par["Y_DSS"]

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
        IRF_Y_u, IRF_pi_u = eval_nn_dev(par, train, linear, nn, u_shock_states, T_u)
        if extra_nn: IRF_Y_u_extra, IRF_pi_u_extra = eval_nn_dev(par, train, linear, extra_nn, u_shock_states, T_u)

        # preference shock
        z_shock_states = jnp.concat([jnp.zeros((T_z, 1)), (z_shock * rho_z**jnp.arange(T_z))[:, None], jnp.zeros((T_z, 1))], axis=-1)
        IRF_Y_z, IRF_pi_z = eval_nn_dev(par, train, linear, nn, z_shock_states, T_z)
        if extra_nn: IRF_Y_z_extra, IRF_pi_z_extra = eval_nn_dev(par, train, linear, extra_nn, z_shock_states, T_z)

        # productivity shock
        ln_Gamma_shock_states = jnp.concat([jnp.zeros((T_Gamma, 1)), jnp.zeros((T_Gamma, 1)), (ln_Gamma_shock * rho_Gamma**jnp.arange(T_Gamma))[:, None]], axis=-1)
        IRF_Y_ln_Gamma, IRF_pi_ln_Gamma = eval_nn_dev(par, train, linear, nn, ln_Gamma_shock_states, T_Gamma)
        if extra_nn: IRF_Y_ln_Gamma_extra, IRF_pi_ln_Gamma_extra = eval_nn_dev(par, train, linear, extra_nn, ln_Gamma_shock_states, T_Gamma)

        IRF = SimpleNamespace()

        IRF.T_u = T_u
        IRF.T_z = T_z
        IRF.T_Gamma = T_Gamma

        IRF.u = u_shock * rho_u**jnp.arange(T_u)
        IRF.z = z_shock * rho_z**jnp.arange(T_z)
        IRF.ln_Gamma = ln_Gamma_shock * rho_Gamma**jnp.arange(T_Gamma)

        IRF.Y_u = IRF_Y_u
        IRF.Y_z = IRF_Y_z
        IRF.Y_ln_Gamma = IRF_Y_ln_Gamma

        IRF.pi_u = IRF_pi_u
        IRF.pi_z = IRF_pi_z
        IRF.pi_ln_Gamma = IRF_pi_ln_Gamma

        if extra_nn:
            IRF.Y_u_extra = IRF_Y_u_extra
            IRF.Y_z_extra = IRF_Y_z_extra
            IRF.Y_ln_Gamma_extra = IRF_Y_ln_Gamma_extra

            IRF.pi_u_extra = IRF_pi_u_extra
            IRF.pi_z_extra = IRF_pi_z_extra
            IRF.pi_ln_Gamma_extra = IRF_pi_ln_Gamma_extra

        self.IRF = IRF

        # compute linear and OccBin IRFs (fills in IRF namespace)
        compute_lin_OccBin_IRFs(self, sigma_dict, rtol=rtol, u_neg=u_neg, z_neg=z_neg, ln_Gamma_neg=ln_Gamma_neg, T=T)

    def compute_GIRF(self, sigma_dict, N, key_number=42, rtol=50, extra_nn=None, u_neg=False, z_neg=False, ln_Gamma_neg=False):

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
        K = linear["K"].T

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
        T_u = T_z = T_Gamma = 7
        T_max = int(jnp.array([T_u, T_z, T_Gamma]).max())

        # compute control
        key = jax.random.key(key_number)
        key, subkey = jax.random.split(key)

        # draw states for initial period and simulate deterministicly
        states_init = draw_states_directly(subkey, par, dtype, N, sigma_dict["sigma_eps_u"], sigma_dict["sigma_eps_z"], sigma_dict["sigma_eps_Gamma"])
        def state_trans(carry, _):
            states = carry
            states_next = carry @ K
            return states_next, states

        _, states = jax.lax.scan(state_trans, states_init, None, length=T_max) # shape (T, N, 3)

        u_lower = jnp.percentile(states[..., 0], 2.5, axis=1)
        u_upper = jnp.percentile(states[..., 0], 97.5, axis=1)

        z_lower = jnp.percentile(states[..., 1], 2.5, axis=1)
        z_upper = jnp.percentile(states[..., 1], 97.5, axis=1)

        ln_Gamma_lower = jnp.percentile(states[..., 2], 2.5, axis=1)
        ln_Gamma_upper = jnp.percentile(states[..., 2], 97.5, axis=1)

        states_flat = states.reshape(-1, 3)

        Y_control, pi_control = eval_nn(par, train, linear, nn, states_flat, T_max*N)
        Y_control, pi_control = Y_control.reshape(T_max, N), pi_control.reshape(T_max, N)
        i_control = taylor_rule(par, Y_control, pi_control, states[..., 0], states[..., 1], states[..., 2], 0.00, 0.00, par["ZLB"], 0.00)
        
        Y_OccBin_control, pi_OccBin_control = eval_OccBin(self, states_flat,  return_dev=False)
        Y_OccBin_control, pi_OccBin_control = Y_OccBin_control.reshape(T_max, N), pi_OccBin_control.reshape(T_max, N)
        i_OccBin_control = taylor_rule(par, Y_OccBin_control, pi_OccBin_control, states[..., 0], states[..., 1], states[..., 2], 0.00, 0.00, par["ZLB"], 0.00)

        if extra_nn is not None:
            Y_extra_control, pi_extra_control = eval_nn(par, train, linear, extra_nn, states_flat, T_max*N)
            Y_extra_control, pi_extra_control = Y_extra_control.reshape(T_max, N), pi_extra_control.reshape(T_max, N)
            i_extra_control = taylor_rule(par, Y_extra_control, pi_extra_control, states[..., 0], states[..., 1], states[..., 2], 0.00, 0.00, par["ZLB"], 0.00)

        # MP shock
        u_shock_states = jnp.concat([(u_shock * rho_u**jnp.arange(T_max))[:, None], jnp.zeros((T_max, 1)), jnp.zeros((T_max, 1))], axis=-1) # (T, 3)
        u_shock_states = states + u_shock_states[:, None, :] # (T, N, 3)

        # MP shock: nn
        Y_u_raw, pi_u_raw = eval_nn(par, train, linear, nn, u_shock_states.reshape(-1, 3), T_max*N)
        Y_u_raw, pi_u_raw = Y_u_raw.reshape(T_max, N), pi_u_raw.reshape(T_max, N)
        Y_u_dist = (Y_u_raw[:T_u] - Y_control[:T_u])/Y_control[:T_u]
        pi_u_dist = pi_u_raw[:T_u] - pi_control[:T_u]
        i_u_raw = taylor_rule(par, Y_u_raw[:T_u], pi_u_raw[:T_u], u_shock_states[:T_u, :, 0], u_shock_states[:T_u, :, 1], u_shock_states[:T_u, :, 2], 0.00, 0.00, par["ZLB"], 0.00)
        i_u_dist = i_u_raw - i_control[:T_u]
        
        Y_u = jnp.mean(Y_u_dist, axis=1)
        Y_u_lower = jnp.percentile(Y_u_dist, 2.5, axis=1)
        Y_u_upper = jnp.percentile(Y_u_dist, 97.5, axis=1)
        pi_u = jnp.mean(pi_u_dist, axis=1)
        pi_u_lower = jnp.percentile(pi_u_dist, 2.5, axis=1)
        pi_u_upper = jnp.percentile(pi_u_dist, 97.5, axis=1)
        i_u = jnp.mean(i_u_dist, axis=1)
        i_u_lower = jnp.percentile(i_u_dist, 2.5, axis=1)
        i_u_upper = jnp.percentile(i_u_dist, 97.5, axis=1)
        u_zlb_share = jnp.mean(i_u_raw <= par["ZLB"], axis=1)

        # MP shock: OccBin
        Y_u_OccBin_raw, pi_u_OccBin_raw = eval_OccBin(self, u_shock_states.reshape(-1,3), return_dev=False)
        Y_u_OccBin_raw, pi_u_OccBin_raw = Y_u_OccBin_raw.reshape(T_max, N), pi_u_OccBin_raw.reshape(T_max, N)
        Y_u_OccBin_dist = (Y_u_OccBin_raw[:T_u]-Y_OccBin_control[:T_u])/Y_OccBin_control[:T_u]
        pi_u_OccBin_dist = pi_u_OccBin_raw[:T_u]-pi_OccBin_control[:T_u]
        i_u_OccBin_raw = taylor_rule(par, Y_u_OccBin_raw[:T_u], pi_u_OccBin_raw[:T_u], u_shock_states[:T_u, :, 0], u_shock_states[:T_u, :, 1], u_shock_states[:T_u, :, 2], 0.00, 0.00, par["ZLB"], 0.00)
        i_u_OccBin_dist = i_u_OccBin_raw - i_OccBin_control[:T_u]
        
        Y_u_OccBin = jnp.mean(Y_u_OccBin_dist, axis=1)
        Y_u_OccBin_lower = jnp.percentile(Y_u_OccBin_dist, 2.5, axis=1)
        Y_u_OccBin_upper = jnp.percentile(Y_u_OccBin_dist, 97.5, axis=1)
        pi_u_OccBin = jnp.mean(pi_u_OccBin_dist, axis=1)
        pi_u_OccBin_lower = jnp.percentile(pi_u_OccBin_dist, 2.5, axis=1)
        pi_u_OccBin_upper = jnp.percentile(pi_u_OccBin_dist, 97.5, axis=1)
        i_u_OccBin = jnp.mean(i_u_OccBin_dist, axis=1)
        i_u_OccBin_lower = jnp.percentile(i_u_OccBin_dist, 2.5, axis=1)
        i_u_OccBin_upper = jnp.percentile(i_u_OccBin_dist, 97.5, axis=1)
        u_zlb_share_OccBin = jnp.mean(i_u_OccBin_raw <= par["ZLB"], axis=1)
        
        if extra_nn:
            Y_u_extra_raw, pi_u_extra_raw = eval_nn(par, train, linear, extra_nn, u_shock_states.reshape(-1, 3), T_max*N)
            Y_u_extra_raw, pi_u_extra_raw = Y_u_extra_raw.reshape(T_max, N), pi_u_extra_raw.reshape(T_max, N)
            Y_u_extra_dist = (Y_u_extra_raw[:T_u] - Y_extra_control[:T_u])/Y_extra_control[:T_u]
            pi_u_extra_dist = pi_u_extra_raw[:T_u] - pi_extra_control[:T_u]
            i_u_extra_raw = taylor_rule(par, Y_u_extra_raw[:T_u], pi_u_extra_raw[:T_u], u_shock_states[:T_u, :, 0], u_shock_states[:T_u, :, 1], u_shock_states[:T_u, :, 2], 0.00, 0.00, par["ZLB"], 0.00)
            i_u_extra_dist = i_u_extra_raw - i_extra_control[:T_u]
            
            Y_u_extra =  jnp.mean(Y_u_extra_dist, axis=1)
            Y_u_extra_lower = jnp.percentile(Y_u_extra_dist, 2.5, axis=1)
            Y_u_extra_upper = jnp.percentile(Y_u_extra_dist, 97.5, axis=1)
            pi_u_extra =  jnp.mean(pi_u_extra_dist, axis=1)
            pi_u_extra_lower = jnp.percentile(pi_u_extra_dist, 2.5, axis=1)
            pi_u_extra_upper = jnp.percentile(pi_u_extra_dist, 97.5, axis=1)
            i_u_extra = jnp.mean(i_u_extra_dist, axis=1)
            i_u_extra_lower = jnp.percentile(i_u_extra_dist, 2.5, axis=1)
            i_u_extra_upper = jnp.percentile(i_u_extra_dist, 97.5, axis=1)
            u_zlb_share_extra = jnp.mean(i_u_extra_raw <= par["ZLB"], axis=1)

        # preference shock
        z_shock_states = jnp.concat([jnp.zeros((T_max, 1)), (z_shock * rho_z**jnp.arange(T_max))[:, None], jnp.zeros((T_max, 1))], axis=-1)
        z_shock_states = states + z_shock_states[:, None, :]
        
        # preference shock: nn
        Y_z_raw, pi_z_raw = eval_nn(par, train, linear, nn, z_shock_states.reshape(-1,3), T_max*N)
        Y_z_raw, pi_z_raw = Y_z_raw.reshape(T_max, N), pi_z_raw.reshape(T_max, N)
        Y_z_dist = (Y_z_raw[:T_z] - Y_control[:T_z])/Y_control[:T_z]
        pi_z_dist = pi_z_raw[:T_z] - pi_control[:T_z]
        i_z_raw = taylor_rule(par, Y_z_raw[:T_z], pi_z_raw[:T_z], z_shock_states[:T_z, :, 0], z_shock_states[:T_z, :, 1], z_shock_states[:T_z, :, 2], 0.00, 0.00, par["ZLB"], 0.00)
        i_z_dist = i_z_raw - i_control[:T_z]
        
        Y_z = jnp.mean(Y_z_dist, axis=1)
        Y_z_lower = jnp.percentile(Y_z_dist, 2.5, axis=1)
        Y_z_upper = jnp.percentile(Y_z_dist, 97.5, axis=1)
        pi_z = jnp.mean(pi_z_dist, axis=1)
        pi_z_lower = jnp.percentile(pi_z_dist, 2.5, axis=1)
        pi_z_upper = jnp.percentile(pi_z_dist, 97.5, axis=1)
        i_z = jnp.mean(i_z_dist, axis=1)
        i_z_lower = jnp.percentile(i_z_dist, 2.5, axis=1)
        i_z_upper = jnp.percentile(i_z_dist, 97.5, axis=1)
        z_zlb_share = jnp.mean(i_z_raw <= par["ZLB"], axis=1)

        # preference shock: OccBin
        Y_z_OccBin_raw, pi_z_OccBin_raw = eval_OccBin(self, z_shock_states.reshape(-1,3), return_dev=False)
        Y_z_OccBin_raw, pi_z_OccBin_raw = Y_z_OccBin_raw.reshape(T_max, N), pi_z_OccBin_raw.reshape(T_max, N)
        Y_z_OccBin_dist = (Y_z_OccBin_raw[:T_z]-Y_OccBin_control[:T_z])/(Y_OccBin_control[:T_z])
        pi_z_OccBin_dist = pi_z_OccBin_raw[:T_z]-pi_OccBin_control[:T_z]
        i_z_OccBin_raw = taylor_rule(par, Y_z_OccBin_raw[:T_z], pi_z_OccBin_raw[:T_z], z_shock_states[:T_z, :, 0], z_shock_states[:T_z, :, 1], z_shock_states[:T_z, :, 2], 0.00, 0.00, par["ZLB"], 0.00)
        i_z_OccBin_dist = i_z_OccBin_raw - i_OccBin_control[:T_z]
        
        Y_z_OccBin = jnp.mean(Y_z_OccBin_dist, axis=1)
        Y_z_OccBin_lower = jnp.percentile(Y_z_OccBin_dist, 2.5, axis=1)
        Y_z_OccBin_upper = jnp.percentile(Y_z_OccBin_dist, 97.5, axis=1)
        pi_z_OccBin = jnp.mean(pi_z_OccBin_dist, axis=1)
        pi_z_OccBin_lower = jnp.percentile(pi_z_OccBin_dist, 2.5, axis=1)
        pi_z_OccBin_upper = jnp.percentile(pi_z_OccBin_dist, 97.5, axis=1)
        i_z_OccBin = jnp.mean(i_z_OccBin_dist, axis=1)
        i_z_OccBin_lower = jnp.percentile(i_z_OccBin_dist, 2.5, axis=1)
        i_z_OccBin_upper = jnp.percentile(i_z_OccBin_dist, 97.5, axis=1)
        z_zlb_share_OccBin = jnp.mean(i_z_OccBin_raw <= par["ZLB"], axis=1)

        if extra_nn:
            Y_z_extra_raw, pi_z_extra_raw = eval_nn(par, train, linear, extra_nn, z_shock_states.reshape(-1, 3), T_max*N)
            Y_z_extra_raw, pi_z_extra_raw = Y_z_extra_raw.reshape(T_max, N), pi_z_extra_raw.reshape(T_max, N)
            Y_z_extra_dist = (Y_z_extra_raw[:T_z] - Y_extra_control[:T_z])/Y_extra_control[:T_z]
            pi_z_extra_dist = pi_z_extra_raw[:T_z] - pi_extra_control[:T_z]
            i_z_extra_raw = taylor_rule(par, Y_z_extra_raw[:T_z], pi_z_extra_raw[:T_z], z_shock_states[:T_z, :, 0], z_shock_states[:T_z, :, 1], z_shock_states[:T_z, :, 2], 0.00, 0.00, par["ZLB"], 0.00)
            i_z_extra_dist = i_z_extra_raw - i_extra_control[:T_z]
            
            Y_z_extra = jnp.mean(Y_z_extra_dist, axis=1)
            Y_z_extra_lower = jnp.percentile(Y_z_extra_dist, 2.5, axis=1)
            Y_z_extra_upper = jnp.percentile(Y_z_extra_dist, 97.5, axis=1)
            pi_z_extra = jnp.mean(pi_z_extra_dist, axis=1)
            pi_z_extra_lower = jnp.percentile(pi_z_extra_dist, 2.5, axis=1)
            pi_z_extra_upper = jnp.percentile(pi_z_extra_dist, 97.5, axis=1)
            i_z_extra = jnp.mean(i_z_extra_dist, axis=1)
            i_z_extra_lower = jnp.percentile(i_z_extra_dist, 2.5, axis=1)
            i_z_extra_upper = jnp.percentile(i_z_extra_dist, 97.5, axis=1)
            z_zlb_share_extra = jnp.mean(i_z_extra_raw <= par["ZLB"], axis=1)

        # productivity shock
        ln_Gamma_shock_states = jnp.concat([jnp.zeros((T_max, 1)), jnp.zeros((T_max, 1)), (ln_Gamma_shock * rho_Gamma**jnp.arange(T_max))[:, None]], axis=-1)
        ln_Gamma_shock_states = states + ln_Gamma_shock_states[:, None, :]
        
        # productivity shock: nn
        Y_ln_Gamma_raw, pi_ln_Gamma_raw = eval_nn(par, train, linear, nn, ln_Gamma_shock_states.reshape(-1, 3), T_max*N)
        Y_ln_Gamma_raw, pi_ln_Gamma_raw = Y_ln_Gamma_raw.reshape(T_max, N), pi_ln_Gamma_raw.reshape(T_max, N)
        Y_ln_Gamma_dist = (Y_ln_Gamma_raw[:T_Gamma] - Y_control[:T_Gamma])/Y_control[:T_Gamma]
        pi_ln_Gamma_dist = pi_ln_Gamma_raw[:T_Gamma] - pi_control[:T_Gamma]
        i_ln_Gamma_raw = taylor_rule(par, Y_ln_Gamma_raw[:T_Gamma], pi_ln_Gamma_raw[:T_Gamma], ln_Gamma_shock_states[:T_Gamma, :, 0], ln_Gamma_shock_states[:T_Gamma, :, 1], ln_Gamma_shock_states[:T_Gamma, :, 2], 0.00, 0.00, par["ZLB"], 0.00)
        i_ln_Gamma_dist = i_ln_Gamma_raw - i_control[:T_Gamma]
        
        Y_ln_Gamma = jnp.mean(Y_ln_Gamma_dist, axis=1)
        Y_ln_Gamma_lower = jnp.percentile(Y_ln_Gamma_dist, 2.5, axis=1)
        Y_ln_Gamma_upper = jnp.percentile(Y_ln_Gamma_dist, 97.5, axis=1)
        pi_ln_Gamma = jnp.mean(pi_ln_Gamma_dist, axis=1)
        pi_ln_Gamma_lower = jnp.percentile(pi_ln_Gamma_dist, 2.5, axis=1)
        pi_ln_Gamma_upper = jnp.percentile(pi_ln_Gamma_dist, 97.5, axis=1)
        i_ln_Gamma = jnp.mean(i_ln_Gamma_dist, axis=1)
        i_ln_Gamma_lower = jnp.percentile(i_ln_Gamma_dist, 2.5, axis=1)
        i_ln_Gamma_upper = jnp.percentile(i_ln_Gamma_dist, 97.5, axis=1)
        ln_Gamma_zlb_share = jnp.mean(i_ln_Gamma_raw <= par["ZLB"], axis=1)

        # productivity shock: OccBin
        Y_ln_Gamma_OccBin_raw, pi_ln_Gamma_OccBin_raw = eval_OccBin(self, ln_Gamma_shock_states.reshape(-1, 3), return_dev=False)
        Y_ln_Gamma_OccBin_raw, pi_ln_Gamma_OccBin_raw = Y_ln_Gamma_OccBin_raw.reshape(T_max, N), pi_ln_Gamma_OccBin_raw.reshape(T_max, N)
        Y_ln_Gamma_OccBin_dist = (Y_ln_Gamma_OccBin_raw[:T_Gamma] - Y_OccBin_control[:T_Gamma])/(Y_OccBin_control[:T_Gamma])
        pi_ln_Gamma_OccBin_dist = pi_ln_Gamma_OccBin_raw[:T_Gamma] - pi_OccBin_control[:T_Gamma]
        i_ln_Gamma_OccBin_raw = taylor_rule(par, Y_ln_Gamma_OccBin_raw[:T_Gamma], pi_ln_Gamma_OccBin_raw[:T_Gamma], ln_Gamma_shock_states[:T_Gamma, :, 0], ln_Gamma_shock_states[:T_Gamma, :, 1], ln_Gamma_shock_states[:T_Gamma, :, 2], 0.00, 0.00, par["ZLB"], 0.00)
        i_ln_Gamma_OccBin_dist = i_ln_Gamma_OccBin_raw - i_OccBin_control[:T_Gamma]
        
        Y_ln_Gamma_OccBin = jnp.mean(Y_ln_Gamma_OccBin_dist, axis=1)
        Y_ln_Gamma_OccBin_lower = jnp.percentile(Y_ln_Gamma_OccBin_dist, 2.5, axis=1)
        Y_ln_Gamma_OccBin_upper = jnp.percentile(Y_ln_Gamma_OccBin_dist, 97.5, axis=1)
        pi_ln_Gamma_OccBin = jnp.mean(pi_ln_Gamma_OccBin_dist, axis=1)
        pi_ln_Gamma_OccBin_lower = jnp.percentile(pi_ln_Gamma_OccBin_dist, 2.5, axis=1)
        pi_ln_Gamma_OccBin_upper = jnp.percentile(pi_ln_Gamma_OccBin_dist, 97.5, axis=1)
        i_ln_Gamma_OccBin = jnp.mean(i_ln_Gamma_OccBin_dist, axis=1)
        i_ln_Gamma_OccBin_lower = jnp.percentile(i_ln_Gamma_OccBin_dist, 2.5, axis=1)
        i_ln_Gamma_OccBin_upper = jnp.percentile(i_ln_Gamma_OccBin_dist, 97.5, axis=1)
        ln_Gamma_zlb_share_OccBin = jnp.mean(i_ln_Gamma_OccBin_raw <= par["ZLB"], axis=1)

        if extra_nn:
            Y_ln_Gamma_extra_raw, pi_ln_Gamma_extra_raw = eval_nn(par, train, linear, extra_nn, ln_Gamma_shock_states.reshape(-1, 3), T_max*N)
            Y_ln_Gamma_extra_raw, pi_ln_Gamma_extra_raw = Y_ln_Gamma_extra_raw.reshape(T_max, N), pi_ln_Gamma_extra_raw.reshape(T_max, N)
            Y_ln_Gamma_extra_dist = (Y_ln_Gamma_extra_raw[:T_Gamma] - Y_extra_control[:T_Gamma])/Y_extra_control[:T_Gamma]
            pi_ln_Gamma_extra_dist = pi_ln_Gamma_extra_raw[:T_Gamma] - pi_extra_control[:T_Gamma]
            i_ln_Gamma_extra_raw = taylor_rule(par, Y_ln_Gamma_extra_raw[:T_Gamma], pi_ln_Gamma_extra_raw[:T_Gamma], ln_Gamma_shock_states[:T_Gamma, :, 0], ln_Gamma_shock_states[:T_Gamma, :, 1], ln_Gamma_shock_states[:T_Gamma, :, 2], 0.00, 0.00, par["ZLB"], 0.00)
            i_ln_Gamma_extra_dist = i_ln_Gamma_extra_raw - i_extra_control[:T_Gamma]
            
            Y_ln_Gamma_extra = jnp.mean(Y_ln_Gamma_extra_dist, axis=1)
            Y_ln_Gamma_extra_lower = jnp.percentile(Y_ln_Gamma_extra_dist, 2.5, axis=1)
            Y_ln_Gamma_extra_upper = jnp.percentile(Y_ln_Gamma_extra_dist, 97.5, axis=1)
            pi_ln_Gamma_extra = jnp.mean(pi_ln_Gamma_extra_dist, axis=1)
            pi_ln_Gamma_extra_lower = jnp.percentile(pi_ln_Gamma_extra_dist, 2.5, axis=1)
            pi_ln_Gamma_extra_upper = jnp.percentile(pi_ln_Gamma_extra_dist, 97.5, axis=1)
            i_ln_Gamma_extra = jnp.mean(i_ln_Gamma_extra_dist, axis=1)
            i_ln_Gamma_extra_lower = jnp.percentile(i_ln_Gamma_extra_dist, 2.5, axis=1)
            i_ln_Gamma_extra_upper = jnp.percentile(i_ln_Gamma_extra_dist, 97.5, axis=1)
            ln_Gamma_zlb_share_extra = jnp.mean(i_ln_Gamma_extra_raw <= par["ZLB"], axis=1)

        GIRF = SimpleNamespace()

        GIRF.T_u = T_u
        GIRF.T_z = T_z
        GIRF.T_Gamma = T_Gamma

        GIRF.u = u_shock * rho_u**jnp.arange(T_max)
        GIRF.z = z_shock * rho_z**jnp.arange(T_max)
        GIRF.ln_Gamma = ln_Gamma_shock * rho_Gamma**jnp.arange(T_max)

        GIRF.u_lower = u_lower
        GIRF.u_upper = u_upper
        
        GIRF.z_lower = z_lower
        GIRF.z_upper = z_upper

        GIRF.ln_Gamma_lower = ln_Gamma_lower
        GIRF.ln_Gamma_upper = ln_Gamma_upper

        # Output (Y)
        GIRF.Y_u = Y_u
        GIRF.Y_z = Y_z
        GIRF.Y_ln_Gamma = Y_ln_Gamma

        GIRF.Y_u_lower = Y_u_lower
        GIRF.Y_u_upper = Y_u_upper
        GIRF.Y_z_lower = Y_z_lower
        GIRF.Y_z_upper = Y_z_upper
        GIRF.Y_ln_Gamma_lower = Y_ln_Gamma_lower
        GIRF.Y_ln_Gamma_upper = Y_ln_Gamma_upper

        # Inflation (pi)
        GIRF.pi_u = pi_u
        GIRF.pi_z = pi_z
        GIRF.pi_ln_Gamma = pi_ln_Gamma

        GIRF.pi_u_lower = pi_u_lower
        GIRF.pi_u_upper = pi_u_upper
        GIRF.pi_z_lower = pi_z_lower
        GIRF.pi_z_upper = pi_z_upper
        GIRF.pi_ln_Gamma_lower = pi_ln_Gamma_lower
        GIRF.pi_ln_Gamma_upper = pi_ln_Gamma_upper

        # Interest rate (i)
        GIRF.i_u = i_u
        GIRF.i_z = i_z
        GIRF.i_ln_Gamma = i_ln_Gamma

        GIRF.i_u_lower = i_u_lower
        GIRF.i_u_upper = i_u_upper
        GIRF.i_z_lower = i_z_lower
        GIRF.i_z_upper = i_z_upper
        GIRF.i_ln_Gamma_lower = i_ln_Gamma_lower
        GIRF.i_ln_Gamma_upper = i_ln_Gamma_upper

        # ZLB share
        GIRF.u_zlb_share = u_zlb_share
        GIRF.z_zlb_share = z_zlb_share
        GIRF.ln_Gamma_zlb_share = ln_Gamma_zlb_share

        # OccBin: Y
        GIRF.Y_u_OccBin = Y_u_OccBin
        GIRF.Y_z_OccBin = Y_z_OccBin
        GIRF.Y_Gamma_OccBin = Y_ln_Gamma_OccBin

        GIRF.Y_u_OccBin_lower = Y_u_OccBin_lower
        GIRF.Y_u_OccBin_upper = Y_u_OccBin_upper
        GIRF.Y_z_OccBin_lower = Y_z_OccBin_lower
        GIRF.Y_z_OccBin_upper = Y_z_OccBin_upper
        GIRF.Y_Gamma_OccBin_lower = Y_ln_Gamma_OccBin_lower
        GIRF.Y_Gamma_OccBin_upper = Y_ln_Gamma_OccBin_upper

        # OccBin: pi
        GIRF.pi_u_OccBin = pi_u_OccBin
        GIRF.pi_z_OccBin = pi_z_OccBin
        GIRF.pi_Gamma_OccBin = pi_ln_Gamma_OccBin

        GIRF.pi_u_OccBin_lower = pi_u_OccBin_lower
        GIRF.pi_u_OccBin_upper = pi_u_OccBin_upper
        GIRF.pi_z_OccBin_lower = pi_z_OccBin_lower
        GIRF.pi_z_OccBin_upper = pi_z_OccBin_upper
        GIRF.pi_Gamma_OccBin_lower = pi_ln_Gamma_OccBin_lower
        GIRF.pi_Gamma_OccBin_upper = pi_ln_Gamma_OccBin_upper

        # OccBin: i
        GIRF.i_u_OccBin = i_u_OccBin
        GIRF.i_z_OccBin = i_z_OccBin
        GIRF.i_Gamma_OccBin = i_ln_Gamma_OccBin

        GIRF.i_u_OccBin_lower = i_u_OccBin_lower
        GIRF.i_u_OccBin_upper = i_u_OccBin_upper
        GIRF.i_z_OccBin_lower = i_z_OccBin_lower
        GIRF.i_z_OccBin_upper = i_z_OccBin_upper
        GIRF.i_Gamma_OccBin_lower = i_ln_Gamma_OccBin_lower
        GIRF.i_Gamma_OccBin_upper = i_ln_Gamma_OccBin_upper

        # ZLB share
        GIRF.u_zlb_share_OccBin = u_zlb_share_OccBin
        GIRF.z_zlb_share_OccBin = z_zlb_share_OccBin
        GIRF.ln_Gamma_zlb_share_OccBin = ln_Gamma_zlb_share_OccBin

        if extra_nn:
            # Extra: Y
            GIRF.Y_extra_u = Y_u_extra
            GIRF.Y_extra_z = Y_z_extra
            GIRF.Y_extra_ln_Gamma = Y_ln_Gamma_extra

            GIRF.Y_extra_u_lower = Y_u_extra_lower
            GIRF.Y_extra_u_upper = Y_u_extra_upper
            GIRF.Y_extra_z_lower = Y_z_extra_lower
            GIRF.Y_extra_z_upper = Y_z_extra_upper
            GIRF.Y_extra_ln_Gamma_lower = Y_ln_Gamma_extra_lower
            GIRF.Y_extra_ln_Gamma_upper = Y_ln_Gamma_extra_upper

            # Extra: pi
            GIRF.pi_extra_u = pi_u_extra
            GIRF.pi_extra_z = pi_z_extra
            GIRF.pi_extra_ln_Gamma = pi_ln_Gamma_extra

            GIRF.pi_extra_u_lower = pi_u_extra_lower
            GIRF.pi_extra_u_upper = pi_u_extra_upper
            GIRF.pi_extra_z_lower = pi_z_extra_lower
            GIRF.pi_extra_z_upper = pi_z_extra_upper
            GIRF.pi_extra_ln_Gamma_lower = pi_ln_Gamma_extra_lower
            GIRF.pi_extra_ln_Gamma_upper = pi_ln_Gamma_extra_upper

            # Extra: i
            GIRF.i_extra_u = i_u_extra
            GIRF.i_extra_z = i_z_extra
            GIRF.i_extra_ln_Gamma = i_ln_Gamma_extra

            GIRF.i_extra_u_lower = i_u_extra_lower
            GIRF.i_extra_u_upper = i_u_extra_upper
            GIRF.i_extra_z_lower = i_z_extra_lower
            GIRF.i_extra_z_upper = i_z_extra_upper
            GIRF.i_extra_ln_Gamma_lower = i_ln_Gamma_extra_lower
            GIRF.i_extra_ln_Gamma_upper = i_ln_Gamma_extra_upper

            # ZLB share
            GIRF.u_zlb_share_extra = u_zlb_share_extra
            GIRF.z_zlb_share_extra = z_zlb_share_extra
            GIRF.ln_Gamma_zlb_share_extra = ln_Gamma_zlb_share_extra

        self.GIRF = GIRF

    def plot_IRF(self, save_path=None, plot_extra=False):

        par = self.par
        train = self.train
        linear = self.linear
        IRF = self.IRF

        T_u = IRF.T_u
        T_z = IRF.T_z
        T_Gamma = IRF.T_Gamma

        f, ax = plt.subplots(3, 3, figsize=(12, 12))

        # shocks (1st row)
        ax[0,0].plot(jnp.arange(T_u), IRF.u)
        ax[0,1].plot(jnp.arange(T_z), IRF.z)
        ax[0,2].plot(jnp.arange(T_Gamma), IRF.ln_Gamma)

        ax[0,0].set_title(r'Monetary Policy Shock: $u_t$')
        ax[0,1].set_title(r'Preference Shifter: $z_t$')
        ax[0,2].set_title(r'Productivity Shock: $\ln(\Gamma_t)$')

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
            ax[1,0].plot(jnp.arange(T_u), IRF.Y_u_extra, label='DEQN w/o ZLB', color='orange', ls='--')
            ax[1,1].plot(jnp.arange(T_z), IRF.Y_z_extra, color='orange', ls='--')
            ax[1,2].plot(jnp.arange(T_Gamma), IRF.Y_ln_Gamma_extra, color='orange', ls='--')
            
            # nn: pi (3rd row)
            ax[2,0].plot(jnp.arange(T_u), IRF.pi_u_extra, color='orange', ls='--')
            ax[2,1].plot(jnp.arange(T_z), IRF.pi_z_extra, color='orange', ls='--')
            ax[2,2].plot(jnp.arange(T_Gamma), IRF.pi_ln_Gamma_extra, color='orange', ls='--')

        # linear: Y (2nd row)
        ax[1,0].plot(jnp.arange(T_u), IRF.Y_u_lin, label='Linear', color='green', marker='o')
        ax[1,1].plot(jnp.arange(T_z), IRF.Y_z_lin, color='green', marker='o')
        ax[1,2].plot(jnp.arange(T_Gamma), IRF.Y_ln_Gamma_lin, color='green', marker='o')

        # linear: pi (3rd row)
        ax[2,0].plot(jnp.arange(T_u), IRF.pi_u_lin, color='green', marker='o')
        ax[2,1].plot(jnp.arange(T_z), IRF.pi_z_lin, color='green', marker='o')
        ax[2,2].plot(jnp.arange(T_Gamma), IRF.pi_ln_Gamma_lin, color='green', marker='o')

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

    def plot_GIRF(self, save_path=None, plot_extra=False, plot_OccBin=True, do_bands=True, label_step=1):

        par = self.par
        linear = self.linear
        IRF = self.IRF
        GIRF = self.GIRF

        T_u = IRF.T_u
        T_z = IRF.T_z
        T_Gamma = IRF.T_Gamma

        plt.rcParams.update({'font.size': 12})

        f, ax = plt.subplots(5, 3, figsize=(10, 15))

        def get_quarter_labels(T, step):
            ticks = [0] + list(range(step, T, step))
            labels = ["Impact"] + [f"Q{t}" for t in ticks[1:]]
            return ticks, labels

        ax[0,0].plot(jnp.arange(T_u), IRF.u)
        ax[0,1].plot(jnp.arange(T_z), IRF.z)
        ax[0,2].plot(jnp.arange(T_Gamma), IRF.ln_Gamma)

        ax[0,0].set_title(r'Monetary Policy Shock: $u_t$')
        ax[0,1].set_title(r'Preference Shifter: $z_t$')
        ax[0,2].set_title(r'Productivity: $\ln(\Gamma_t)$')

        # if do_bands: ax[0,0].fill_between(jnp.arange(T_u), GIRF.u_lower, GIRF.u_upper, alpha=0.15)
        # if do_bands: ax[0,1].fill_between(jnp.arange(T_z), GIRF.z_lower, GIRF.z_upper, alpha=0.15)
        # if do_bands: ax[0,2].fill_between(jnp.arange(T_Gamma), GIRF.ln_Gamma_lower, GIRF.ln_Gamma_upper, alpha=0.15)

        # Række 1: Output Y (ganget med 100)
        ax[1,0].plot(jnp.arange(T_u), GIRF.Y_u * 100, label='DEQN', color='red')
        if do_bands: ax[1,0].fill_between(jnp.arange(T_u), GIRF.Y_u_lower * 100, GIRF.Y_u_upper * 100, color='red', alpha=0.15, label='DEQN: 95 pct.')
        
        ax[1,1].plot(jnp.arange(T_z), GIRF.Y_z * 100, color='red')
        if do_bands: ax[1,1].fill_between(jnp.arange(T_z), GIRF.Y_z_lower * 100, GIRF.Y_z_upper * 100, color='red', alpha=0.15)
        
        ax[1,2].plot(jnp.arange(T_Gamma), GIRF.Y_ln_Gamma * 100, color='red')
        if do_bands: ax[1,2].fill_between(jnp.arange(T_Gamma), GIRF.Y_ln_Gamma_lower * 100, GIRF.Y_ln_Gamma_upper * 100, color='red', alpha=0.15)
        
        for i in range(3): 
            ax[1,i].set_title(r'Output: $Y_t$')
        ax[1,0].set_ylabel("pct.", rotation=0, labelpad=15, ha='right')

        # Række 2: Inflation pi (ganget med 100)
        ax[2,0].plot(jnp.arange(T_u), GIRF.pi_u * 100, color='red')
        if do_bands: ax[2,0].fill_between(jnp.arange(T_u), GIRF.pi_u_lower * 100, GIRF.pi_u_upper * 100, color='red', alpha=0.15)
        
        ax[2,1].plot(jnp.arange(T_z), GIRF.pi_z * 100, color='red')
        if do_bands: ax[2,1].fill_between(jnp.arange(T_z), GIRF.pi_z_lower * 100, GIRF.pi_z_upper * 100, color='red', alpha=0.15)
        
        ax[2,2].plot(jnp.arange(T_Gamma), GIRF.pi_ln_Gamma * 100, color='red')
        if do_bands: ax[2,2].fill_between(jnp.arange(T_Gamma), GIRF.pi_ln_Gamma_lower * 100, GIRF.pi_ln_Gamma_upper * 100, color='red', alpha=0.15)
        
        for i in range(3): 
            ax[2,i].set_title(r'Inflation: $\pi_t$')
        ax[2,0].set_ylabel("p.p.", rotation=0, labelpad=15, ha='right')

        # Række 3: Nominal Interest Rate i (ganget med 100)
        ax[3,0].plot(jnp.arange(T_u), GIRF.i_u * 100, color='red')
        if do_bands: ax[3,0].fill_between(jnp.arange(T_u), GIRF.i_u_lower * 100, GIRF.i_u_upper * 100, color='red', alpha=0.15)
        
        ax[3,1].plot(jnp.arange(T_z), GIRF.i_z * 100, color='red')
        if do_bands: ax[3,1].fill_between(jnp.arange(T_z), GIRF.i_z_lower * 100, GIRF.i_z_upper * 100, color='red', alpha=0.15)
        
        ax[3,2].plot(jnp.arange(T_Gamma), GIRF.i_ln_Gamma * 100, color='red')
        if do_bands: ax[3,2].fill_between(jnp.arange(T_Gamma), GIRF.i_ln_Gamma_lower * 100, GIRF.i_ln_Gamma_upper * 100, color='red', alpha=0.15)
        
        for i in range(3): 
            ax[3,i].set_title(r'Nominal Interest Rate: $i_t$')
        ax[3,0].set_ylabel("p.p.", rotation=0, labelpad=15, ha='right')

        # Række 4: ZLB frequency (Beholdes som de er, da de sandsynligvis allerede er andele/procenter)
        ax[4,0].plot(jnp.arange(T_u), 100*GIRF.u_zlb_share, color='red')
        ax[4,1].plot(jnp.arange(T_z), 100*GIRF.z_zlb_share, color='red')
        ax[4,2].plot(jnp.arange(T_Gamma), 100*GIRF.ln_Gamma_zlb_share, color='red')

        for i in range(3): 
            ax[4,i].set_title(r'Frequency of ZLB')
        ax[4,0].set_ylabel("pct.", rotation=0, labelpad=15, ha='right')

        if plot_extra:
            ax[1,0].plot(jnp.arange(T_u), GIRF.Y_u_extra * 100, label='DEQN w/o ZLB', color='orange', ls='--')
            if do_bands: ax[1,0].fill_between(jnp.arange(T_u), GIRF.Y_u_extra_lower * 100, GIRF.Y_u_extra_upper * 100, color='orange', alpha=0.15)
            
            ax[1,1].plot(jnp.arange(T_z), GIRF.Y_z_extra * 100, color='orange', ls='--')
            if do_bands: ax[1,1].fill_between(jnp.arange(T_z), GIRF.Y_z_extra_lower * 100, GIRF.Y_z_extra_upper * 100, color='orange', alpha=0.15)
            
            ax[1,2].plot(jnp.arange(T_Gamma), GIRF.Y_ln_Gamma_extra * 100, color='orange', ls='--')
            if do_bands: ax[1,2].fill_between(jnp.arange(T_Gamma), GIRF.Y_ln_Gamma_extra_lower * 100, GIRF.Y_ln_Gamma_extra_upper * 100, color='orange', alpha=0.15)
            
            ax[2,0].plot(jnp.arange(T_u), GIRF.pi_u_extra * 100, color='orange', ls='--')
            if do_bands: ax[2,0].fill_between(jnp.arange(T_u), GIRF.pi_extra_u_lower * 100, GIRF.pi_extra_u_upper * 100, color='orange', alpha=0.15)
            
            ax[2,1].plot(jnp.arange(T_z), GIRF.pi_z_extra * 100, color='orange', ls='--')
            if do_bands: ax[2,1].fill_between(jnp.arange(T_z), GIRF.pi_extra_z_lower * 100, GIRF.pi_extra_z_upper * 100, color='orange', alpha=0.15)
            
            ax[2,2].plot(jnp.arange(T_Gamma), GIRF.pi_ln_Gamma_extra * 100, color='orange', ls='--')
            if do_bands: ax[2,2].fill_between(jnp.arange(T_Gamma), GIRF.pi_extra_ln_Gamma_lower * 100, GIRF.pi_extra_ln_Gamma_upper * 100, color='orange', alpha=0.15)

            ax[3,0].plot(jnp.arange(T_u), GIRF.i_extra_u * 100, color='orange', ls='--')
            if do_bands: ax[3,0].fill_between(jnp.arange(T_u), GIRF.i_extra_u_lower * 100, GIRF.i_extra_u_upper * 100, color='orange', alpha=0.15)
            
            ax[3,1].plot(jnp.arange(T_z), GIRF.i_extra_z * 100, color='orange', ls='--')
            if do_bands: ax[3,1].fill_between(jnp.arange(T_z), GIRF.i_extra_z_lower * 100, GIRF.i_extra_z_upper * 100, color='orange', alpha=0.15)
            
            ax[3,2].plot(jnp.arange(T_Gamma), GIRF.i_extra_ln_Gamma * 100, color='orange', ls='--')
            if do_bands: ax[3,2].fill_between(jnp.arange(T_Gamma), GIRF.i_extra_ln_Gamma_lower * 100, GIRF.i_extra_ln_Gamma_upper * 100, color='orange', alpha=0.15)

        # Lineære IRF'er (ganget med 100)
        ax[1,0].plot(jnp.arange(T_u), IRF.Y_u_lin * 100, label='linear', color='green', ls='dotted', marker='.', ms=8)
        ax[1,1].plot(jnp.arange(T_z), IRF.Y_z_lin * 100, color='green', ls='dotted', marker='.', ms=8)
        ax[1,2].plot(jnp.arange(T_Gamma), IRF.Y_ln_Gamma_lin * 100, color='green', ls='dotted', marker='.', ms=8)

        ax[2,0].plot(jnp.arange(T_u), IRF.pi_u_lin * 100, color='green', ls='dotted', marker='.', ms=8)
        ax[2,1].plot(jnp.arange(T_z), IRF.pi_z_lin * 100, color='green', ls='dotted', marker='.', ms=8)
        ax[2,2].plot(jnp.arange(T_Gamma), IRF.pi_ln_Gamma_lin * 100, color='green', ls='dotted', marker='.', ms=8)

        ax[3,0].plot(jnp.arange(T_u), IRF.i_u_lin * 100, color='green', ls='dotted', marker='.', ms=8)
        ax[3,1].plot(jnp.arange(T_z), IRF.i_z_lin * 100, color='green', ls='dotted', marker='.', ms=8)
        ax[3,2].plot(jnp.arange(T_Gamma), IRF.i_ln_Gamma_lin * 100, color='green', ls='dotted', marker='.', ms=8)

        # OccBin IRF'er (ganget med 100)
        if plot_OccBin:
            ax[1,0].plot(jnp.arange(T_u), GIRF.Y_u_OccBin * 100, label='OccBin', ls ='--', color='purple', marker='D', ms=4)
            if do_bands: ax[1,0].fill_between(jnp.arange(T_u), GIRF.Y_u_OccBin_lower * 100, GIRF.Y_u_OccBin_upper * 100, color='purple', alpha=0.15, label='OccBin: 95 pct.')
            
            ax[1,1].plot(jnp.arange(T_z), GIRF.Y_z_OccBin * 100, ls='--', color='purple', marker='D', ms=4)
            if do_bands: ax[1,1].fill_between(jnp.arange(T_z), GIRF.Y_z_OccBin_lower * 100, GIRF.Y_z_OccBin_upper * 100, color='purple', alpha=0.15)
            
            ax[1,2].plot(jnp.arange(T_Gamma), GIRF.Y_Gamma_OccBin * 100, ls='--', color='purple', marker='D', ms=4)
            if do_bands: ax[1,2].fill_between(jnp.arange(T_Gamma), GIRF.Y_Gamma_OccBin_lower * 100, GIRF.Y_Gamma_OccBin_upper * 100, color='purple', alpha=0.15)

            ax[2,0].plot(jnp.arange(T_u), GIRF.pi_u_OccBin * 100, ls ='--', color='purple', marker='D', ms=4)
            if do_bands: ax[2,0].fill_between(jnp.arange(T_u), GIRF.pi_u_OccBin_lower * 100, GIRF.pi_u_OccBin_upper * 100, color='purple', alpha=0.15)
            
            ax[2,1].plot(jnp.arange(T_z), GIRF.pi_z_OccBin * 100, ls ='--', color='purple', marker='D', ms=4)
            if do_bands: ax[2,1].fill_between(jnp.arange(T_z), GIRF.pi_z_OccBin_lower * 100, GIRF.pi_z_OccBin_upper * 100, color='purple', alpha=0.15)
            
            ax[2,2].plot(jnp.arange(T_Gamma), GIRF.pi_Gamma_OccBin * 100, ls ='--', color='purple', marker='D', ms=4)
            if do_bands: ax[2,2].fill_between(jnp.arange(T_Gamma), GIRF.pi_Gamma_OccBin_lower * 100, GIRF.pi_Gamma_OccBin_upper * 100, color='purple', alpha=0.15)

            ax[3,0].plot(jnp.arange(T_u), GIRF.i_u_OccBin * 100, ls ='--', color='purple', marker='D', ms=4)
            if do_bands: ax[3,0].fill_between(jnp.arange(T_u), GIRF.i_u_OccBin_lower * 100, GIRF.i_u_OccBin_upper * 100, color='purple', alpha=0.15)
            
            ax[3,1].plot(jnp.arange(T_z), GIRF.i_z_OccBin * 100, ls ='--', color='purple', marker='D', ms=4)
            if do_bands: ax[3,1].fill_between(jnp.arange(T_z), GIRF.i_z_OccBin_lower * 100, GIRF.i_z_OccBin_upper * 100, color='purple', alpha=0.15)
            
            ax[3,2].plot(jnp.arange(T_Gamma), GIRF.i_Gamma_OccBin * 100, ls ='--', color='purple', marker='D', ms=4)
            if do_bands: ax[3,2].fill_between(jnp.arange(T_Gamma), GIRF.i_Gamma_OccBin_lower * 100, GIRF.i_Gamma_OccBin_upper * 100, color='purple', alpha=0.15)

            ax[4,0].plot(jnp.arange(T_u), 100*GIRF.u_zlb_share_OccBin, ls ='--', color='purple', marker='D', ms=4)
            ax[4,1].plot(jnp.arange(T_z), 100*GIRF.z_zlb_share_OccBin, ls ='--', color='purple', marker='D', ms=4)
            ax[4,2].plot(jnp.arange(T_Gamma), 100*GIRF.ln_Gamma_zlb_share_OccBin, ls ='--', color='purple', marker='D', ms=4)
            for i in range(3):
                ax[4,i].set_ylim(0.00, 100*jnp.max(jnp.stack([GIRF.u_zlb_share_OccBin, GIRF.z_zlb_share_OccBin, GIRF.ln_Gamma_zlb_share_OccBin,
                                                        GIRF.u_zlb_share, GIRF.z_zlb_share, GIRF.ln_Gamma_zlb_share]))+5)

        # Loop til xticks og rotation
        for row in range(5):

            ticks_u, labels_u = get_quarter_labels(T_u, label_step)
            ax[row, 0].set_xticks(ticks_u)
            ax[row, 0].set_xticklabels(labels_u, rotation=45) # Drejet 45 grader
            
            ticks_z, labels_z = get_quarter_labels(T_z, label_step)
            ax[row, 1].set_xticks(ticks_z)
            ax[row, 1].set_xticklabels(labels_z, rotation=45) # Drejet 45 grader
            
            ticks_Gamma, labels_Gamma = get_quarter_labels(T_Gamma, label_step)
            ax[row, 2].set_xticks(ticks_Gamma)
            ax[row, 2].set_xticklabels(labels_Gamma, rotation=45) # Drejet 45 grader

        f.tight_layout(rect=[0, 0.08, 1, 1])

        f.legend(loc='lower center', 
                bbox_to_anchor=(0.5, 0.05), 
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

def implied_Rotemberg_kappa(alpha, beta, epsilon, theta_Calvo=0.75):

    num = epsilon*(1-theta_Calvo)*(1-beta*theta_Calvo)*(1-alpha)
    denom = (epsilon-1)*theta_Calvo*(1-alpha+alpha*epsilon)

    return num/denom

def compute_sunspot_DSS(par):

    alpha = par["alpha"]
    beta = par["beta"]
    sigma = par["sigma"]
    varphi = par["varphi"]
    epsilon = par["epsilon"]
    kappa = par["kappa"]
    ZLB = par["ZLB"]

    mu = epsilon/(epsilon-1)

    exponent = (1-alpha)/(varphi+alpha+sigma-alpha*sigma)

    first_frac = (beta*(1+ZLB)*(beta*(1+ZLB)-1)*(1-beta)*(1-alpha))/kappa
    second_frac = (1-alpha)/mu

    Y = (first_frac+second_frac)**exponent
    pi = beta*(1+ZLB)-1
    i = ZLB

    return Y, pi, i