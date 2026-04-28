import numpy as np
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.core import FrozenDict
import math

from aux_ import draw_shocks, draw_states_directly, next_states_quad, next_states
from model_funcs import euler_error, NKPC_error, taylor_rule
from neural_nets import eval_nn

############
# TRAINING #
############

def loss(nn, par, train, linear, dtype, states, ZLB, gh_x, gh_w):

    # 1. quad nodes and weights
    N = states.shape[0]
    eps_z = gh_x[:,1]
    eps_Gamma = gh_x[:,2]

    # 2. unpack state space: shape (N,)
    u = states[:, 0]
    z = states[:, 1]
    ln_Gamma = states[:, 2]

    # 3. compute next-period states: shape (N, gh_n)
    states_p = next_states_quad(par, dtype, states, gh_x)

    # 4. call nn
    Y, pi = eval_nn(par, train, linear, nn, states, N)

    # 5. call nn for states in next-period with quad
    Y_p, pi_p = eval_nn(par, train, linear, nn, states_p, N)

    # 6. evaluate equilibrium equations
    i = taylor_rule(par, Y, pi, u, z, ln_Gamma, eps_z, eps_Gamma, ZLB, gh_w) 
    ee = euler_error(par, Y, Y_p, pi, pi_p, i, u, z, eps_z, gh_w)
    nkpce = NKPC_error(par, Y, Y_p, pi, pi_p, i, u, ln_Gamma, gh_w)

    return ee, nkpce, Y, Y_p, pi, pi_p, i

def phase_training_loop(
    par, train, linear, dtype, nn, opt, # model
    phase_episodes, ZLB, sigma_quad, sigma_sim, N, zero_var, # phase
    print_freq, states_test, # training
    key, info # each episode
):

    par = FrozenDict(par)
    train = FrozenDict(train)
    linear = FrozenDict(linear)

    # unpack
    gh_n_per_shock = train["gh_n_per_shock"]
    gh_x, gh_w = construct_gh_nodes(dtype, gh_n_per_shock, sigma_quad, zero_var=zero_var)

    # define jitted train step function
    @nnx.jit
    def train_step(nn, opt, key):

        key, subkey = jax.random.split(key)
        states = draw_states_directly(subkey, par, dtype, N, sigma_sim["sigma_eps_u"], sigma_sim["sigma_eps_z"], sigma_sim["sigma_eps_Gamma"])

        def loss_fn(m):
            ee, nkpce, _, _, _, _, _ = loss(m, par, train, linear, dtype, states, ZLB, gh_x, gh_w)

            ee_mse = jnp.mean(optax.losses.squared_error(ee))
            nkpce_mse = jnp.mean(optax.losses.squared_error(nkpce))

            return ee_mse + nkpce_mse
    
        loss_train, grad = nnx.value_and_grad(loss_fn)(nn)

        opt.update(grad)

        return key, loss_train

    @nnx.jit
    def test_loss_fn(nn):
        ee, nkpce, Y, Y_p, pi, pi_p, i = loss(nn, par, train, linear, dtype, states_test, ZLB, gh_x, gh_w)
        ee_mse = jnp.mean(optax.losses.squared_error(ee))
        nkpce_mse = jnp.mean(optax.losses.squared_error(nkpce))

        loss_value = ee_mse + nkpce_mse

        return loss_value, ee, nkpce, Y, Y_p, pi, pi_p, i

    best_loss = jnp.inf
    for k in range(phase_episodes):

        # c. print
        if (k % print_freq == 0) or (k == 0):

            # compute evaluation loss
            loss_test, ee, nkpce, Y, Y_p, pi, pi_p, i = test_loss_fn(nn)
            if loss_test < best_loss: best_loss = loss_test

            # compute metrics
            Y_SSS, pi_SSS = eval_nn(par, train, linear, nn, jnp.zeros((1, 3)), 1)
            Y_SSS, pi_SSS = Y_SSS[0], pi_SSS[0]
            ee_mae = jnp.mean(jnp.abs(ee))
            nkpce_mae = jnp.mean(jnp.abs(nkpce))
            ZLB_share = jnp.mean(taylor_rule(par, Y, pi, states_test[:, 0], states_test[:, 1], states_test[:, 2], gh_x[:, 1], gh_x[:, 2], 0.00, gh_w, return_shadow=True) < 0)

            # print
            print(f'Episode {k}:\tLoss = {loss_test:.8f}\tBest Loss = {best_loss:.8f}\t\tSSS: Y = {Y_SSS:.3f}, pi = {pi_SSS:.3f}\t\tee = {ee_mae:.8f}\tnkpce = {nkpce_mae:.8f}\tZLB share = {ZLB_share:.3f}')
        
        key, loss_train = train_step(nn, opt, key)

        info["train_losses"].append(loss_train.item())
        info["test_losses"].append(loss_test.item())
        #info["best_test_losses"].append(best_loss.item())
        info["ee_test_losses"].append(ee_mae.item())
        info["nkpce_test_losses"].append(nkpce_mae.item())

    return key, info

def train_nn(
    model, episodes, sigma_sim, sigma_quad, # model, training generally
    ZLB_list, lr_list, N_list, zero_var_list, # phase
    print_freq=50 # aux
):

    test_key = jax.random.PRNGKey(42)
    train_key = jax.random.PRNGKey(43)

    # 1. unpack
    par = model.par
    train = model.train
    linear = model.linear
    nn = model.nn
    opt = model.opt
    dtype = model.dtype

    # 4. initialize lists for storing losses
    info = {
        "train_losses" : [],
        "test_losses" : [],
        "best_test_losses" : [],
        "ee_test_losses" : [],
        "nkpce_test_losses" : []
    }
    
    # 5. train
    for phase in range(len(episodes)):

        zero_var = zero_var_list[phase]
        states_test = draw_states_directly(test_key, par, dtype, train["Nparallel_test"], sigma_sim["sigma_eps_u"], sigma_sim["sigma_eps_z"], sigma_sim["sigma_eps_Gamma"])
        states_test = jax.lax.stop_gradient(states_test)

        # a. phase-specific: #episodes, ZLB, lr, N
        phase_episodes = episodes[phase]
        ZLB = ZLB_list[phase]
        opt.opt_state.hyperparams['learning_rate'] = lr_list[phase]
        #opt.opt_state.hyperparams['clipping_value'] =1.0
        N = N_list[phase]

        print(f'{50*"-"} PHASE {phase}:  {50*"-"}')
        print(f'###### sigma_sim = {sigma_sim["sigma_eps_u"]}, sigma_quad = {sigma_quad["sigma_eps_u"]}, ZLB = {ZLB}, lr = {lr_list[phase]}, N = {N} ######\n')

        train_key, info = phase_training_loop(
            par, train, linear, dtype, nn, opt,
            phase_episodes, ZLB, sigma_quad, sigma_sim, N, zero_var,
            print_freq, states_test,
            train_key, info)

        train[f"nn_phase{phase}"] = nn
        model.info = info

    model.nn, model.opt = nn, opt

############
# SIMULATE #
############

def simulate(model, T, shocks, extra_nn=None):

    par = model.par
    train = model.train
    linear = model.linear
    nn = model.nn
    dtype = model.dtype

    key = jax.random.PRNGKey(42)

    Nstates = par["Nstates"]
    P = linear["P"]

    Y_SSS, pi_SSS = eval_nn(par, train, linear, nn, jnp.zeros((1,3)),1)

    # start at SSS
    states = jnp.zeros((T, Nstates))

    states_i = jnp.zeros((1, Nstates))

    for t in range(T):
        states = states.at[t].set(states_i[0])
        key, subkey = jax.random.split(key)
        eps = draw_shocks(subkey, dtype, 1, shocks[0], shocks[1], shocks[2])
        states_i = next_states(par, states_i, eps)

    # nn
    Y, pi = eval_nn(par, train, linear, nn, states, T)


    # linear
    out_lin = states @ linear["P"].T
    Y_lin, pi_lin = out_lin[:, 0], out_lin[:, 1]
    Y_interp_OccBin, pi_interp_OccBin = linear["Y_interp_OccBin"], linear["pi_interp_OccBin"]
    Y_OccBin = Y_interp_OccBin(states)
    pi_OccBin = pi_interp_OccBin(states)

    sim = SimpleNamespace()

    sim.Y = Y
    sim.pi = pi
    sim.i = taylor_rule(par, Y, pi, states[:, 0], states[:, 2], 0.0, 0.0, 0.0, 0.0, 0.00)

    sim.Y_lin = Y_lin + par["Y_DSS"]
    sim.pi_lin = pi_lin
    sim.i_lin = taylor_rule(par, Y_lin + par["Y_DSS"], pi_lin, states[:, 0], states[:, 2], 0.0, 0.0, 0.0,-100, 0.0)

    sim.Y_OccBin = Y_OccBin + par["Y_DSS"]
    sim.pi_OccBin = pi_OccBin
    sim.i_OccBin = taylor_rule(par, Y_OccBin + par["Y_DSS"], pi_OccBin, states[:, 0], states[:, 2], 0.0, 0.0, 0.0, 0.0, 0.0)

    if extra_nn is not None:
        Y_extra, pi_extra = eval_nn(par, train, linear, extra_nn, states, T)
        sim.Y_extra = Y_extra
        sim.pi_extra = pi_extra
        sim.i_extra = taylor_rule(par, Y_extra, pi_extra, states[:, 0], states[:, 2], 0.0, 0.0, 0.0,-100, 0.0)
    
    sim.states = states

    model.sim = sim

def construct_gh_nodes(dtype, gh_n_per_shock, sigma_quad, zero_var = None):

    sigma_eps_u = sigma_quad["sigma_eps_u"]
    sigma_eps_z = sigma_quad["sigma_eps_z"]
    sigma_eps_Gamma = sigma_quad["sigma_eps_Gamma"]

    if zero_var == 'u': sigma_eps_u = 0.0
    if zero_var == 'z': sigma_eps_z = 0.0
    if zero_var == 'ln_Gamma': sigma_eps_Gamma = 0.0

    # 1. get raw nodes, weights
    x, w = gauss_hermite(gh_n_per_shock)

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

def gauss_hermite(n):

    # a. calculations
    i = np.arange(1,n)
    a = np.sqrt(i/2)
    CM = np.diag(a,1) + np.diag(a,-1)
    L,V = np.linalg.eig(CM)
    I = L.argsort()
    V = V[:,I].T

    # b. nodes and weights
    x = L[I]
    w = np.sqrt(math.pi)*V[:,0]**2

    return x,w