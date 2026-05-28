
import jax
import jax.numpy as jnp
from flax import nnx

from model_funcs import taylor_rule

class Policy(nnx.Module):

  # layers should be flax compatible list: syntax for parameter updates in list
  layers: list[nnx.Linear]
  
  def __init__(self, din: int, dout: int, neurons: list, rngs: nnx.Rngs, dtype, device):

    # 1. initialize
    layers = []

    # 2. 1st layer
    layers.append(nnx.Linear(din, neurons[0], rngs=rngs, dtype=dtype, param_dtype=dtype))
    
    # 3. hidden layers
    for layer in range(len(neurons)-1):
      
      layers.append(nnx.Linear(neurons[layer], neurons[layer+1], rngs=rngs))

    # 4. output layer
    layers.append(nnx.Linear(neurons[-1], dout, rngs=rngs))

    # 5. assign to neural network
    self.layers = nnx.List(layers)

  def __call__(self, x: jax.Array):

    # 1. 1st and hiden layers
    for layer in self.layers[:-1]:

      x = nnx.gelu(layer(x)) # ReLU activation for input and all hidden layers

    # 2. output layer
    z = self.layers[-1](x) 

    z1, z2 = jnp.split(z, 2, axis=-1)

    Y = nnx.softplus(z1+0.5) # [0, infty]
    pi = nnx.tanh(z2)

    out = jnp.concatenate([Y, pi], axis=-1)
    
    return out

def eval_nn(par, train, linear, nn, states, N):

    #Nstates = states.shape[-1] + 3 #+ 8 + int(linear["max_expected_ZLB"]) + 1 #int(train["do_ZLB_dummy"])# + 2*int(train["do_shadow_taylor_rule"]) #+ 12

    # Y_inp = linear["Y_interp_OccBin"]
    # pi_inp = linear["pi_interp_OccBin"]

    # out_lin = states @ linear["P"].T

    # if train["do_shadow_taylor_rule"]:
    #   shadow_taylor = compute_shadow_taylor_rule(par, linear["P"], states)
    #   shadow_taylor_hinged = jnp.minimum(shadow_taylor, 0.0)
    #   states = jnp.concatenate([states, shadow_taylor[..., None], shadow_taylor_hinged[..., None]],  axis=-1)

    # if train["do_ZLB_dummy"]:
    #   ZLB_dummy = shadow_taylor < par["ZLB"] #compute_ZLB_dummy(par, linear["P"], states)
    #   states = jnp.concatenate([states, ZLB_dummy[..., None]],  axis=-1)

    # u_dummy_pos = states[..., 0] > 0.0
    # z_dummy_pos = states[..., 1] > 0.0
    # Gamma_dummy_pos = states[..., 2] > 0.0

    # u_dummy_neg = states[..., 0] < 0.0
    # z_dummy_neg = states[..., 1] < 0.0
    # Gamma_dummy_neg = states[..., 2] < 0.0

    # u_u_dummy_pos = states[..., 0] * u_dummy_pos
    # z_z_dummy_pos = states[..., 1] * z_dummy_pos
    # Gamma_Gamma_dummy_pos = states[..., 2] * Gamma_dummy_pos

    # u_u_dummy_neg = states[..., 0] * u_dummy_neg
    # z_z_dummy_neg = states[..., 1] * z_dummy_neg
    # Gamma_Gamma_dummy_neg = states[..., 2] * Gamma_dummy_neg

    # states = jnp.concatenate(
    #   [
    #     states,
    #     u_dummy_pos[..., None], z_dummy_pos[..., None], Gamma_dummy_pos[..., None],
    #     u_dummy_neg[..., None], z_dummy_neg[..., None], Gamma_dummy_neg[..., None],
    #     u_u_dummy_pos[..., None], z_z_dummy_pos[..., None], Gamma_Gamma_dummy_pos[..., None],
    #     u_u_dummy_neg[..., None], z_z_dummy_neg[..., None], Gamma_Gamma_dummy_neg[..., None],
    #   ], axis = -1
    # )

    # 1. flatten to 2D
    #input = states.reshape(-1, Nstates) # (N, 3) or (N * gh_n, 3)

    # 4. compute OccBin
    # Y_OccBin = Y_inp(states) # # (N, 3) or (N, gh*n, 3) 
    # pi_OccBin = pi_inp(states)

    # Y_lin, pi_lin = eval_lin_nn(par, linear, states, return_dev=False)
    # i_lin = taylor_rule(par, Y_lin, pi_lin, states[..., 0], states[..., 1], states[..., 2], 0.00, 0.00, -100, 0.00, return_shadow=True)
    # ZLB_dummy = i_lin <= par["ZLB"]
    # Y_lin, pi_lin = eval_lin_nn(par, linear, states, return_dev=True)


    # #2. call nn
    # states = jnp.concatenate(
    #   [
    #     states,                                               # state-vector
    #     Y_lin[..., None], pi_lin[..., None],                  # linear solution in non-ZLB regime
    #     ZLB_dummy[..., None], i_lin[..., None],               # shadow taylor rule and ZLB dummy for non-ZLB regime
    #     #Y_lin_ZLB[..., None], pi_lin_ZLB[..., None],         # linear solution in ZLB regime
    #     #Y_OccBin[..., None], pi_OccBin[..., None],
    #     # time_dummies
    #   ], axis = -1
    # )

    Ninputs = states.shape[-1]

    input = states.reshape(-1, Ninputs)
    out_nn = nn(input) # (N, 2) or (N * gh_n_combined, 2)

    # 3. unpack output
    Y_nn = out_nn[:, 0]
    pi_nn = out_nn[:, 1]

    # 3. expand to (Nparallel, gh_n_combined) if quad
    if len(states.shape) == 2:
      
      Y = Y_nn
      pi = pi_nn

      return Y, pi

    else:

      Y = Y_nn.reshape(N, -1)
      pi = pi_nn.reshape(N, -1)

      return Y, pi

def compute_SSS(par, train, linear, nn, states, N):

  states = jnp.zeros((1,3))
  Y_SSS, pi_SSS = eval_nn(par, train, linear, nn, states, 1)

  return Y_SSS.item(), pi_SSS.item()

def eval_nn_dev(par, train, linear, nn, states, N, return_i=False, ZLB=-1):

  Y_raw, pi_raw = eval_nn(par, train, linear, nn, states, N)
  
  Y_SSS, pi_SSS = compute_SSS(par, train, linear, nn, states, N)

  Y_dev = (Y_raw-Y_SSS)/Y_SSS
  pi_dev = pi_raw-pi_SSS

  if return_i:
    i = taylor_rule(par, Y_raw, pi_raw, states[..., 0], states[..., 1], states[..., 2], 0.0, 0.0, ZLB, 0.00)
    i_SSS = taylor_rule(par, Y_SSS, pi_SSS, 0.00, 0.00, 0.00, 0.0, 0.0, par["ZLB"], 0.00)
    i_dev = i - i_SSS

    return Y_dev, pi_dev, i_dev

  else:
    return Y_dev, pi_dev

def compute_time_dummies(linear, states):

  max_expected_ZLB = linear["max_expected_ZLB"]
  time_to_ZLB_slack_interp = linear["time_to_ZLB_slack_interp"]

  time_cont = time_to_ZLB_slack_interp(states) # (N, ) or (N, gh_n)

  return nnx.one_hot(time_cont.astype(int), 43)