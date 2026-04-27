
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

    Y = nnx.softplus(z1 + 0.5) # [0, infty]
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

    out_lin = states @ linear["P"].T
    # out_lin_ZLB = states @ linear["P_ZLB"].T

    Y_lin = out_lin[..., 0]
    pi_lin = out_lin[..., 1]

    # Y_lin_ZLB = out_lin_ZLB[..., 0]
    # pi_lin_ZLB = out_lin_ZLB[..., 1]

    shadow_taylor = compute_shadow_taylor_rule(par, linear["P"], states)
    ZLB_dummy = shadow_taylor < par["ZLB"] #compute_ZLB_dummy(par, linear["P"], states)
    #time_dummies = compute_time_dummies(linear, states) # (N, gh_n, max_expected_ZLB)
    
    # input = jnp.concatenate([
    #   states, Y_OccBin[..., None], pi_OccBin[..., None],
    #   out_lin[..., 0, None], out_lin[..., 1, None],
    #   ZLB_dummy[..., None], shadow_taylor[..., None],
    #   time_dummies, Y_lin_ZLB[..., None], pi_lin_ZLB[..., None]
    # ], axis=-1)

    #input = jnp.concatenate([states, Y_OccBin[..., None], pi_OccBin[..., None], ZLB_dummy[..., None]], axis=-1)

    # append to input

    # 2. call nn
    states = jnp.concatenate(
      [
        states,                                               # state-vector
        Y_lin[..., None], pi_lin[..., None],                  # linear solution in non-ZLB regime
        ZLB_dummy[..., None], shadow_taylor[..., None],       # shadow taylor rule and ZLB dummy for non-ZLB regime
        #Y_lin_ZLB[..., None], pi_lin_ZLB[..., None],           # linear solution in ZLB regime
        #Y_OccBin[..., None], pi_OccBin[..., None],
        # time_dummies
      ], axis = -1
    )

    Ninputs = states.shape[-1]

    input = states.reshape(-1, Ninputs)
    out_nn = nn(input) # (N, 2) or (N * gh_n_combined, 2)

    # 3. unpack output
    Y_nn = out_nn[:, 0] #+ par["Y_DSS"] 
    pi_nn = out_nn[:, 1] #+ par["pi_DSS"]
    #i_nn = out_nn[:, 2]

    # 3. expand to (Nparallel, gh_n_combined) if quad
    if len(states.shape) == 2:
      
      Y = Y_nn #+ Y_OccBin #+ Y_known
      pi = pi_nn #+ pi_OccBin #+ pi_known
      #i = i_nn

      return Y, pi#, i

    else:

      Y = Y_nn.reshape(N, -1)# + Y_OccBin # + out_lin[..., 0] + par["Y_DSS"]Y_OccBinY_known +
      pi = pi_nn.reshape(N, -1) #+ pi_OccBin # + out_lin[..., 1] + par["pi_DSS"]pi_OccBin + pi_known +
      #i = i_nn.reshape(N, -1)

      return Y, pi #, i

def compute_ZLB_dummy(par, P, states):

    # states: (Nparallel, 3) or (Nparallel, gh_n_combined, 3)

    beta = par["beta"]
    phi = par["phi"]
    ZLB = par["ZLB"]

    P_pi = P[1, :, None] # (3, 1)

    lin_pi = (states @ P_pi)[..., 0]
    u = states[..., 0]

    ZLB_dummy = 1/beta - 1 + phi*lin_pi+u <= ZLB

    return ZLB_dummy.astype(jnp.float32)

def compute_shadow_taylor_rule(par, P, states):

    # states: (Nparallel, 3) or (Nparallel, gh_n_combined, 3)

    X_lin = states @ P.T
    Y_lin = X_lin[..., 0]
    pi_lin = X_lin[..., 1]
    u = states[..., 0]
    z = states[..., 1]
    ln_Gamma = states[..., 2]

    i = taylor_rule(par, Y_lin, pi_lin, u, z, ln_Gamma, jnp.zeros(1), jnp.zeros(1), jnp.ones(1), -100)

    return i

def compute_time_dummies(linear, states):

  max_expected_ZLB = linear["max_expected_ZLB"]
  time_to_ZLB_slack_interp = linear["time_to_ZLB_slack_interp"]

  time_cont = time_to_ZLB_slack_interp(states) # (N, ) or (N, gh_n)

  return nnx.one_hot(time_cont.astype(int), 43)