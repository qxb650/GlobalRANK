
import jax.numpy as jnp

# def compute_linearized_IRF(model, shock):

#     par = model.par
#     train = model.train

#     linear = SimpleNamespace()

#     kappa_tilde = par['kappa']*(par['sigma']+par['varphi'])

#     A = np.array([
#         [1.0, par['sigma']],
#         [0.0, par['beta']]
#         ])

#     B = np.array([
#         [1.0, par['sigma']*par['phi']],
#         [-kappa_tilde, 1.0]
#     ])

#     C = np.array([
#         [par['sigma']],
#         [0.0]
#     ])

#     P = np.linalg.inv( A*par['rho_i'] - B ) @ C

#     linear.A, linear.B, linear.C, linear.P = A, B, C, P

#     IRF_ti = np.zeros((train['T'], train["Nstates"])) + np.nan

#     u_path = np.ones(train['T'])*shock
#     u_path = u_path*par['rho_i']**(np.arange(train['T']))

#     Y_dev = P[0]*u_path
#     pi_dev = P[1]*u_path

#     linear.Y_dev = Y_dev
#     linear.pi_dev = pi_dev

#     model.linear = linear

# #
# #
# #
# def train_SS(model, DSS_episodes=False, SSS_episodes=1000, tol=1e-10):

#     # 1. unpack and freeze dicts
#     par = FrozenDict(model.par)
#     DSS = FrozenDict(model.DSS)
#     train = FrozenDict(model.train)
#     device = model.device
#     dtype = model.dtype
#     nn = model.nn
#     opt = model.opt

#     # 2. unpack train specs
#     Nparallel = train["Nparallel"]
#     T = train["T"]
#     gh_nodes = train['gh_nodes']
#     gh_weights = train['gh_weights']

#     # 3. initialize zero shocks
#     zero_shocks = jnp.zeros((1,T), device=device, dtype=dtype)

#     # 4. DSS training
#     if DSS_episodes:

#         print('Beginning DSS training')

#         def train_step(nn, zero_shocks):

#             loss_fn = lambda nn: DSS_loss(nn, DSS, zero_shocks)
            
#             loss, grad = jax.value_and_grad(loss_fn)(nn)

#             return loss, grad

#         train_step_jit = jax.jit(train_step)
        
#         for k in range(DSS_episodes):

#             loss, grad = train_step_jit(nn, zero_shocks)

#             if k % 10 == 0: print(f'Episode {k}: Loss = {loss:.8f}')

#             opt.update(grad)

#             if loss < tol:
#                 print(f'Tolerance reached: breaking loop')
#                 break

#     # 5. SSS training
#     print('Beginning SSS training')

#     def train_step(nn, zero_shocks):

#         loss_fn = lambda nn: SSS_loss(nn, par, DSS, train, zero_shocks)
        
#         loss, grad = jax.value_and_grad(loss_fn)(nn)

#         return loss, grad

#     train_step_jit = jax.jit(train_step, static_argnums=())

#     for k in range(SSS_episodes):
        
#         loss, grad = train_step_jit(nn, zero_shocks)
        
#         if k % 10 == 0: print(f'Episode {k}: Loss = {loss:.8f}')
#         opt.update(grad)

#         if loss<tol:
#             print(f'Tolerance reached: breaking loop')
#             break
    
#     print(nn(jnp.zeros(T)))

# def SSS_loss(nn, par, DSS, train, zero_shocks):

#     gh_nodes = train["gh_nodes"]
#     gh_weights = train["gh_weights"]
#     Nparallel = train["Nparallel"]

#     # 1. call nn
#     out = nn(zero_shocks)
#     Y = out[:,0]
#     pi = out[:,1]

#     # 2. call next-period nn on quadrature nodes and unpack MP-shock
#     zero_shocks_quad = next_eps_hist_quad(zero_shocks, gh_nodes)
#     out_p = nn(zero_shocks_quad).reshape(1,-1,2)
#     Y_p = out_p[...,0]
#     pi_p = out_p[...,1]
#     zero_u_i = zero_shocks[:,-1]

#     # 3. compute equilibrium errors
#     Y_errors = euler_error(par, DSS, Y, Y_p, pi, pi_p, zero_u_i, gh_weights)
#     pi_errors = NKPC_error(par, DSS, pi, pi_p, Y, Y_p, zero_u_i, gh_weights)

#     # 4. compute mse
#     loss = jnp.mean(Y_errors**2) + 5*jnp.mean(pi_errors**2)
    
#     return loss

# def full_loss_fn(nn, par, train, states, eps_hist, return_errors=False):

#     gh_nodes = train["gh_nodes"]
#     gh_weights = train["gh_weights_combined"]
#     gh_n = train["gh_n"]
#     gh_n_combined = train["gh_n_combined"]
#     #Nparallel = train["Nparallel"]
#     #T = train["T"]

#     Nparallel = eps_hist.shape[0]
#     T = eps_hist.shape[1]

#     u = states[:, 0].reshape(-1,1,1)
#     z = states[:, 1].reshape(-1,1,1)
#     ln_Gamma = states[:, 2].reshape(-1,1,1)

#     # a. evaluate nn in MA-history + compute MP-shock -> state space in period t
#     #in_nn = eps_hist.reshape(Nparallel, 3 * T) # eps_hist.transpose(0, 2, 1).reshape(Nparallel, 3 * T)
#     #ZLB_dummy = u > 1 + par["ZLB"] - 1/par["beta"]
#     #in_nn = jnp.concat([ZLB_dummy.reshape(-1,1), in_nn], axis=1)
#     #out = nn(in_nn) # shape (Nparallel, 2)
#     #Y = out[:, 0].reshape(-1,1)
#     #pi = out[:, 1].reshape(-1,1)

#     # b. evaluate nn in MA-history with quadrature as latest shock, no need for MP-shock in next period -> state space in period t+1
#     eps_hist_quad = next_eps_hist_quad(eps_hist, gh_nodes) # .reshape(Nparallel * gh_n_combined, 3 * T) # next_eps_hist_quad(eps_hist, gh_nodes).transpose(0, 2, 1).reshape(Nparallel * gh_n_combined, 3 * T)
#     #ZLB_dummy_p = par["rho_i"] * u + gh_nodes[None, :, 0] > 1 + par["ZLB"] - 1/par["beta"]
#     #in_nn_p = jnp.concat([ZLB_dummy_p.reshape(-1,1), eps_hist_quad], axis=1)
#     #in_nn_p = eps_hist_quad
#     #out_p = nn(in_nn_p).reshape(Nparallel, gh_n_combined, 2)
#     #Y_p = out_p[...,0] # shape (Nparallel, gh_n_combined)
#     #pi_p = out_p[...,1] # shape (Nparallel, gh_n_combined)
#     z_p = par["rho_z"]*z + gh_nodes[None, :, 1] # shape (Nparallel, gh_n_combined)

#     Y, pi = eval_nn(nn, eps_hist, Nparallel)
#     Y_p, pi_p = eval_nn(nn, eps_hist_quad, Nparallel)

    
#     # c. evaluate equilibrium equations
#     Y_errors = euler_error(par, Y, Y_p, pi, pi_p, u, z, z_p, gh_weights)
#     pi_errors = NKPC_error(par, Y, Y_p, pi, pi_p, u, ln_Gamma, gh_weights)

#     # d. compute loss
#     loss = jnp.mean(Y_errors**2) + jnp.mean(pi_errors**2)
    
#     if return_errors == False:
#         return loss
    
#     else:
#         return jnp.mean(Y_errors), jnp.mean(pi_errors)

# # def compute_DSS(par, do_print=False):

# #     # unpack
# #     beta = par["beta"]
# #     Gamma = par["Gamma"]
# #     varphi = par["varphi"]
# #     sigma = par["sigma"]
# #     mu = par["mu"]
# #     phi = par["phi"]
# #     kappa = par["kappa"]

# #     # closed form
# #     pi = 0.0
# #     i = 1/beta - 1
# #     Y = (Gamma**((1+varphi)/(sigma+varphi)))/(mu**(1/(sigma+varphi)))

# #     # evaluate DSS equations
# #     taylor_error = i - i - phi*pi
# #     euler_error = inv_marg_util(par, beta*( (1+i)/(1+pi) )*marg_util(par,Y))/Y - 1
# #     NKPC_error = pi*(1+pi)-kappa*( (Y**(sigma+varphi))/(Gamma**(1+varphi))-1/mu) - (Y/Y)*((1+i)/(1+pi)) * pi*(1+pi)

# #     if do_print:
# #         print(f'Taylor rule error:\t\t{taylor_error:.4f}')
# #         print(f'Euler Equation error:\t\t{euler_error:.4f}')
# #         print(f'NKPC error:\t\t\t{NKPC_error:.4f}')

# #     return {'Y' : Y, 'pi' : pi, 'i':i}

# # def DSS_loss(nn, DSS, zero_shocks):

# #     # 1. call nn
# #     out = nn(zero_shocks)
# #     Y = out[:,0]
# #     pi = out[:,1]

# #     # 2. compute mse
# #     Y_errors = (Y - DSS.get("Y"))**2
# #     pi_errors = (pi - DSS.get("pi"))**2
# #     loss = jnp.mean(Y_errors) + jnp.mean(pi_errors)
    
# #     return loss

# def eval_nn(nn, states, Nparallel):

#     # eps_hist shape (Nparallel * gh_n_combined, T, 3)
#     #T = eps_hist.shape[1]

#     # 1. flatten to 2D
#     #in_nn = eps_hist.transpose(0,2,1).reshape(-1, 3 * T)
#     in_nn = states.reshape(-1, 3)

#     # 2. infer quad dim
#     #gh_n_combined = int(in_nn.shape[0]/Nparallel)
    
#     # 3. call nn
#     out_nn = nn(in_nn) # shape (Nparallel * gh_n_combined, 2)

#     # 4. unpack output
#     Y = out_nn[:, 0]
#     pi = out_nn[:, 1]

#     # 4. expand to (Nparallel, gh_n_combined) if quad
#     #if gh_n_combined > 1:
#     Y = out_nn[:, 0].reshape(Nparallel, -1) 
#     pi = out_nn[:, 1].reshape(Nparallel, -1)

#     return Y, pi

##########################
# IMPLIED LOSS FUNCTIONS #
##########################

def NKPC_implied_loss(par, Y, Y_p, pi, pi_p, u, ln_Gamma, weights):

    Y_p = jax.lax.stop_gradient(Y_p)
    pi_p = jax.lax.stop_gradient(pi_p)

    # Y is (Nparallel,)
    # Y_p,r_p is (Nparallel,weigths)
    beta = par["beta"]
    kappa = par["kappa"]
    mu = par["mu"]
    sigma = par["sigma"]
    varphi = par["varphi"]

    # only expand to grids cause only used in "today term"
    Gamma = jnp.exp(ln_Gamma)

    # taylor rule
    i = taylor_rule(par, pi, u)

    # NKPC terms
    frac = (Y**(sigma+varphi))/(Gamma**(1+varphi)) # (Y*((sigma+varphi)/(1+varphi)))/(Gamma**(1+varphi))
    today = kappa * (frac - 1/mu) # (Nparallel,)

    tomorrow = Y_p * pi_p * (1 + pi_p) #**2 # (Nparallel,weights)
    Etomorrow = beta*jnp.sum(weights[None, :] * tomorrow, axis=-1, keepdims=False)/Y # (Nparallel,)

    A = today + Etomorrow

    pi_implied = (-1+jnp.sqrt(1+4*A))/2

    return jnp.mean(optax.losses.log_cosh(pi-pi_implied))

def euler_implied_loss(par, Y, Y_p, pi, pi_p, u, z, eps_z, weights):

    Y_p = jax.lax.stop_gradient(Y_p)
    pi_p = jax.lax.stop_gradient(pi_p)

    # Y,pi is (Nparallel,1)
    # Y_p,pi_p is (Nparallel,weigths)
    beta = par["beta"]
    sigma = par["sigma"]
    rho_z = par["rho_z"]
    theta = par["theta"]

    # taylor rule
    i = taylor_rule(par, pi, u)

    # expected marginal utility next period
    C_p = Y_p - theta* pi_p**2 * Y_p
    MU_p = marg_util(par, C_p) * jnp.exp(eps_z[None, :]) * (1/(1+pi_p)) 
    EMU_p = beta * (1 + i) * jnp.exp((rho_z-1)*z) * jnp.sum(weights[None, :] * MU_p, axis=-1, keepdims=False)

    C_implied = inv_marg_util(par, EMU_p)

    return jnp.mean(optax.losses.log_cosh(C_implied - Y + theta*pi**2 * Y)) # EMU_p/(marg_util(par, Y)*Z) - 1

    # def smooth_max(x, floor, alpha=100.0):
    #     return floor + nnx.softplus(alpha*(x-floor))/alpha

# def gauss_hermite(dtype, n): ### FRA GEMINI
#     # 1. Skab indekser fra 1 til n-1
#     i = jnp.arange(1, n, dtype=dtype)
    
#     # 2. Beregn off-diagonale led for den symmetriske Jacobi-matrix
#     # For Hermite-polynomier er leddene sqrt(i / 2)
#     a = jnp.sqrt(i / 2.0)
    
#     # 3. Konstruer matricen manuelt for at sikre præcision
#     # Vi bruger 'eigh' på en tridiagonal matrix, hvilket er meget stabilt
#     L, V = jnp.linalg.eigh(jnp.diag(a, k=1) + jnp.diag(a, k=-1))
    
#     # 4. Sortering er kritisk for symmetri
#     idx = jnp.argsort(L)
#     x = L[idx]
#     V = V[:, idx]
    
#     # 5. Vægte: integralet af exp(-x^2) er sqrt(pi)
#     # Vi bruger kvadratet af det første element i hver egenvektor
#     w = jnp.sqrt(jnp.pi) * (V[0, :]**2)
    
#     # 6. Tving symmetri for den midterste node (ved ulige n)
#     # Numerisk præcision kan give 1e-16 i stedet for 0
#     x = jnp.where(jnp.abs(x) < 1e-7, 0, x)
    
#     return x, w

# def compute_ZLB_dummy(par, P, states):

#     # states: (Nparallel, 3) or (Nparallel, gh_n_combined, 3)

#     beta = par["beta"]
#     phi = par["phi"]
#     ZLB = par["ZLB"]

#     P_pi = P[1, :, None] # (3, 1)

#     lin_pi = (states @ P_pi)[..., 0]
#     u = states[..., 0]

#     ZLB_dummy = 1/beta - 1 + phi*lin_pi+u <= ZLB

#     return ZLB_dummy.astype(jnp.float32)

# def draw_test_data(par, train):

#     T = train["T"]
#     Nparallel_test = train["Nparallel_test"]

#     rho_i = par["rho_i"]
#     rho_z = par["rho_z"]
#     rho_Gamma = par["rho_Gamma"]
#     sigma_eps_i = par["sigma_eps_i"]
#     sigma_eps_z = par["sigma_eps_z"]
#     sigma_eps_Gamma = par["sigma_eps_Gamma"]

#     eps_i = np.random.normal(0, sigma_eps_i, size=(Nparallel_test, T))
#     eps_z = np.random.normal(0, sigma_eps_z, size=(Nparallel_test, T))
#     eps_Gamma = np.random.normal(0, sigma_eps_Gamma, size=(Nparallel_test, T))

#     shocks = jnp.stack([eps_i, eps_z, eps_Gamma], axis=-1) # (Nparallel_test, T, 3)

#     Rho = jnp.array([rho_i, rho_z, rho_Gamma])[None, None, :]
#     states = jnp.sum(shocks * Rho**jnp.flip(jnp.arange(T))[None, :, None], axis=1)

#     return shocks, states

# def next_eps_hist_quad(eps_hist, quad): 
    
#     # eps_hist is (Nparallel,T)
#     Nparallel = eps_hist.shape[0]
#     Nquad = len(quad)

#     eps_hist_repeated = eps_hist[:,1:].repeat(Nquad,axis=0)
#     quad_tiled = jnp.tile(quad, Nparallel)

#     return jnp.concatenate((eps_hist_repeated, quad_tiled.reshape(-1,1)), axis=1)

# def next_eps_hist_quad(eps_hist, quad):

#     # eps_hist with shape (Nparallel, T, 3)
#     # quad with shape (gh_n_combined, 3)
#     N_parallel = eps_hist.shape[0]
#     N_quad_nodes = quad.shape[0]

#     # drop last period
#     eps_hist = eps_hist[:, 1:]

#     # repeat along T-dimension: gh_n_combined identical worlds for each parallel world
#     eps_hist_repeated = eps_hist.repeat(N_quad_nodes,axis=0)

#     # tile quad along Nparallel dimension
#     quad_tiled = jnp.tile(quad, (N_parallel, 1))

#     # add time dimension
#     quad_tiled = quad_tiled[:, None, :]

#     # combine
#     eps_hist_next = jnp.concat([eps_hist_repeated, quad_tiled], axis=1) # shape (Nparallel*gh_n_combined, T, 3)

#     return eps_hist_next

# def next_eps_hist(eps_hist, eps):

#     # eps_hist is shape (Nparallel, T, 3)
#     # eps is shape (Nparallel, 3)

#     # truncate eps_hist at latest period
#     eps_hist_truncated = eps_hist[:, 1:]

#     # add T-dimension to eps
#     eps = eps[:, None, :]

#     # combine
#     eps_hist_next = jnp.concat([eps_hist_truncated, eps], axis=1)
    
#     return eps_hist_next

def fischer_equation(i, pi_p):

    return (1+i[:, None])/(1+pi_p)-1

####################
# INITIAL TRAINING #
####################

# def linear_loss_fn(par, train, linear, nn, states):

#     # 1. compute linear policies
#     Y_lin, pi_lin = simulate_linear_policy(par, linear, states)

#     # 2. compute nn policy
#     Nparallel = states.shape[0]
#     Y, pi = eval_nn(par, train, linear, nn, states, Nparallel)

#     # 3. compute nn grads
#     #nn_jac_fn = jax.vmap(jax.jacrev(nn))
#     #nn_jacs = nn_jac_fn(states)
#     #jac_loss = jnp.mean((nn_jacs - linear["P"][None, :, :])**2)

#     Y_error = optax.losses.squared_error(Y-Y_lin)
#     pi_error = optax.losses.squared_error(pi-pi_lin)

#     return jnp.mean(Y_error) + jnp.mean(pi_error)

# def train_linear(key, par, train, linear, dtype, Nparallel, T, linear_episodes, nn, opt, final_sigma_dict):

#     rho_u = par["rho_u"]
#     rho_z = par["rho_z"]
#     rho_Gamma = par["rho_Gamma"]
#     sigma_eps_u = par["sigma_eps_u"]
#     sigma_eps_z = par["sigma_eps_z"]
#     sigma_eps_Gamma = par["sigma_eps_Gamma"]
#     states = draw_states_directly(key, par, dtype, Nparallel, final_sigma_dict)

#     @nnx.jit
#     def train_step_linear(nn, opt, states):

#         loss_fn_linear = lambda nn: linear_loss_fn(par, train, linear, nn, states)
#         loss, grad = nnx.value_and_grad(loss_fn_linear)(nn)
#         opt.update(grad)

#         return loss

#     for k_pre in range(linear_episodes):

#         loss = train_step_linear(nn, opt, states)

#         if k_pre % 10 == 0:

#             print(f'Pre-episode {k_pre}:\tLinear in-sample loss: {loss:.8f}')

#         states = draw_states_directly(key, par, dtype, Nparallel, final_sigma_dict)
def OccBin(par, states, P, A_ZLB, B_ZLB, C_ZLB, D_ZLB, K_ZLB, T_max, print_=False):

    # infer N
    N = states.shape[0]
    d = jnp.zeros((2,1))

    # allocate solutions
    X_sol_init = jnp.zeros((N, 2)) + jnp.nan
    solved_init = jnp.zeros(N, dtype=jnp.bool)
    states_T_init = states @ K_ZLB.T

    carry_init = (X_sol_init, solved_init, states_T_init)

    P_hist, d_hist = compute_P_star(P, A_ZLB, B_ZLB, C_ZLB, D_ZLB, K_ZLB, T_max)

    def scan_fun(carry, T):

        # unpack carry
        X_sol, solved, states_T = carry

        # find implied ZLB policy today if ZLB holds until T periods into the future
        P_next, d_next = P_hist[T], d_hist[T]
        X, ZLB_binds = compute_policy_and_ZLB(par, states, P_next, d_next)
        ZLB_binds = ZLB_binds | (T == 0)

        # compute future policy: check if ZLB is expected to be slack in period T
        _, ZLB_binds_T = compute_policy_and_ZLB(par, states_T, P, d)
        ZLB_slack_T = ~ZLB_binds_T

        # if ZLB is expected to be slack in period T, it binds in this period and the period was not solved before -> fill in
        fill_in_mask = ZLB_slack_T & ZLB_binds & (~solved) 
        X_sol_next = jnp.where(fill_in_mask[:, None], X, X_sol)
        solved_next = solved | fill_in_mask

        # state transition in expectations
        states_T_next = states_T @ K_ZLB.T

        return (X_sol_next, solved_next, states_T_next), None


    # # backwards induction loop
    # T = 0
    # for T in range(T_max): # T = 0, 1, 2, 3, ..., today, going into the future

    #     # find implied ZLB policy today if ZLB holds until T periods into the future
    #     P_next, d_next = P_hist[T], d_hist[T]
    #     X, ZLB_binds = compute_policy_and_ZLB(par, states, P_next, d_next)
    #     ZLB_binds = ZLB_binds | (T == 0)

    #     # compute future policy: check if ZLB is expected to be slack in period T
    #     states_T = states @ jnp.linalg.matrix_power(K_ZLB, T + 1).T
    #     d = jnp.zeros((2,1))
    #     X_T, ZLB_binds_T = compute_policy_and_ZLB(par, states_T, P, d)
    #     ZLB_slack_T = jnp.logical_not(ZLB_binds_T)

    #     # if ZLB is expected to be slack in period T, it binds in this period and the period was not solved before -> fill in
    #     fill_in_mask = ZLB_slack_T & ZLB_binds & (~solved) 
    #     X_sol = jnp.where(fill_in_mask[:, None], X, X_sol)

    #     # prepare for next
    #     solved = solved | fill_in_mask
        
    #     if print_:

    #         print(f'{fill_in_mask.sum()} was filled in in T = {T}')

    #     if solved.sum() == N:
    #         if print_: print(f'All economics were solved before T = {T+1}')
    #         break

    # if (T == T_max-1) and (N-solved.sum() > 0.0):
    #     print(f'T_max was reached and all economies have not been solved: Missing {N-solved.sum()}')


    (X_sol_final, solved_final, _), _ = jax.lax.scan(scan_fun, carry_init, jnp.arange(T_max))

    return X_sol_final

    #states = jnp.zeros((Nparallel, 3))

    # def phase_episode_scan():

    #     # c. print
    #     if (k % print_freq == 0) or (k == 0):

    #         loss_test, ee, nkpce, te, comp_slack = loss(nn, par, train, linear, dtype, states_test, gh_x, gh_w, return_all_errors=True)
    #         if loss_test < best_loss: best_loss = loss_test

    #         # compute SSS
    #         Y_SSS, pi_SSS, _, _ = eval_nn(par, train, linear, nn, jnp.zeros((1, 3)), 1)
    #         Y_SSS, pi_SSS = Y_SSS[0], pi_SSS[0]
    #         print(f'Episode {k}:\tLoss = {loss_test:.8f}\tBest Loss = {best_loss:.8f}\t\tY_SSS = {Y_SSS:.3f}, pi_SSS = {pi_SSS:.3f}\t\tee = {ee:.8f}\tnkpce = {nkpce:.8f}\tte = {te:.8f}\tcomp_slack = {comp_slack:.8f}')
    #         #print(jnp.max(Y_errors), jnp.max(pi_errors))

    #     key, subkey = jax.random.split(key)
    #     eps = draw_shocks(subkey, dtype, Nparallel, sigma_eps_u, sigma_eps_z, sigma_eps_Gamma)
    #     states = next_states(par, states, eps)

    #     train_step(nn, opt, states)

    # jax.lax.scan(phase_episode)

# def taylor_rule_error(par, pi, i, u, ZLB):

#     i_shadow = taylor_rule(par, pi, u, ZLB)

# #     return i-i_shadow

#         # shocks
#         common_sigma_state = 0.1

#         # MP-shock: pushes up nominal interest rate 1:1
#         par["rho_u"] = 0.9747
#         par["sigma_eps_u"] = compute_sigma_eps(par["rho_u"], common_sigma_state)
#         par["mu_u"] = 0.0

#         # Demand shock: shock to MU in period t
#         par["rho_z"] = 0.9
#         par["sigma_eps_z"] = compute_sigma_eps(par["rho_z"], common_sigma_state)
#         par["mu_z"] = 0.0 # compute_log_AR_mean(par["z_DSS"], par["rho_z"], par["sigma_eps_z"])

#         # Supply shock: shock to MC
#         par["rho_Gamma"] = 0.7559
#         par["sigma_eps_Gamma"] = compute_sigma_eps(par["rho_Gamma"], common_sigma_state)
#         par["mu_Gamma"] = 0.0 # compute_log_AR_mean(par["ln_Gamma_DSS"], par["rho_Gamma"], par["sigma_eps_Gamma"])

#         # shocks: matrix form:
#         par["Rho"] = jnp.array([
#             [par["rho_u"], 0.0, 0.0],
#             [0.0, par["rho_z"], 0.0],
#             [0.0, 0.0, par["rho_Gamma"]]
#             ])

#         par["Mu"] = jnp.array([
#             [par["mu_u"]],
#             [par["mu_z"]],
#             [par["mu_Gamma"]]
#         ])