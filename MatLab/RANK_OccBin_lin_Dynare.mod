var y_hat pi_hat i_hat i_not u z ln_Gamma;
varexo eps_u eps_z eps_Gamma;
parameters
  BETA       
  KAPPA      
  PHI_Y      
  PHI_PI     
  RHO_U      
  RHO_Z      
  RHO_GAMMA  
  PHI        
  ALPHA      
  SIGMA      
  MU         
  ZLB        
  % parameters to change shock sizes in IRFs
  SHOCK_SIZE_U
  SHOCK_SIZE_Z
  SHOCK_SIZE_GAMMA
;
% Baseline parametre
BETA      = 0.99;
KAPPA     = 0.099;
PHI_Y     = 0.5;
PHI_PI    = 1.5;
RHO_U     = 0.84;
RHO_Z     = 0.74872;
RHO_GAMMA = 0.68245;
PHI       = 1.0;
ALPHA     = 0.25;
SIGMA     = 1.0;
MU        = 9.0/8.0;
ZLB       = -0.01510;
% set shock scale to 0
SHOCK_SIZE_U     = 0;
SHOCK_SIZE_Z     = 0;
SHOCK_SIZE_GAMMA = 0;
model;
[name='Euler equation']
z - SIGMA*y_hat = i_hat + RHO_Z*z - pi_hat(+1) - SIGMA*y_hat(+1);
[name='NKPC']
pi_hat = (KAPPA/MU)*((PHI+ALPHA+SIGMA-ALPHA*SIGMA)/(1-ALPHA))*y_hat
    - (KAPPA/MU)*((1+PHI)/(1-ALPHA))*ln_Gamma
    + BETA*pi_hat(+1);
[name='Notional Taylor rule']
i_not = PHI_PI*pi_hat + PHI_Y*y_hat
    - PHI_Y*((1+PHI)/(PHI+ALPHA+SIGMA-ALPHA*SIGMA))*ln_Gamma
    + u;
[name='Observed interest rate', relax='zlb']
i_hat = i_not;
[name='Observed interest rate', bind='zlb']
i_hat = ZLB;
% multiply shocks by scales
[name='u process']
u = RHO_U*u(-1) + SHOCK_SIZE_U * eps_u;
[name='z process']
z = RHO_Z*z(-1) + SHOCK_SIZE_Z * eps_z;
[name='ln_Gamma process']
ln_Gamma = RHO_GAMMA*ln_Gamma(-1) + SHOCK_SIZE_GAMMA * eps_Gamma;
end;
steady_state_model;
y_hat    = 0;
pi_hat   = 0;
i_hat    = 0;
i_not    = 0;
u        = 0;
z        = 0;
ln_Gamma = 0;
end;
steady;
occbin_constraints;
name 'zlb'; bind i_not <= ZLB;
end;
% consider all 3 shocks
shocks(surprise);
var eps_u;     periods 1; values 1;
var eps_z;     periods 1; values 1;
var eps_Gamma; periods 1; values 1;
end;
shock_names = {'u_pos', 'u_neg', 'z_pos', 'z_neg', 'Gamma_pos', 'Gamma_neg'};

% allocate matrix for results
all_results = zeros(20, M_.endo_nbr, 6);
all_linear_results = zeros(20, M_.endo_nbr, 6); % NY LINJE: Matrix til lineære IRF'er

for idx = 1:6
    % 1. zero out shock scales
    set_param_value('SHOCK_SIZE_U', 0);
    set_param_value('SHOCK_SIZE_Z', 0);
    set_param_value('SHOCK_SIZE_GAMMA', 0);
    
    % 2. set differences shocks to pm 0.05
    switch idx
        case 1; set_param_value('SHOCK_SIZE_U', 0.013500000000000002);       % u positive
        case 2; set_param_value('SHOCK_SIZE_U', -0.013500000000000002);      % u negative
        case 3; set_param_value('SHOCK_SIZE_Z', 0.20814);       % z positive
        case 4; set_param_value('SHOCK_SIZE_Z', -0.20814);      % z negative
        case 5; set_param_value('SHOCK_SIZE_GAMMA', 0.12778);   % Gamma positive
        case 6; set_param_value('SHOCK_SIZE_GAMMA', -0.12778);  % Gamma negative
    end
    
    % 3. call OccBin
    occbin_setup;
    occbin_solver;

    % 4. save 20 first periods of IRF
    all_results(:, :, idx) = oo_.occbin.simul.piecewise(1:20, :);
    all_linear_results(:, :, idx) = oo_.occbin.simul.linear(1:20, :); % NY LINJE: Gemmer den lineære simulation
end
% save all 6 shock processes in matlab file
var_names = cellstr(M_.endo_names);
save('all_6_shocks_results.mat', 'all_results', 'all_linear_results', 'shock_names', 'var_names');