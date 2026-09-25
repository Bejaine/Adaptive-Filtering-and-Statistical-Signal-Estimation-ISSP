%% Problem 2

clear; clc; close all;

load('Signal2.mat'); 
xn = xn(:); yn = yn(:);
N = length(xn);

%% Part A - SDAF (Steepest Descent Adaptive Filter)

p_const = 5;

% Estimate Autocorrelation (R) and Cross-correlation (P) assuming WSS/Ergodicity
r_full = xcorr(xn, p_const-1, 'biased');
R = toeplitz(r_full(p_const:end));
p_full = xcorr(yn, xn, p_const-1, 'biased');
P_vec = p_full(p_const:end); 
P_vec = P_vec(:);

% 10% of the maximum mu limit for stability
mu = 0.1 * (2 / max(eig(R))); 

w_sdaf = zeros(p_const, 1);
iterations = 1000;
for i = 1:iterations
    w_sdaf = w_sdaf + mu * (P_vec - R * w_sdaf);
end

%% Part B - RLS Lambda Analysis & Comparison

% A smaller lambda (0.9) adapts fast but "shuffles" (more noise).
% A larger lambda (1.0) adapts slow but is smoother in steady-state.
lambdas = [0.9, 0.95, 0.99, 1.0];
delta = 1.0; 
mse_comparison = zeros(length(lambdas), 1);

figure; hold on;
for i = 1:length(lambdas)
    l = lambdas(i);
    w = zeros(p_const, 1);
    P_mat = eye(p_const) / delta; 
    e_plot = zeros(N, 1);
    
    for n = p_const:N
        u = xn(n:-1:n-p_const+1);
        
        % Gain vector update with stabilizer to prevent matrix explosion
        denom = l + u' * P_mat * u + 1e-6;
        k = (P_mat * u) / denom;
        
        e_plot(n) = yn(n) - w' * u; 
        w = w + k * e_plot(n);      
        P_mat = (P_mat - k * u' * P_mat) / l;
    end
    
    % Plotting raw squared error to see the "shuffling" noise
    plot(e_plot.^2, 'DisplayName', ['\lambda = ', num2str(l)]);
    
    % Calculate MSE using second half of the signal for steady-state analysis
    mse_comparison(i) = mean(e_plot(round(N/2):end).^2);
end

title('Part (b): RLS Learning Curves (Linear Scale)');
xlabel('Time index n'); ylabel('Squared Error (e^2)');
legend; grid on;

ylim([0 0.015]); 

% Selection of the best parameter based on steady-state MSE
[~, best_idx] = min(mse_comparison);
opt_lambda = lambdas(best_idx);
fprintf('Optimum lambda selected: %.2f\n', opt_lambda);

%% Part C - Plot Filter Coefficients for Selected Lambda and p=5

% Re-run RLS with the selected optimum lambda to track weights
w_rls = zeros(p_const, 1);
P_mat = eye(p_const) / delta;
w_history = zeros(p_const, N);
e_rls = zeros(N, 1);

for n = p_const:N
    u = xn(n:-1:n-p_const+1);
    k = (P_mat * u) / (opt_lambda + u' * P_mat * u + 1e-6);
    e_rls(n) = yn(n) - w_rls' * u;
    w_rls = w_rls + k * e_rls(n);
    P_mat = (P_mat - k * u' * P_mat) / opt_lambda;
    w_history(:, n) = w_rls;
end

figure;
plot(w_history');
title(['Part (c): RLS Coefficient Convergence (\lambda = ', num2str(opt_lambda), ')']);
xlabel('Time index n'); ylabel('Coefficient Values');
legend('w1','w2','w3','w4','w5'); grid on;

%% Part D - System Approximation Accuracy

% Plotting the difference between given output y[n] and system output.
figure;
plot(e_rls);
title('Part (d): Instantaneous Accuracy (y[n] - \hat{y}[n])');
xlabel('Time index n'); ylabel('Error'); grid on;

%% Part E-  Overall Accuracy vs. Filter Length (p = 1 to 10)

% Analysis of system approximation accuracy as a function of the filter order p.
p_range = 1:10;
mse_results = zeros(length(p_range), 1);

for idx = 1:length(p_range)
    p_curr = p_range(idx);
    w_tmp = zeros(p_curr, 1);
    P_tmp = eye(p_curr) / delta;
    e_tmp = zeros(N, 1);
    
    for n = p_curr:N
        u_tmp = xn(n:-1:n-p_curr+1);
        % Gain update with numerical stabilizer
        k_tmp = (P_tmp * u_tmp) / (opt_lambda + u_tmp' * P_tmp * u_tmp + 1e-6);
        err = yn(n) - (w_tmp' * u_tmp);
        e_tmp(n) = err;
        w_tmp = w_tmp + k_tmp * err;
        P_tmp = (P_tmp - k_tmp * u_tmp' * P_tmp) / opt_lambda;
    end
    mse_results(idx) = mean(e_tmp(round(N/2):end).^2);
end

figure;
plot(p_range, mse_results, 'r-o', 'LineWidth', 1.5);
title('Part (e): Steady-State Accuracy vs. Filter Order p');
xlabel('Filter Order p'); ylabel('Overall MSE'); grid on;