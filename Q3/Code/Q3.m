% Problem 3
% AR Compression Analysis using NLMS

clear; clc; close all;

filename = 'Signal3.wav';

[x, Fs] = audioread(filename);

% Converting to mono if stereo
if size(x, 2) > 1
    x = mean(x, 2);
end

% Ensuring x is a column vector
x = x(:);

%% Part (a) : Compute and plot the (average) compression gain as a function of filter length. 

M = 5; % Number of equal portions
N_pct = 0.1; % N = 10% for training
p_values = [2, 5, 10, 20, 30, 50, 75, 100]; % Different filter lengths to test
mu = 0.05; % Step size for Normalized LMS (NLMS)

avg_gain_p = zeros(length(p_values), 1);

for i = 1:length(p_values)
    p = p_values(i);
    avg_gain_p(i) = run_compression_sim(x, M, N_pct, p, mu);
end

figure;
plot(p_values, avg_gain_p, '-o', 'LineWidth', 2, 'MarkerSize', 6);
title('Part (a): Average Compression Gain vs. Filter Length (p)');
xlabel('Filter Length (p)');
ylabel('Average Compression Gain (Ratio of Norms)');
grid on;

%% Part (b): Effect of changing M (Number of Portions)

p_opt = 2; % Optimal length based on the monotonic decrease seen in Part (a)
N_pct = 0.1; % Fixed at 10%
M_values = [1, 2, 5, 10, 20, 50, 100]; % Different M values

avg_gain_M = zeros(length(M_values), 1);

for i = 1:length(M_values)
    M = M_values(i);
    avg_gain_M(i) = run_compression_sim(x, M, N_pct, p_opt, mu);
end

figure;
semilogx(M_values, avg_gain_M, '-s', 'LineWidth', 2, 'MarkerSize', 6, 'Color', 'r');
title(sprintf('Part (b): Average Compression Gain vs. M (Fixed p=%d, N=10%%)', p_opt));
xlabel('Number of Portions (M) - Log Scale');
ylabel('Average Compression Gain');
grid on;

%% Part (b): Effect of changing N (Training Percentage)

M = 5; % Fixed back to 5
p_opt = 2; % Fixed optimal filter length
N_values = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]; % Different percentages

avg_gain_N = zeros(length(N_values), 1);

for i = 1:length(N_values)
    N_pct = N_values(i);
    avg_gain_N(i) = run_compression_sim(x, M, N_pct, p_opt, mu);
end

figure;
plot(N_values * 100, avg_gain_N, '-d', 'LineWidth', 2, 'MarkerSize', 6, 'Color', 'g');
title(sprintf('Part (b): Average Compression Gain vs. Training Percentage N (Fixed p=%d, M=%d)', p_opt, M));
xlabel('Training Percentage (%)');
ylabel('Average Compression Gain');
grid on;

disp('Simulations complete. Please review the generated figures.');

%% Function for running compression simulation

function avg_gain = run_compression_sim(x, M, N_pct, p, mu)
    
    % x : Full signal
    % M : Number of portions
    % N_pct : Fraction of portion used for training (0 to 1)
    % p : Filter length (AR order)
    % mu : Step size for NLMS
    
    total_len = length(x);
    block_size = floor(total_len / M);
    train_len = floor(block_size * N_pct);
    test_len = block_size - train_len;
    
    % Ensuring training length is larger than filter order
    if train_len <= p
        avg_gain = NaN; % Cannot train if data is shorter than filter
        return;
    end
    
    gains = zeros(M, 1);
    
    for m = 1:M

        % Extracting current block
        start_idx = (m-1)*block_size + 1;
        end_idx = start_idx + block_size - 1;
        block = x(start_idx:end_idx);
        
        % Splitting into train and test
        x_train = block(1:train_len);
        x_test = block(train_len+1:end);
        
        % Training Phase (Learning AR parameters using NLMS)
        w = zeros(p, 1); % Filter weights
        epsilon = 1e-6; % Preventing division by zero in NLMS
        
        for n = p+1:train_len
            u = x_train(n-1 : -1 : n-p); % Input vector (past p samples)
            y_hat = w' * u; % Prediction
            e_train = x_train(n) - y_hat; % Error
            
            % NLMS Update: w(n+1) = w(n) + [mu / (||u||^2 + eps)] * e(n) * u
            w = w + (mu / (u'*u + epsilon)) * e_train * u; 
        end
        
        % Compression/Test Phase (Apply fixed weights to 90%)
        % To predict the first p samples of x_test, we need the last p samples of x_train
        buffer = [x_train(end-p+1:end); x_test];
        e_test = zeros(test_len, 1);
        
        for n = 1:test_len
            % Buffer index is shifted by p
            u = buffer(n+p-1 : -1 : n); 
            y_hat = w' * u; % Prediction using FIXED weights
            e_test(n) = x_test(n) - y_hat; % Residual error
        end
        
        % Compression Gain
        norm_signal = norm(x_test);
        norm_error = norm(e_test);
        
        if norm_error == 0
            gains(m) = Inf;
        else
            gains(m) = norm_signal / norm_error;
        end
    end
    
    % Average the gain across all M portions
    avg_gain = mean(gains);
end