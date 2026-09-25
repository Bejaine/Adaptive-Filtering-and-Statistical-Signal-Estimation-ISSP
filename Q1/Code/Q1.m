%% Problem 1

% Noise cancellation of ECG signals wihtout reference signal and
% qualitative analysis of k0 and p through NLMS

clear; clc; close all;

load('Signal1a.mat'); % y_new_2
load('Signal1b.mat'); % y_new_1

data_list = {y_new_2, y_new_1}; 
signal_labels = {'Signal 1a', 'Signal 1b'};

% k0 and p values
k0_vals = [2, 10, 50]; 
p_fixed = 5;

p_vals = [2, 5, 10];
k0_fixed = 10;

% Normalized step size (beta) since we are using NLMS
beta = 0.1;
ignore_samples = 200; % Samples to skip in plots to hide filter convergence

% Variables to store results for Part C comparison
best_y_1a = [];
best_y_1b = [];

for s = 1:2
    x = data_list{s}(:); % Ensuring signal is a column vector
    
    %% Figure 1: Effect of k0 (with a fixed p)

    figure('Name', [signal_labels{s}, ' - Effect of k0'], 'Position', [100, 100, 800, 600]);
    for i = 1:length(k0_vals)
        k0 = k0_vals(i);
        n0 = k0 + 1;
        
        % Running NLMS Adaptive Line Enhancer
        [y_out, ~] = nlms_ale(x, p_fixed, n0, beta);
        
        subplot(3, 1, i);
        % Plot of skipping the convergence transient
        n_plot = ignore_samples:length(x);
        plot(n_plot, x(n_plot), 'Color', [0.8 0.8 0.8]); hold on;
        plot(n_plot, y_out(n_plot), 'r', 'LineWidth', 1.2);
        
        title(sprintf('%s | k_0 = %d (Fixed p = %d)', signal_labels{s}, k0, p_fixed));
        xlabel('Samples'); ylabel('Amplitude');
        xlim([500 1500]);
        legend('Original Noisy', 'Denoised', 'Location', 'northeast');
        grid on;
    end
    
    %% Figure 2: Effect of Filter Order p (with a fixed k0)

    figure('Name', [signal_labels{s}, ' - Effect of p'], 'Position', [150, 150, 800, 600]);
    for j = 1:length(p_vals)
        p = p_vals(j);
        n0 = k0_fixed + 1;
        
        % Running NLMS Adaptive Line Enhancer
        [y_out, ~] = nlms_ale(x, p, n0, beta);
        
        % Save the best result (p=10, k0=10) for Part C
        if p == 10
            if s == 1, best_y_1a = y_out; else, best_y_1b = y_out; end
        end
        
        subplot(3, 1, j);
        plot(n_plot, x(n_plot), 'Color', [0.8 0.8 0.8]); hold on;
        plot(n_plot, y_out(n_plot), 'b', 'LineWidth', 1.2);
        
        title(sprintf('%s | p = %d (Fixed k_0 = %d)', signal_labels{s}, p, k0_fixed));
        xlabel('Samples'); ylabel('Amplitude');
        xlim([500 1500]);
        legend('Original Noisy', 'Denoised', 'Location', 'northeast');
        grid on;
    end
end

%% Part C - Comparison of Signal 1a vs 1b

figure('Name', 'Part C: Denoising Comparison', 'Position', [200, 200, 800, 400]);

subplot(2,1,1);
plot(n_plot, best_y_1a(n_plot), 'r', 'LineWidth', 1.2);
title('Best Denoised Signal 1a (p=10, k_0=10)');
xlim([500 1500]); grid on;

subplot(2,1,2);
plot(n_plot, best_y_1b(n_plot), 'b', 'LineWidth', 1.2);
title('Best Denoised Signal 1b (p=10, k_0=10)');
xlim([500 1500]); grid on;

sgtitle('Part C: Which signal is easier to denoise?');

%% Function: Normalized LMS Adaptive Line Enhancer (NLMS-ALE)

function [y, e] = nlms_ale(x, p, n0, beta)
    N = length(x);
    y = zeros(N, 1);
    e = zeros(N, 1);
    w = zeros(p, 1); 
    epsilon = 1e-6; % Small constant to prevent division by zero
    
    for n = (p + n0):N
        % u is the delayed signal (reference input)
        u = x(n-n0 : -1 : n-n0-p+1);
        
        y(n) = w' * u; % Prediction of the clean signal
        e(n) = x(n) - y(n); % Prediction error (Noise)
        
        % NLMS Weight Update (Divided by u'*u)
        norm_u = u' * u;
        w = w + (beta / (norm_u + epsilon)) * e(n) * u; 
    end
end