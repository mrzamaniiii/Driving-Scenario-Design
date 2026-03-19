function Unsupervised

clear; clc; close all;
rng(1);

% -----------------------------
% 0) User settings
% -----------------------------
measurement_noise_std = 5;   % GPS measurement noise std [m]
dt_default = 0.01;           % default sample time [s]

% 1) Generate data (ground truth + noisy GPS)
[scenario, egoVehicle] = Trajectory();
scenario.SampleTime = dt_default;

time_history = [];
true3_history = [];

while advance(scenario)
    time_history  = [time_history;  scenario.SimulationTime];
    true3_history = [true3_history; egoVehicle.Position];
end

N = size(true3_history,1);
fprintf('Simulation complete. Generated %d samples.\n', N);

vars = var(true3_history,0,1);
[~, idx] = sort(vars,'descend');
planeIdx = idx(1:2);

pos_true = true3_history(:,planeIdx);                            % [N x 2] ground truth
pos_meas = pos_true + randn(N,2)*measurement_noise_std;          % [N x 2] noisy GPS

% Compute metrics for noisy GPS baseline
m_noisy = compute_metrics(pos_meas, pos_true);

fprintf('\nNoisy GPS:\n');
fprintf('  RMSE = %.4f m\n', m_noisy.rmse);
fprintf('  MAE  = %.4f m\n', m_noisy.mae);
fprintf('  MAX  = %.4f m\n', m_noisy.maxe);

% KF measurement model (H) and base measurement noise (R0)
% Measurement: z = H*x + v, where x = [px, py, vx, vy, ax, ay]^T
H = single([1 0 0 0 0 0;
            0 1 0 0 0 0]);

% Base measurement noise covariance
R0 = diag([measurement_noise_std^2 measurement_noise_std^2]);

% 3) Normalization for NN inputs (stability)
pos_meas_single = single(pos_meas);

% Normalize inputs for network
mu_z    = mean(pos_meas_single, 1);
sigma_z = std(pos_meas_single, 0, 1);
sigma_z(sigma_z < 1e-3) = 1;

% Build dt statistics for normalized dt feature
dt_vec = diff(time_history);
dt_vec(~(isfinite(dt_vec) & dt_vec > 0)) = dt_default;
dt_mean = mean(dt_vec);
dt_std  = std(dt_vec);
if dt_std < 1e-6
    dt_std = 1;
end

% Convert constants to dlarray for training graph
mu_z_dl     = dlarray(single(mu_z));
sigma_z_dl  = dlarray(single(sigma_z));
dt_mean_dl  = dlarray(single(dt_mean));
dt_std_dl   = dlarray(single(dt_std));



% 4) RNN definition (learn q_acc(t) and r_scale(t))
% Input u_k = [innov_n(2); z_n(2); dt_n(1)] => inDim = 5
hiddenDim = 32;
inDim = 5;

scale = 0.05;
params = struct();
params.Wxh = dlarray(scale*randn(hiddenDim, inDim, 'single'));     % input-to-hidden
params.Whh = dlarray(scale*randn(hiddenDim, hiddenDim, 'single')); % hidden-to-hidden
params.bh  = dlarray(zeros(hiddenDim,1,'single'));                 % hidden bias

% Output layer produces two scalars: [q_raw; r_raw]
params.Wo  = dlarray(scale*randn(2, hiddenDim, 'single'));
params.bo  = dlarray(zeros(2,1,'single'));

% Adam optimizer states
avgGrad   = struct('Wxh',[], 'Whh',[], 'bh',[], 'Wo',[], 'bo',[]);
avgSqGrad = avgGrad;

% -----------------------------
% Training hyperparameters
% -----------------------------
numEpochs  = 100;
learnRate  = 5e-4;
beta1      = 0.9;
beta2      = 0.999;
epsAdam    = 1e-8;
clipVal    = 1.0;
lambda_reg = 1e-4;

% Output ranges
q_min = 0.5;   % minimum process noise intensity
q_max = 25.0;  % maximum process noise intensity
r_min = 0.5;   % minimum measurement noise scaling
r_max = 2.0;   % maximum measurement noise scaling

% Training data as dlarray
pos_meas_dl = dlarray(single(pos_meas));        % [N x 2]
time_dl     = dlarray(single(time_history));    % [N x 1]
H_dl        = dlarray(H);
R0_dl       = dlarray(single(R0));

% -----------------------------
% Training loop
% -----------------------------
for epoch = 1:numEpochs

    % Compute loss and gradients through differentiable KF + RNN
    [loss, grad, nis_mean_mon] = dlfeval(@modelGradients_UnsupervisedNLL, ...
        params, pos_meas_dl, time_dl, dt_default, H_dl, R0_dl, ...
        mu_z_dl, sigma_z_dl, dt_mean_dl, dt_std_dl, ...
        q_min, q_max, r_min, r_max, lambda_reg);

    % Gradient clipping
    grad = clipGradStruct(grad, clipVal);

    % Adam update
    [params, avgGrad, avgSqGrad] = adamUpdateStruct(params, grad, avgGrad, avgSqGrad, ...
        epoch, learnRate, beta1, beta2, epsAdam);

    % Logging
    loss_val = double(gather(extractdata(loss)));
    nis_val  = double(gather(extractdata(nis_mean_mon)));
    fprintf('Epoch %3d: loss = %.6f   meanNIS(train) = %.4f\n', epoch, loss_val, nis_val);
end

% Inference with learned model + NIS gating
opts.use_nis_gating = true;
opts.gate_prob = 0.99;  % chi-square gate probability for 2D measurements

[x_hat, q_series, r_series, stats] = infer_unsup_kfca(params, pos_meas, time_history, dt_default, ...
    H, R0, mu_z, sigma_z, dt_mean, dt_std, q_min, q_max, r_min, r_max, opts);

% Metrics versus ground truth
m_hat = compute_metrics(x_hat(:,1:2), pos_true);

fprintf('\nUnsupervised (NLL-trained):\n');
fprintf('  RMSE = %.4f m\n', m_hat.rmse);
fprintf('  MAE  = %.4f m\n', m_hat.mae);
fprintf('  MAX  = %.4f m\n', m_hat.maxe);
fprintf('  Bias = [%.3f, %.3f] m\n', m_hat.bias(1), m_hat.bias(2));
fprintf('  Mean NIS = %.4f (expected ~2)\n', stats.nis_mean);
fprintf('  Rejected updates = %.2f %%\n', 100*stats.reject_rate);
fprintf('  q_acc range = [%.3f, %.3f]\n', min(q_series), max(q_series));
fprintf('  r_scale range = [%.3f, %.3f]\n', min(r_series), max(r_series));

fprintf('\n===== SUMMARY =====\n');
fprintf('Metric             Noisy GPS      Unsupervised KF-CA\n');
fprintf('RMSE [m]        =  %8.4f      %8.4f\n', m_noisy.rmse, m_hat.rmse);
fprintf('MAE  [m]        =  %8.4f      %8.4f\n', m_noisy.mae,  m_hat.mae);
fprintf('MAX  [m]        =  %8.4f      %8.4f\n', m_noisy.maxe, m_hat.maxe);
fprintf('Bias X [m]      =  %8.4f      %8.4f\n', m_noisy.bias(1), m_hat.bias(1));
fprintf('Bias Y [m]      =  %8.4f      %8.4f\n', m_noisy.bias(2), m_hat.bias(2));
fprintf('Mean NIS        =      n/a      %8.4f\n', stats.nis_mean);

axisNames = {'X','Y','Z'};
d1 = axisNames{planeIdx(1)};
d2 = axisNames{planeIdx(2)};

figure;
plot(pos_true(:,1), pos_true(:,2), 'b-', 'LineWidth', 2); hold on;
plot(pos_meas(:,1), pos_meas(:,2), 'rx', 'MarkerSize', 3);
plot(x_hat(:,1), x_hat(:,2), 'm-.', 'LineWidth', 2);
legend('True','Noisy GPS','Unsup KF-CA','Location','best');
axis equal; grid on;
title('Unsupervised KF-CA (NLL) Tracking');
xlabel([d1 ' [m]']); ylabel([d2 ' [m]']);

figure;
subplot(2,1,1);
plot(time_history, pos_true(:,1), 'b-', 'LineWidth', 2, 'DisplayName', ['True ' d1]); hold on;
plot(time_history, pos_meas(:,1), 'rx', 'MarkerSize', 3, 'DisplayName', ['Measured ' d1]);
plot(time_history, x_hat(:,1), 'm-.', 'LineWidth', 2, 'DisplayName', ['Unsup KF-CA ' d1]);
xlabel('Time [s]');
ylabel([d1 ' [m]']);
title([d1 ' position over time']);
legend('Location','best');
grid on;

subplot(2,1,2);
plot(time_history, pos_true(:,2), 'b-', 'LineWidth', 2, 'DisplayName', ['True ' d2]); hold on;
plot(time_history, pos_meas(:,2), 'rx', 'MarkerSize', 3, 'DisplayName', ['Measured ' d2]);
plot(time_history, x_hat(:,2), 'm-.', 'LineWidth', 2, 'DisplayName', ['Unsup KF-CA ' d2]);
xlabel('Time [s]');
ylabel([d2 ' [m]']);
title([d2 ' position over time']);
legend('Location','best');
grid on;

figure;
plot(time_history(2:end), stats.nis_series, 'k-', 'LineWidth', 1.2); hold on;
yline(stats.gate_thr, 'r--', 'LineWidth', 1.2);
xlabel('Time [s]');
ylabel('NIS');
title('NIS and gating threshold');
grid on;

figure;
subplot(2,1,1);
plot(time_history(2:end), q_series, 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('q\_acc(t) [m/s^2]');
title('Learned q\_acc(t)');
grid on;

subplot(2,1,2);
plot(time_history(2:end), r_series, 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('r\_scale(t)');
title('Learned measurement noise scale r(t)');
grid on;

end


% Local helper functions

function m = compute_metrics(est,truep)
% Compute standard tracking metrics (2D Euclidean error)
% est, truep: [N x 2]
e = est - truep;                 % [N x 2] error in each axis
err = sqrt(sum(e.^2,2));         % [N x 1] Euclidean error magnitude
m.rmse = sqrt(mean(err.^2));     % root mean square error
m.mae  = mean(err);              % mean absolute error (in Euclidean sense)
m.maxe = max(err);               % maximum Euclidean error
m.bias = mean(e,1);              % bias per axis
end

function [loss, grad, nis_mean_mon] = modelGradients_UnsupervisedNLL( ...
    params, pos_meas_dl, time_dl, dt_default, H, R0, ...
    mu_z_dl, sigma_z_dl, dt_mean_dl, dt_std_dl, ...
    q_min, q_max, r_min, r_max, lambda_reg)
% Compute loss and gradients for the unsupervised NLL objective.

% Forward pass
[loss_nll, nis_mean_mon] = forward_unsup_nll(params, pos_meas_dl, time_dl, dt_default, ...
    H, R0, mu_z_dl, sigma_z_dl, dt_mean_dl, dt_std_dl, q_min, q_max, r_min, r_max);

% L2 weight decay regularization (helps prevent overfitting / blow-up)
wd = sum(params.Wxh.^2,'all') + sum(params.Whh.^2,'all') + sum(params.bh.^2,'all') + ...
     sum(params.Wo.^2 ,'all') + sum(params.bo.^2 ,'all');

loss = loss_nll + lambda_reg * wd;

% Backprop through the entire computation graph
[grad.Wxh, grad.Whh, grad.bh, grad.Wo, grad.bo] = dlgradient(loss, ...
    params.Wxh, params.Whh, params.bh, params.Wo, params.bo);
end



function [loss_nll, nis_mean_mon] = forward_unsup_nll(params, pos_meas_dl, time_dl, dt_default, ...
    H, R0, mu_z_dl, sigma_z_dl, dt_mean_dl, dt_std_dl, q_min, q_max, r_min, r_max) % Unsupervised learning
% Differentiable forward pass:
% Runs a CA Kalman Filter

T = size(pos_meas_dl,1);
hiddenDim = size(params.bh,1);

% Normalization constants
mu    = mu_z_dl.';        % [2 x 1]
sigma = sigma_z_dl.';     % [2 x 1]

% KF initial state and covariance (kept simple here)
x = [pos_meas_dl(1,1)'; pos_meas_dl(1,2)'; 0; 0; 0; 0];
P = dlarray(diag(single([25 25 900 900 2500 2500])));
I = dlarray(eye(6,'single'));

% RNN hidden state
h = zeros(hiddenDim,1,'like',pos_meas_dl);

% Accumulators
acc_loss  = dlarray(single(0));
acc_nis   = dlarray(single(0));
count     = dlarray(single(0));

for k = 2:T

    % Use measured time step
    dt_k = time_dl(k) - time_dl(k-1);
    dt_k = dlarray(single(dt_default)) + 0*dt_k + dt_k; % keeps graph stable
    dt_k = max(dt_k, dlarray(single(1e-6)));            % avoid dt <= 0

    % CA state transition matrix F(dt)
    F = [1 0 dt_k 0 0.5*dt_k^2 0;
         0 1 0 dt_k 0 0.5*dt_k^2;
         0 0 1 0 dt_k 0;
         0 0 0 1 0 dt_k;
         0 0 0 0 1 0;
         0 0 0 0 0 1];

    % Process-noise injection matrix G(dt)
    G = [0.5*dt_k^2 0;
         0 0.5*dt_k^2;
         dt_k 0;
         0 dt_k;
         1 0;
         0 1];

    % KF prediction
    x_pred = F*x;
    P_pred = F*P*F';

    % Measurement and predicted measurement
    z    = pos_meas_dl(k,:)';
    y_hat = H*x_pred;

    % Normalized innovation and normalized measurement
    z_n     = (z    - mu) ./ sigma;
    y_n     = (y_hat - mu) ./ sigma;
    innov_n = z_n - y_n;

    % Normalized dt feature
    dt_n = (dt_k - dt_mean_dl) ./ dt_std_dl;

    % RNN input: [innov_n; z_n; dt_n]
    u = [innov_n; z_n; dt_n];

    % RNN update
    h = tanh(params.Wxh*u + params.Whh*h + params.bh);

    % Output: [q_raw; r_raw]  ___ Network generate Q matrix
    o = params.Wo*h + params.bo;

    % Map outputs into valid ranges using sigmoid
    q01 = 1 ./ (1 + exp(-o(1)));
    r01 = 1 ./ (1 + exp(-o(2)));

    q = q_min + (q_max - q_min) * q01;   % process noise intensity
    r = r_min + (r_max - r_min) * r01;   % measurement noise scaling

    % Build Q and R for this step
    Q = (q.^2) * (G*G');
    R = (r.^2) * R0;

    % KF prediction covariance including process noise
    P_pred = P_pred + Q;

    % Innovation and innovation covariance
    innov = z - y_hat;
    S = H*P_pred*H' + dlarray(single(R));

    % 2x2 inverse and determinant
    s11 = S(1,1); s12 = S(1,2);
    s21 = S(2,1); s22 = S(2,2);
    detS = s11.*s22 - s12.*s21;
    detS = detS + dlarray(single(1e-6));  % numerical safety

    invS = [ s22, -s12;
            -s21,  s11] ./ detS;

    % NIS = innov' * invS * innov
    nis = innov' * (invS * innov);

    % Innovation NLL (up to constants): NIS + log|S|
    logdetS = log(detS);
    acc_loss = acc_loss + (nis + logdetS);

    % Monitor mean NIS
    acc_nis = acc_nis + nis;
    count = count + 1;

    % KF update
    K = (P_pred*H') * invS;              
    x = x_pred + K*innov;
    P = (I-K*H)*P_pred*(I-K*H)' + K*dlarray(single(R))*K';
    P = 0.5*(P+P');                     % enforce symmetry
end

loss_nll     = acc_loss / max(count, dlarray(single(1)));
nis_mean_mon = acc_nis  / max(count, dlarray(single(1)));
end



function [x_hat, q_series, r_series, stats] = infer_unsup_kfca(params, pos_meas, time, dt_default, ...
    H, R0, mu_z, sigma_z, dt_mean, dt_std, q_min, q_max, r_min, r_max, opts)
% Inference phase (numeric KF + learned RNN)
% - Runs a CA Kalman Filter using learned q(t), r(t)

T = size(pos_meas,1);

% Initialize velocity from early measurement differences
k0 = min(max(2, round(0.2/dt_default)), T-1);
dp = pos_meas(1+k0,:) - pos_meas(1,:);
v0 = dp / (k0*dt_default);

% Initial state and covariance
x = [pos_meas(1,1); pos_meas(1,2); v0(1); v0(2); 0; 0];
P = diag([25 25 900 900 2500 2500]);
I = eye(6);

% Outputs
x_hat = zeros(T,6);
x_hat(1,:) = x.';

% RNN hidden state (numeric)
hiddenDim = size(params.bh,1);
h = zeros(hiddenDim,1,'single');

% Normalization constants
mu = mu_z(:);
sigma = sigma_z(:);

% Store learned noise sequences
q_series = zeros(T-1,1);
r_series = zeros(T-1,1);

% NIS gating threshold for 2D measurement
gate_thr = chi2inv(opts.gate_prob,2);

% NIS statistics
nis_sum = 0;
nis_count = 0;
rejected = 0;
nis_series = zeros(T-1,1);

% Extract trained weights to numeric arrays
Wxh = gather(extractdata(params.Wxh));
Whh = gather(extractdata(params.Whh));
bh  = gather(extractdata(params.bh));
Wo  = gather(extractdata(params.Wo));
bo  = gather(extractdata(params.bo));

for k = 2:T

    % dt from timestamps
    dt_k = time(k) - time(k-1);
    if ~(isfinite(dt_k) && dt_k > 0)
        dt_k = dt_default;
    end

    % CA transition F(dt)
    F = [1 0 dt_k 0 0.5*dt_k^2 0;
         0 1 0 dt_k 0 0.5*dt_k^2;
         0 0 1 0 dt_k 0;
         0 0 0 1 0 dt_k;
         0 0 0 0 1 0;
         0 0 0 0 0 1];

    % Noise injection G(dt)
    G = [0.5*dt_k^2 0;
         0 0.5*dt_k^2;
         dt_k 0;
         0 dt_k;
         1 0;
         0 1];

    % KF prediction
    x_pred = F*x;
    P_pred = F*P*F';

    % Measurement and predicted measurement
    z = pos_meas(k,:)';
    y_hat = H*x_pred;

    % Normalized innovation and normalized measurement
    z_n     = (z - mu) ./ sigma;
    y_n     = (y_hat - mu) ./ sigma;
    innov_n = z_n - y_n;

    % Normalized dt feature
    dt_n = (dt_k - dt_mean) / dt_std;

    % RNN input: [innov_n; z_n; dt_n]
    u = single([innov_n; z_n; dt_n]);

    % RNN update
    h = tanh(Wxh*u + Whh*h + bh);

    % Output: [q_raw; r_raw]
    o = Wo*h + bo;

    % Map outputs to valid ranges
    q01 = 1 ./ (1 + exp(-o(1)));
    r01 = 1 ./ (1 + exp(-o(2)));

    q = q_min + (q_max - q_min) * q01;
    r = r_min + (r_max - r_min) * r01;

    % Store learned values
    q_series(k-1) = q;
    r_series(k-1) = r;

    % Build Q and R
    Q = (q^2) * (G*G');
    R = (r^2) * R0;

    % Finish prediction covariance
    P_pred = P_pred + Q;

    % Innovation and innovation covariance
    innov = z - y_hat;
    S = H*P_pred*H' + R;

    % NIS for gating and monitoring
    nis = innov'*(S\innov);
    nis_series(k-1) = nis;

    % Gating: reject update if NIS too large
    if opts.use_nis_gating && (nis > gate_thr)
        x = x_pred;
        P = P_pred;
        rejected = rejected + 1;
    else
        % Standard KF update (Joseph form)
        K = P_pred*H'/S;
        x = x_pred + K*innov;
        P = (I-K*H)*P_pred*(I-K*H)' + K*R*K';
        P = 0.5*(P+P');
        nis_sum = nis_sum + nis;
        nis_count = nis_count + 1;
    end

    x_hat(k,:) = x.';
end

% Pack stats
stats.nis_mean    = nis_sum/max(nis_count,1);
stats.reject_rate = rejected/max(T-1,1);
stats.nis_series  = nis_series;
stats.gate_thr    = gate_thr;
end

function grad = clipGradStruct(grad, clipVal)
% Clip each gradient tensor by its global L2 norm
fields = fieldnames(grad);
for i = 1:numel(fields)
    f = fields{i};
    g = grad.(f);
    g_norm = sqrt(sum(g.^2,'all'));
    if g_norm > clipVal
        grad.(f) = g * (clipVal / (g_norm + 1e-12));
    end
end
end

function [params, avgGrad, avgSqGrad] = adamUpdateStruct(params, grad, avgGrad, avgSqGrad, t, lr, beta1, beta2, epsAdam)
% Adam optimizer update for a struct of parameters
fields = fieldnames(params);
for i = 1:numel(fields)
    f = fields{i};
    g = grad.(f);

    % Initialize moving averages if needed
    if isempty(avgGrad.(f))
        avgGrad.(f)   = zeros(size(g), 'like', g);
        avgSqGrad.(f) = zeros(size(g), 'like', g);
    end

    % Exponential moving averages
    avgGrad.(f)   = beta1 * avgGrad.(f)   + (1 - beta1) * g;
    avgSqGrad.(f) = beta2 * avgSqGrad.(f) + (1 - beta2) * (g.^2);

    % Bias correction
    avgGradCorr   = avgGrad.(f)   / (1 - beta1^t);
    avgSqGradCorr = avgSqGrad.(f) / (1 - beta2^t);

    % Parameter update
    params.(f) = params.(f) - lr * avgGradCorr ./ (sqrt(avgSqGradCorr) + epsAdam);
end
end
