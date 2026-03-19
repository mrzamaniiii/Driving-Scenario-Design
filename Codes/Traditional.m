function IMM_CV_CA_Full
clear; clc; close all;
rng(1);

measurement_noise_std = 5;
dt = 0.01;

[scenario, egoVehicle] = Trajectory();
scenario.SampleTime = dt;

time_history = [];
true3_history = [];

while advance(scenario)
    time_history = [time_history; scenario.SimulationTime];
    true3_history = [true3_history; egoVehicle.Position];
end

N = size(true3_history,1);
fprintf('Simulation complete. Generated %d samples.\n', N);

vars = var(true3_history,0,1);
[~, idx] = sort(vars,'descend');
planeIdx = idx(1:2);

pos_true = true3_history(:,planeIdx);
pos_meas = pos_true + randn(size(pos_true))*measurement_noise_std;

m_noisy = compute_metrics(pos_meas,pos_true);

opts.use_nis_gating = true;
opts.gate_prob = 0.99;

q_ca = 9;
q_cv = 1;

[x_imm, stats] = imm_cv_ca( ...
    pos_meas, time_history, dt, measurement_noise_std, q_ca, q_cv, opts);

m_imm = compute_metrics(x_imm(:,1:2),pos_true);

fprintf('\nTraditional Method:\n');
fprintf('  q_cv = %.3f m/s^2\n', q_cv);
fprintf('  q_ca = %.3f m/s^2\n', q_ca);
fprintf('  RMSE = %.4f m\n', m_imm.rmse);
fprintf('  MAE  = %.4f m\n', m_imm.mae);
fprintf('  MAX  = %.4f m\n', m_imm.maxe);
fprintf('  Bias = [%.3f, %.3f] m\n', m_imm.bias(1), m_imm.bias(2));
fprintf('  Mean NIS = %.4f\n', stats.nis_mean);
fprintf('  Rejected updates = %.2f %%\n', 100*stats.reject_rate);

fprintf('\n===== SUMMARY =====\n');
fprintf('Metric             Noisy GPS      Traditional Method\n');
fprintf('RMSE [m]        =  %8.4f      %8.4f\n', m_noisy.rmse, m_imm.rmse);
fprintf('MAE  [m]        =  %8.4f      %8.4f\n', m_noisy.mae,  m_imm.mae);
fprintf('MAX  [m]        =  %8.4f      %8.4f\n', m_noisy.maxe, m_imm.maxe);
fprintf('Bias X [m]      =  %8.4f      %8.4f\n', m_noisy.bias(1), m_imm.bias(1));
fprintf('Bias Y [m]      =  %8.4f      %8.4f\n', m_noisy.bias(2), m_imm.bias(2));
fprintf('Mean NIS        =      n/a      %8.4f\n', stats.nis_mean);

axisNames = {'X','Y','Z'};
d1 = axisNames{planeIdx(1)};
d2 = axisNames{planeIdx(2)};

figure;
plot(pos_true(:,1),pos_true(:,2),'b-','LineWidth',2,'DisplayName','True'); hold on;
plot(pos_meas(:,1),pos_meas(:,2),'rx','MarkerSize',3,'DisplayName','Noisy GPS');
plot(x_imm(:,1),x_imm(:,2),'g--','LineWidth',2,'DisplayName','Traditional Method');
xlabel([d1 ' position [m]']);
ylabel([d2 ' position [m]']);
title('Traditional Method Tracking');
legend('Location','best');
grid on; axis equal;

% X/Y position over time (added)
figure;
subplot(2,1,1);
plot(time_history, pos_true(:,1), 'b-', 'LineWidth', 2, 'DisplayName', ['True ' d1]);
hold on;
plot(time_history, pos_meas(:,1), 'rx', 'MarkerSize', 3, 'DisplayName', ['Measured ' d1]);
plot(time_history, x_imm(:,1), 'g--', 'LineWidth', 2, 'DisplayName', ['IMM ' d1]);
xlabel('Time [s]');
ylabel([d1 ' [m]']);
title([d1 ' position over time']);
legend('Location','best');
grid on;

subplot(2,1,2);
plot(time_history, pos_true(:,2), 'b-', 'LineWidth', 2, 'DisplayName', ['True ' d2]);
hold on;
plot(time_history, pos_meas(:,2), 'rx', 'MarkerSize', 3, 'DisplayName', ['Measured ' d2]);
plot(time_history, x_imm(:,2), 'g--', 'LineWidth', 2, 'DisplayName', ['IMM ' d2]);
xlabel('Time [s]');
ylabel([d2 ' [m]']);
title([d2 ' position over time']);
legend('Location','best');
grid on;

figure;
plot(time_history(2:end), stats.nis_series, 'k-','LineWidth',1.2); hold on;
yline(stats.gate_thr,'r--','LineWidth',1.2);
xlabel('Time [s]');
ylabel('NIS');
title('NIS and gating threshold');
grid on;

figure;
plot(time_history, stats.mu_series(:,1), 'LineWidth',1.5); hold on;
plot(time_history, stats.mu_series(:,2), 'LineWidth',1.5);
xlabel('Time [s]');
ylabel('Model probability');
legend('CV model','CA model','Location','best');
title('IMM model probabilities');
grid on;

end

function m = compute_metrics(est,truep)
e = est - truep;
err = sqrt(sum(e.^2,2));
m.rmse = sqrt(mean(err.^2));
m.mae = mean(err);
m.maxe = max(err);
m.bias = mean(e,1);
end

function [x_imm, stats] = imm_cv_ca(meas,time,dt_default,meas_std,q_ca,q_cv,opts)

N = size(meas,1);

R = diag([meas_std^2 meas_std^2]);
H = [1 0 0 0 0 0;
     0 1 0 0 0 0];

gate_thr = chi2inv(opts.gate_prob,2);

PI = [0.5 0.5;
      0.5 0.5];

mu = [0.5; 0.5];

dt0 = dt_default;
Kinit = min(max(3,round(0.3/dt0)),N-1);
v_samples = zeros(Kinit,2);
for i = 1:Kinit
    v_samples(i,:) = (meas(1+i,:) - meas(1,:)) / (i*dt0);
end
v0 = mean(v_samples,1);

x0 = [meas(1,1); meas(1,2); v0(1); v0(2); 0; 0];
P0 = diag([meas_std^2, meas_std^2, 20^2, 20^2, 10^2, 10^2]);

x1 = x0; P1 = P0;
x2 = x0; P2 = P0;

x_imm = zeros(N,6);
x_imm(1,:) = x0.';

nis_series = zeros(N-1,1);
mu_series = zeros(N,2);
mu_series(1,:) = mu.';

rejected = 0;
nis_sum = 0;
nis_count = 0;

for k = 2:N
    dtk = time(k) - time(k-1);
    if ~(isfinite(dtk) && dtk > 0)
        dtk = dt_default;
    end

    F = [1 0 dtk 0 0.5*dtk^2 0;
         0 1 0 dtk 0 0.5*dtk^2;
         0 0 1 0 dtk 0;
         0 0 0 1 0 dtk;
         0 0 0 0 1 0;
         0 0 0 0 0 1];

    G = [0.5*dtk^2 0;
         0 0.5*dtk^2;
         dtk 0;
         0 dtk;
         1 0;
         0 1];

    Q1 = (q_cv^2) * (G*G');
    Q2 = (q_ca^2) * (G*G');

    c = PI.'*mu;
    mu_ij = (PI .* (mu*ones(1,2))) ./ (ones(2,1)*c.');
    mu_ij(~isfinite(mu_ij)) = 0;

    x1_mix = mu_ij(1,1)*x1 + mu_ij(2,1)*x2;
    x2_mix = mu_ij(1,2)*x1 + mu_ij(2,2)*x2;

    P1_mix = mu_ij(1,1)*(P1 + (x1-x1_mix)*(x1-x1_mix)') + ...
             mu_ij(2,1)*(P2 + (x2-x1_mix)*(x2-x1_mix)');

    P2_mix = mu_ij(1,2)*(P1 + (x1-x2_mix)*(x1-x2_mix)') + ...
             mu_ij(2,2)*(P2 + (x2-x2_mix)*(x2-x2_mix)');

    [x1, P1, lik1, nis1, used1] = kf_step(F,H,Q1,R,x1_mix,P1_mix,meas(k,:)',gate_thr,opts.use_nis_gating);
    [x2, P2, lik2, nis2, used2] = kf_step(F,H,Q2,R,x2_mix,P2_mix,meas(k,:)',gate_thr,opts.use_nis_gating);

    if ~(used1 || used2)
        rejected = rejected + 1;
    end

    lik = [lik1; lik2] + 1e-300;
    mu = c .* lik;
    mu = mu / sum(mu);

    x = mu(1)*x1 + mu(2)*x2;
    P = mu(1)*(P1 + (x1-x)*(x1-x)') + mu(2)*(P2 + (x2-x)*(x2-x)');

    nis = mu(1)*nis1 + mu(2)*nis2;
    nis_series(k-1) = nis;

    if isfinite(nis) && (used1 || used2)
        nis_sum = nis_sum + nis;
        nis_count = nis_count + 1;
    end

    mu_series(k,:) = mu.'
    x_imm(k,:) = x.';

   
end

stats.nis_mean = nis_sum/max(nis_count,1);
stats.reject_rate = rejected/max(N-1,1);
stats.nis_series = nis_series;
stats.gate_thr = gate_thr;
stats.mu_series = mu_series;

end

function [x,P,lik,nis,used] = kf_step(F,H,Q,R,x,P,z,gate_thr,use_gating)
I = eye(size(P,1));

x_pred = F*x;
P_pred = F*P*F' + Q;

innov = z - H*x_pred;
S = H*P_pred*H' + R;

nis = innov'*(S\innov);

if use_gating && (nis > gate_thr)
    x = x_pred;
    P = P_pred;
    used = false;
    lik = 1.0;
else
    K = P_pred*H'/S;
    x = x_pred + K*innov;
    P = (I-K*H)*P_pred*(I-K*H)' + K*R*K';
    P = 0.5*(P+P');
    used = true;

    d = 2;
    detS = max(det(S), 1e-300);
    lik = exp(-0.5*nis) / sqrt(((2*pi)^d) * detS);
end
end
