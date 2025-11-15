clear;
clc;
close all;

% Simulation Parameters
measurement_noise_std = 0.75;
dt = 0.1;

% Setup Scenario
[scenario, egoVehicle] = Simulate();
scenario.SampleTime = dt; 

% Simulation Loop
true_history = [];
measured_history = [];
time_history = [];

while advance(scenario)
    current_time = scenario.SimulationTime;
    true_position_3d = egoVehicle.Position;
    true_position_2d = true_position_3d(1:2);
    
    noise_vec = randn(1, 2) * measurement_noise_std;
    measured_position_2d = true_position_2d + noise_vec;
    
    time_history = [time_history; current_time];
    true_history = [true_history; true_position_2d];
    measured_history = [measured_history; measured_position_2d];
end
fprintf('Simulation complete. Generated %d data points.\n', length(time_history));

% EKF Implementation
fprintf('Running Tuned Extended Kalman Filter (EKF)...\n');

R = [measurement_noise_std^2  0;
     0                        measurement_noise_std^2];

q_pos_stability = 0.01;
q_theta_stability = 0.001;
q_v = 0.1;
q_omega = 2.0;

Q = diag([q_pos_stability^2, ...
          q_pos_stability^2, ...
          q_v^2, ...
          q_theta_stability^2, ...
          q_omega^2]);

x_k = [measured_history(1, 1);
       measured_history(1, 2);
       30;
       0;
       0];
P_k = diag([10, 10, 10, 0.1, 0.1]);

h = @(x) [x(1); x(2)];
Hj = [1 0 0 0 0;
      0 1 0 0 0];

kalman_history = zeros(length(time_history), 5);
kalman_history(1, :) = x_k';

% Main EKF Loop
for k = 2:length(time_history)
    
    x_prev = x_k;
    P_prev = P_k;
    
    x_pos = x_prev(1);
    y_pos = x_prev(2);
    v     = x_prev(3);
    theta = x_prev(4);
    omega = x_prev(5);

    if abs(omega) < 1e-4
        x_pred = x_pos + v * dt * cos(theta);
        y_pred = y_pos + v * dt * sin(theta);
    else
        x_pred = x_pos + (v/omega) * (sin(theta + omega*dt) - sin(theta));
        y_pred = y_pos + (v/omega) * (-cos(theta + omega*dt) + cos(theta));
    end
    
    x_k_predict = [
        x_pred;
        y_pred;
        v;
        theta + omega*dt;
        omega;
    ];

    Fj = eye(5);
    if abs(omega) < 1e-4
        Fj(1, 3) = dt * cos(theta);
        Fj(1, 4) = -v * dt * sin(theta);
        Fj(2, 3) = dt * sin(theta);
        Fj(2, 4) = v * dt * cos(theta);
        Fj(4, 5) = dt;
    else
        Fj(1, 3) = (sin(theta + omega*dt) - sin(theta)) / omega;
        Fj(1, 4) = (v/omega) * (cos(theta + omega*dt) - cos(theta));
        Fj(1, 5) = (v*dt*omega*cos(theta + omega*dt) - v*sin(theta + omega*dt) + v*sin(theta)) / omega^2;
        Fj(2, 3) = (-cos(theta + omega*dt) + cos(theta)) / omega;
        Fj(2, 4) = (v/omega) * (sin(theta + omega*dt) - sin(theta));
        Fj(2, 5) = (v*dt*omega*sin(theta + omega*dt) + v*cos(theta + omega*dt) - v*cos(theta)) / omega^2;
        Fj(4, 5) = dt;
    end
    
    P_k_predict = Fj * P_prev * Fj' + Q;
    
    z_k = measured_history(k, :)';
    y_k = z_k - h(x_k_predict);
    S_k = Hj * P_k_predict * Hj' + R;
    K_k = P_k_predict * Hj' / S_k;
    x_k = x_k_predict + (K_k * y_k);
    P_k = (eye(5) - K_k * Hj) * P_k_predict;
    
    kalman_history(k, :) = x_k';
end

fprintf('EKF (Tuned & Stable) complete.\n');

% Plot Final Comparison
figure;
plot(true_history(:, 1), true_history(:, 2), 'b-', 'LineWidth', 2, 'DisplayName', 'Ground Truth Path');
hold on;
plot(measured_history(:, 1), measured_history(:, 2), 'rx', 'MarkerSize', 6, 'DisplayName', 'Noisy Measurements (GPS)');
plot(kalman_history(:, 1), kalman_history(:, 2), 'g--', 'LineWidth', 2, 'DisplayName', 'EKF Estimate (Tuned)');
xlabel('X Position (m)');
ylabel('Y Position (m)');
title('EKF Tracking Performance (Tuned for Turn)');
legend('show', 'Location', 'best');
grid on;
axis equal;

figure;
subplot(2, 1, 1);
plot(time_history, true_history(:, 1), 'b-', 'LineWidth', 2, 'DisplayName', 'True X');
hold on;
plot(time_history, measured_history(:, 1), 'rx', 'MarkerSize', 4, 'DisplayName', 'Measured X');
plot(time_history, kalman_history(:, 1), 'g--', 'LineWidth', 2, 'DisplayName', 'EKF Estimate X');
xlabel('Time (s)');
ylabel('X Position (m)');
title('X Position over Time');
legend('show', 'Location', 'best');
grid on;
subplot(2, 1, 2);
plot(time_history, true_history(:, 2), 'b-', 'LineWidth', 2, 'DisplayName', 'True Y');
hold on;
plot(time_history, measured_history(:, 2), 'rx', 'MarkerSize', 4, 'DisplayName', 'Measured Y');
plot(time_history, kalman_history(:, 2), 'g--', 'LineWidth', 2, 'DisplayName', 'EKF Estimate Y');
xlabel('Time (s)');
ylabel('Y Position (m)');
title('Y Position over Time');
legend('show', 'Location', 'best');
grid on;

% Calculate Advanced Metrics (RMSE, MAE, MAXE, Bias)
fprintf('\n--- Performance Metrics (Advanced) ---\n');

pos_true = true_history(:, 1:2);
pos_measured = measured_history(:, 1:2);
pos_filtered = kalman_history(:, 1:2);

error_noisy_vec = sqrt(sum((pos_measured - pos_true).^2, 2));
error_filtered_vec = sqrt(sum((pos_filtered - pos_true).^2, 2));

rmse_noisy = sqrt(mean(error_noisy_vec.^2));
rmse_filtered = sqrt(mean(error_filtered_vec.^2));

mae_noisy = mean(error_noisy_vec);
mae_filtered = mean(error_filtered_vec);

maxe_noisy = max(error_noisy_vec);
maxe_filtered = max(error_filtered_vec);

raw_error_noisy = pos_measured - pos_true;
raw_error_filtered = pos_filtered - pos_true;

bias_noisy = mean(raw_error_noisy);
bias_filtered = mean(raw_error_filtered);

fprintf('Metric:          Noisy Meas.   Filtered Est.\n');
fprintf('----------------------------------------------\n');
fprintf('RMSE:            %.4f m      %.4f m\n', rmse_noisy, rmse_filtered);
fprintf('MAE (Avg Error): %.4f m      %.4f m\n', mae_noisy, mae_filtered);
fprintf('MAX (Worst Error): %.4f m      %.4f m\n', maxe_noisy, maxe_filtered);

fprintf('\nFilter Bias (X, Y):\n');
fprintf('  Noisy:       [%.3f m, %.3f m]\n', bias_noisy(1), bias_noisy(2));
fprintf('  Filtered:    [%.3f m, %.3f m]\n', bias_filtered(1), bias_filtered(2));

improvement_percent = 100 * (rmse_noisy - rmse_filtered) / rmse_noisy;
fprintf('\nOverall RMSE Improvement: %.2f %%\n', improvement_percent);