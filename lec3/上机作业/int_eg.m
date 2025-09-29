%% 重要性采样计算尾概率 P(X>4), X~N(0,1)
clc, clear
% 真实概率密度函数（标准正态）
p = @(x) exp(-x.^2/2) / sqrt(2*pi);

true_p = erfc(2*sqrt(2)) / 2;

%% 方法1：简单蒙特卡洛（效率极低）
fprintf('=== 方法比较：计算 P(X>4) ===\n\n');

N = 100000;
x_mc = randn(N, 1);  % 从标准正态采样
success_mc = sum(x_mc >= 4);
p_mc = success_mc / N;
fprintf('简单蒙特卡洛:\n');
fprintf('  估计值: %.6e\n', p_mc);
fprintf('  有效样本数: %d/%d (效率: %.4f%%)\n', success_mc, N, 100*success_mc/N);
fprintf('  相对误差: %.2f%%\n\n', 100*abs(p_mc - true_p)/true_p);

%% 方法2：重要性采样（使用平移的正态分布作为建议分布）
% 选择建议分布：N(4.5, 1)，集中在重要区域
mu_q = 4.5;
sigma_q = 1;
q = @(x) exp(-(x-mu_q).^2/(2*sigma_q^2)) / (sigma_q*sqrt(2*pi));

% 从建议分布采样
x_is = mu_q + sigma_q * randn(N, 1);

% 计算重要性权重
weights = p(x_is) ./ q(x_is);

% 计算指示函数和加权平均
indicator = (x_is > 4);
p_is = mean(indicator .* weights);

fprintf('重要性采样:\n');
fprintf('  估计值: %.6e\n', p_is);
fprintf('  有效样本数: %d/%d (效率: %.2f%%)\n', sum(x_is>4), N, 100*sum(x_is>4)/N);
fprintf('  相对误差: %.2f%%\n\n', 100*abs(p_is - true_p)/true_p);

% 理论值参考
fprintf('理论值: %.6e\n', true_p);



%% 可视化比较
figure;

% 子图1：分布对比
subplot(2,3,1);

x_plot = 2:0.1:7;
plot(x_plot, p(x_plot), 'b-', 'LineWidth', 3); hold on;
plot(x_plot, q(x_plot), 'r--', 'LineWidth', 2);
legend('真实分布 p(x)', '建议分布 q(x)', 'Location', 'northeast');
title('分布函数对比');
xlabel('x'); ylabel('概率密度');
grid on;

% 子图2：蒙特卡洛采样分布
subplot(2,3,2);
histogram(x_mc, 50, 'Normalization', 'pdf', 'FaceColor', 'blue', 'FaceAlpha', 0.6);
hold on;
plot(x_plot, p(x_plot), 'b-', 'LineWidth', 2);
xline(4, 'r--', 'LineWidth', 2, 'Label', 'x=4');
title('蒙特卡洛采样分布');
xlabel('x'); ylabel('密度');
xlim([2, 7]);
grid on;

% 子图3：重要性采样分布
subplot(2,3,3);
histogram(x_is, 50, 'Normalization', 'pdf', 'FaceColor', 'red', 'FaceAlpha', 0.6);
hold on;
plot(x_plot, p(x_plot), 'b-', 'LineWidth', 2);
plot(x_plot, q(x_plot), 'r--', 'LineWidth', 2);
xline(4, 'r--', 'LineWidth', 2, 'Label', 'x=4');
title('重要性采样分布');
xlabel('x'); ylabel('密度');
xlim([2, 7]);
grid on;

% 子图4：权重分布
subplot(2,3,4);
valid_samples = x_is > 4;
histogram(weights(valid_samples), 30, 'FaceColor', 'green', 'FaceAlpha', 0.6);
title('有效样本的权重分布');
xlabel('权重 w(x)'); ylabel('频数');
grid on;

% 子图5：收敛性比较
subplot(2,3,5);
sample_sizes = round(logspace(2, 5, 50)); % 从1e2 ~ 1e5
mc_estimates = zeros(size(sample_sizes));
is_estimates = zeros(size(sample_sizes));

for i = 1:length(sample_sizes)
    n = sample_sizes(i);
    
    % 蒙特卡洛
    x_temp = randn(n, 1);
    mc_estimates(i) = sum(x_temp > 4) / n;
    
    % 重要性采样
    x_temp = mu_q + sigma_q * randn(n, 1);
    w_temp = p(x_temp) ./ q(x_temp);
    is_estimates(i) = mean((x_temp > 4) .* w_temp);
end

loglog(sample_sizes, abs(mc_estimates - true_p), 'b-', 'LineWidth', 2); hold on;
loglog(sample_sizes, abs(is_estimates - true_p), 'r-', 'LineWidth', 2);
plot(sample_sizes, 1./sqrt(sample_sizes)*1e-4, 'k--', 'LineWidth', 1); % 可以不画
legend('蒙特卡洛误差', '重要性采样误差', '1/√N 参考', 'Location', 'northeast');
title('收敛速度比较');
xlabel('样本数'); ylabel('绝对误差');
grid on;

% 子图6：方差比较
subplot(2,3,6);
num_trials = 100;
mc_vars = zeros(num_trials, 1);
is_vars = zeros(num_trials, 1);

% 实验100轮绘制箱型图
for trial = 1:num_trials
    % 蒙特卡洛
    x_mc_var = randn(N, 1);
    mc_vars(trial) = sum(x_mc_var > 4) / N;

    % 重要性采样
    x_is_var = mu_q + sigma_q * randn(N, 1);
    w_var = p(x_is_var) ./ q(x_is_var);
    is_vars(trial) = mean((x_is_var > 4) .* w_var);
end

boxplot([mc_vars, is_vars], 'Labels', {'蒙特卡洛', '重要性采样'});
title('估计值方差比较');
ylabel('P(X>4)估计值');
grid on;

sgtitle('重要性采样 vs 简单蒙特卡洛：尾概率计算比较', 'FontSize', 14, 'FontWeight', 'bold');

%% 不同建议分布的效果比较
fprintf('\n=== 不同建议分布的效果比较 ===\n');

proposal_params = [
    4.0, 0.5;
    4.5, 1.0;
    5.0, 1.5;
    4.2, 0.8;
    4.8, 1.2
];

for i = 1:size(proposal_params, 1)
    mu = proposal_params(i, 1);
    sigma = proposal_params(i, 2);
    
    q_test = @(x) exp(-(x-mu).^2/(2*sigma^2)) / (sigma*sqrt(2*pi));
    x_test = mu + sigma * randn(N, 1);
    w_test = p(x_test) ./ q_test(x_test);
    p_test = mean((x_test > 4) .* w_test);
    
    efficiency = 100 * sum(x_test > 4) / N;
    variance = var((x_test > 4) .* w_test);
    
    fprintf('建议分布 N(%.1f, %.1f): 估计值=%.6e, 效率=%.1f%%, 方差=%.6e\n', ...
            mu, sigma, p_test, efficiency, variance);
end