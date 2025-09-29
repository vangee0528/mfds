% 作业

% 计算积分
f = @(x) exp(2 * x);
p = @(x) exp(-x.^2 / 2) / sqrt(2);

true_p = exp(2) / 2;

% 积分区间 [2, +inf)

%% 直接使用蒙特卡罗方法
N = 1000;

x_mc = randn(1, N);
success_mc = x_mc >= 2;

p_mc = sum(f(x_mc) .* success_mc) / N;

fprintf('简单蒙特卡洛:\n');
fprintf('  估计值: %.6e\n', p_mc);
fprintf('  有效样本数: %d/%d (效率: %.4f%%)\n', sum(success_mc), N, 100*sum(success_mc)/N);
fprintf('  相对误差: %.2f%%\n\n', 100*abs(p_mc - true_p)/true_p);

%% 重要性采样



%% 绘制结果图