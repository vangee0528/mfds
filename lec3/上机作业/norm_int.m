
p = @(x) exp(-x.^2/2) / sqrt(2*pi);
true_p = 1/2 * (erf(4 * sqrt(2)) - erf(2 * sqrt(2)));

a = 4;
b = 8;

N = 100000;
%% 使用均匀分布作为建议分布，计算积分
x_mc = rand(N, 1) * (b - a) + a;  % 从均匀分布中采样
success_mc = sum((x_mc >= a) .* (x_mc <= b));  % 所有点都在被积区域内
p_mc = mean(p(x_mc)) * (b - a);

fprintf('重要性采样（均匀分布）:\n');
fprintf('  估计值: %.6e\n', p_mc);
fprintf('  有效样本数: %d/%d (效率: %.4f%%)\n', success_mc, N, 100*success_mc/N);
fprintf('  相对误差: %.2f%%\n\n', 100*abs(p_mc - true_p)/true_p);


%% 使用linspace代替均匀分布采样
x_mc = linspace(a, b, N);

success_mc = sum((x_mc >= a) .* (x_mc <= b));
p_mc = mean(p(x_mc)) * (b - a);

fprintf('重要性采样（linspace）:\n');
fprintf('  估计值: %.6e\n', p_mc);
fprintf('  有效样本数: %d/%d (效率: %.4f%%)\n', success_mc, N, 100*success_mc/N);
fprintf('  相对误差: %.2f%%\n\n', 100*abs(p_mc - true_p)/true_p);