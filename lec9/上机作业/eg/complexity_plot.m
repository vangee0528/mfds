% 模拟不同复杂度模型的偏差-方差变化
% 简单模型：高偏差，低方差
% 复杂模型：低偏差，高方差

% 可视化关系
figure;
complexity = 1:10;
bias = 1./complexity;           % 偏差随复杂度增加而减少
variance = 0.1 * complexity;    % 方差随复杂度增加而增加
total_error = bias + variance + 0.1; % 总误差

plot(complexity, bias, 'r-o', 'LineWidth', 2, 'DisplayName', '偏差^2');
hold on;
plot(complexity, variance, 'b-s', 'LineWidth', 2, 'DisplayName', '方差');
plot(complexity, total_error, 'k-^', 'LineWidth', 3, 'DisplayName', '总误差');

xlabel('模型复杂度');
ylabel('泛化误差');
title('偏差-方差权衡');
legend('show');
grid on;

% 标记最优点
[min_error, opt_idx] = min(total_error);
plot(complexity(opt_idx), min_error, 'ro', 'MarkerSize', 10, 'LineWidth', 3, 'DisplayName', '最佳模型');
text(complexity(opt_idx), min_error+0.1, '最优复杂度', 'HorizontalAlignment', 'center');