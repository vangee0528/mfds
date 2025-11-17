% 演示复杂模型对数据变化的敏感性
figure('Position', [100, 100, 1000, 600]);

% 生成两组略有不同的训练数据
rng(1);
x = 0:0.1:2;
y_true = 2*x + 1;

% 第一组数据
y1 = y_true + 0.3*randn(size(x));
% 第二组数据（稍微不同）
y2 = y_true + 0.3*randn(size(x));

% 简单模型（1次多项式）
subplot(2,2,1);
p1_simple = polyfit(x, y1, 1);
p2_simple = polyfit(x, y2, 1);
plot(x, polyval(p1_simple, x), 'r-', 'LineWidth', 2); hold on;
plot(x, polyval(p2_simple, x), 'b--', 'LineWidth', 2);
plot(x, y_true, 'k-', 'LineWidth', 1);
title('简单模型 (1次多项式)');
legend('数据集1拟合', '数据集2拟合', '真实关系');
grid on;

% 复杂模型（7次多项式）
subplot(2,2,2);
p1_complex = polyfit(x, y1, 7);
p2_complex = polyfit(x, y2, 7);
plot(x, polyval(p1_complex, x), 'r-', 'LineWidth', 2); hold on;
plot(x, polyval(p2_complex, x), 'b--', 'LineWidth', 2);
plot(x, y_true, 'k-', 'LineWidth', 1);
title('复杂模型 (7次多项式)');
legend('数据集1拟合', '数据集2拟合', '真实关系');
grid on;

% 计算方差对比
subplot(2,2,3);
x_test = 0:0.01:2;
% 生成多个数据集测试方差
n_models = 20;
simple_models = zeros(n_models, length(x_test));
complex_models = zeros(n_models, length(x_test));

for i = 1:n_models
    y_train = y_true + 0.3*randn(size(x));
    p_simple = polyfit(x, y_train, 1);
    p_complex = polyfit(x, y_train, 7);
    simple_models(i,:) = polyval(p_simple, x_test);
    complex_models(i,:) = polyval(p_complex, x_test);
end

% 计算每个点的预测方差
variance_simple = var(simple_models);
variance_complex = var(complex_models);

plot(x_test, variance_simple, 'g-', 'LineWidth', 3, 'DisplayName', '简单模型方差');
hold on;
plot(x_test, variance_complex, 'r-', 'LineWidth', 3, 'DisplayName', '复杂模型方差');
xlabel('x');
ylabel('预测方差');
title('模型方差比较');
legend('show');
grid on;

% 平均方差对比
subplot(2,2,4);
avg_variance = [mean(variance_simple), mean(variance_complex)];
bar(avg_variance);
set(gca, 'XTickLabel', {'简单模型', '复杂模型'});
ylabel('平均方差');
title('平均预测方差比较');