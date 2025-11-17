% 过拟合MATLAB演示
x = 0:0.1:2;
y_true = 2*x + 1;                    % 真实关系：线性
y_noise = y_true + 0.1*randn(size(x)); % 加入噪声的观测值

% 拟合不同复杂度的模型
p1 = polyfit(x, y_noise, 1);         % 1次多项式（正确复杂度）
p7 = polyfit(x, y_noise, 12);        % 12次多项式（过于复杂）

% 绘制结果对比
figure;
plot(x, y_true, 'k-', 'LineWidth', 2); hold on;
plot(x, y_noise, 'bo');
plot(x, polyval(p1, x), 'r-');
plot(x, polyval(p7, x), 'g-');
legend('真实关系', '观测数据', '1次多项式', '12次多项式');