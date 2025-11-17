% 可视化偏差-方差的概念
figure('Position', [100, 100, 1200, 400]);

% 低偏差低方差（理想情况）
subplot(1,4,1);
theta = 0:0.01:2*pi;
plot(cos(theta), sin(theta), 'r-', 'LineWidth', 2); hold on;
plot(0.1*randn(20,1), 0.1*randn(20,1), 'bo', 'MarkerSize', 6);
axis equal; title('低偏差低方差(理想情况)'); grid on;

% 高偏差低方差（系统性误差）
subplot(1,4,2);
plot(cos(theta), sin(theta), 'r-', 'LineWidth', 2); hold on;
plot(0.6+0.1*randn(20,1), 0.3+0.1*randn(20,1), 'bo', 'MarkerSize', 6);
axis equal; title('高偏差低方差(系统性误差)'); grid on;

% 低偏差高方差（过拟合）
subplot(1,4,3);
plot(cos(theta), sin(theta), 'r-', 'LineWidth', 2); hold on;
plot(0.5*randn(20,1), 0.5*randn(20,1), 'bo', 'MarkerSize', 6);
axis equal; title('低偏差高方差(过拟合)'); grid on;

% 高偏差高方差（最差情况）
subplot(1,4,4);
plot(cos(theta), sin(theta), 'r-', 'LineWidth', 2); hold on;
plot(0.5+0.5*randn(20,1), 0.5+0.5*randn(20,1), 'bo', 'MarkerSize', 6);
axis equal; title('高偏差高方差(最差情况)'); grid on;