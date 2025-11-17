% 创建有冗余特征的数据
rng(42);
n = 50; p = 10;
X = randn(n, p);

% 只有前3个特征真正相关
true_beta = [2, -1.5, 1, zeros(1, p-3)]';
y = X * true_beta + 0.5 * randn(n, 1);

fprintf('真实情况：只有前3个特征有影响，后7个是噪声特征\n');
fprintf('真实非零系数个数：%d\n', sum(true_beta ~= 0));

% 可视化对比三种损失函数
figure('Position', [100, 100, 1200, 400]);

[b1, b2] = meshgrid(-3:0.1:3, -3:0.1:3);

% OLS损失（椭圆）
subplot(1,3,1);
% 简化的二次损失
loss_ols = b1.^2 + 2*b1.*b2 + 3*b2.^2;
contour(b1, b2, loss_ols, 30);
title('OLS损失函数');
xlabel('\beta_1'); ylabel('\beta_2');

% 岭回归惩罚（圆形）
subplot(1,3,2);
penalty_ridge = b1.^2 + b2.^2;
contour(b1, b2, loss_ols + 1*penalty_ridge, 30);
title('岭回归损失 (L2惩罚)');
xlabel('\beta_1'); ylabel('\beta_2');

% Lasso惩罚（菱形）
subplot(1,3,3);
penalty_lasso = abs(b1) + abs(b2);
contour(b1, b2, loss_ols + 1*penalty_lasso, 30);
title('Lasso损失 (L1惩罚)');
xlabel('\beta_1'); ylabel('\beta_2');

% 展示为什么Lasso产生稀疏解
figure('Position', [100, 100, 1000, 500]);

% 创建更简单的例子便于可视化
X_simple = [1 0.8; 1 0.9; 1 1.0];
y_simple = [2; 2.1; 2.2];

[b1, b2] = meshgrid(0:0.05:2.5, 0:0.05:2.5);
loss = zeros(size(b1));

for i = 1:size(b1,1)
    for j = 1:size(b1,2)
        y_pred = X_simple * [b1(i,j); b2(i,j)];
        loss(i,j) = sum((y_simple - y_pred).^2);
    end
end

% OLS解
beta_ols = X_simple \ y_simple;

% 岭回归解
lambda = 5;
beta_ridge = (X_simple'*X_simple + lambda*eye(2)) \ (X_simple'*y_simple);

% Lasso解（近似）
beta_lasso = lasso(X_simple, y_simple);

subplot(1,2,1);
contour(b1, b2, loss, 30, 'LineWidth', 1); hold on;
% 绘制约束区域
theta = 0:0.01:2*pi;
plot(beta_ols(1), beta_ols(2), 'bo', 'MarkerSize', 8, 'LineWidth', 3, 'DisplayName', 'OLS');
plot(beta_ridge(1), beta_ridge(2), 'rs', 'MarkerSize', 8, 'LineWidth', 3, 'DisplayName', 'Ridge');
plot(beta_lasso(1), beta_lasso(2), 'g^', 'MarkerSize', 8, 'LineWidth', 3, 'DisplayName', 'Lasso');
xlabel('\beta_1'); ylabel('\beta_2');
title('约束优化视角');
legend('show'); grid on;

% 系数路径对比
subplot(1,2,2);
lambda_range = logspace(-2, 2, 30);

% 岭回归路径
beta_ridge_path = zeros(2, length(lambda_range));
for i = 1:length(lambda_range)
    beta_ridge_path(:,i) = (X_simple'*X_simple + lambda_range(i)*eye(2)) \ (X_simple'*y_simple);
end

% Lasso路径
[beta_lasso_path, fitinfo] = lasso(X_simple, y_simple, 'Lambda', lambda_range);

plot(lambda_range, beta_ridge_path(1,:), 'r-', 'LineWidth', 2, 'DisplayName', 'Ridge-β₁');
hold on;
plot(lambda_range, beta_ridge_path(2,:), 'r--', 'LineWidth', 2, 'DisplayName', 'Ridge-β₂');
plot(lambda_range, beta_lasso_path(1,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Lasso-β₁');
plot(lambda_range, beta_lasso_path(2,:), 'b--', 'LineWidth', 2, 'DisplayName', 'Lasso-β₂');
xlabel('λ'); ylabel('系数值');
title('系数路径对比');
legend('show'); grid on;
set(gca, 'XScale', 'log');