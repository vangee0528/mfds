X = [1 2; 1 2.1; 1 2.2; 1 2.3];  % 高度相关的特征
y = [3; 3.2; 3.4; 3.5];

beta_ols = X \ y;

% 可视化对比OLS和岭回归的损失函数
figure('Position', [100, 100, 1200, 500]);

% OLS损失函数
subplot(1,3,1);
[b1, b2] = meshgrid(-2:0.1:4, -2:0.1:4);
loss_ols = zeros(size(b1));

for i = 1:size(b1,1)
    for j = 1:size(b1,2)
        y_pred = X * [b1(i,j); b2(i,j)];
        loss_ols(i,j) = sum((y - y_pred).^2);
    end
end

contour(b1, b2, loss_ols, 50);
hold on;
plot(beta_ols(1), beta_ols(2), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
title('OLS损失函数');
xlabel('\beta_1'); ylabel('\beta_2');
colorbar;

% 岭回归惩罚项
subplot(1,3,2);
penalty = b1.^2 + b2.^2;
contour(b1, b2, penalty, 50);
title('L2惩罚项');
xlabel('\beta_1'); ylabel('\beta_2');
colorbar;

% 岭回归总损失
subplot(1,3,3);
lambda = 1;
loss_ridge = loss_ols + lambda * penalty;
contour(b1, b2, loss_ridge, 50);
hold on;

% 计算岭回归解
I = eye(2);
beta_ridge = (X'*X + lambda*I) \ (X'*y);
plot(beta_ridge(1), beta_ridge(2), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
title('岭回归损失函数 (\lambda=1)');
xlabel('\beta_1'); ylabel('\beta_2');
colorbar;