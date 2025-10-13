% 初始化参数和权重
n = 100;
x = rand(1,n);
x = sort(x);
y = x + sin(x) + 1e-2 * randn(1, n);
w1 = randn(20, 1);
b1 = randn(20, 1);
w2 = randn(1, 20);
b2 = randn(1, 1);
eta = 1e-3;
error_set = zeros(1, 1000);

% 训练循环
for iter = 1:1000
    % 前向传播
    t1 = w1 * x + repmat(b1, 1, n);
    x1 = t1 .* (t1 > 0);
    t2 = w2 * x1 + repmat(b2, 1, n);
    x2 = t2;
    error_set(iter) = norm(x2 - y);
    
    % 反向传播和权重更新
    delta2 = (x2 - y);
    delta1 = (w2' * delta2) .* (t1 > 0);
    b2 = b2 - eta * mean(delta2, 2);
    w2 = w2 - eta * mean(delta2 * x1', 2);
    b1 = b1 - eta * mean(delta1, 2);
    w1 = w1 - eta * mean(delta1 * x', 2);
end

% 显示最终误差
disp(error_set(end));
plot(x,x2,'r');
hold on 
plot(x,y,'b')