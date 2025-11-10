clear
clc

rng(42);

f_target = @(x1, x2) 2 .* x1 + x1 .* x2;

N = 21 * 21;
% 构建训练集合
[X1, X2] = meshgrid(linspace(-1, 1, 21), linspace(-1, 1, 21));
X_train = [X1(:), X2(:)];
Y_train = f_target(X_train(:, 1), X_train(:, 2));


s = 0.5;
% 定义网络 第一个位置是上标 后面的是下标
w_1_11.value = s * randn; w_1_11.grad = 0;  % n1
w_1_21.value = s * randn; w_1_21.grad = 0;
b_1_1.value = 0; b_1_1.grad = 0;

w_1_12.value = s * randn; w_1_12.grad = 0;  % n2
w_1_22.value = s * randn; w_1_22.grad = 0;
b_1_2.value = 0; b_1_2.grad = 0;

w_1_13.value = s * randn; w_1_13.grad = 0;  % n3
w_1_23.value = s * randn; w_1_23.grad = 0;
b_1_3.value = 0; b_1_3.grad = 0;

w_2_1.value = s * randn; w_2_1.grad = 0; % layer2
w_2_2.value = s * randn; w_2_2.grad = 0;
w_2_3.value = s * randn; w_2_3.grad = 0;
b_2_1.value = 0; b_2_1.grad = 0;

% 定义中间节点
n_1_11.value = 0; n_1_11.grad = 0;  % n1
n_1_21.value = 0; n_1_21.grad = 0;

n_1_12.value = 0; n_1_12.grad = 0;  % n2
n_1_22.value = 0; n_1_22.grad = 0;

n_1_13.value = 0; n_1_13.grad = 0;  % n3
n_1_23.value = 0; n_1_23.grad = 0;

n_1.value = 0; n_1.grad = 0;
n_2.value = 0; n_2.grad = 0;
n_3.value = 0; n_3.grad = 0;

t_1.value = 0; t_1.grad = 0;
t_2.value = 0; t_2.grad = 0;
t_3.value = 0; t_3.grad = 0;

n_2_1.value = 0; n_2_1.grad = 0;
n_2_2.value = 0; n_2_2.grad = 0;
n_2_3.value = 0; n_2_3.grad = 0; 

y1.value = 0; y1.grad = 0;



epoches = 2000; % 训练轮数
lr = 1e-2;


train_loss = zeros(1, epoches);

for epoch = 1:epoches
    % 当前轮次中计算前向传播的值
    
    % ------------ 梯度清零 --------------
    % layer1
    w_1_11.grad = 0; w_1_21.grad = 0; b_1_1.grad = 0; % n1
    w_1_12.grad = 0; w_1_22.grad = 0; b_1_2.grad = 0; % n2
    w_1_13.grad = 0; w_1_23.grad = 0; b_1_3.grad = 0; % n3
    
    % layer2
    w_2_1.grad = 0; w_2_2.grad = 0; w_2_3.grad = 0;
    b_2_1.grad = 0;
    
    % 中间节点
    n_1_11.grad = 0; n_1_21.grad = 0;
    n_1_12.grad = 0; n_1_22.grad = 0;
    n_1_13.grad = 0; n_1_23.grad = 0;
    n_1.grad = 0; n_2.grad = 0; n_3.grad = 0;
    t_1.grad = 0; t_2.grad = 0; t_3.grad = 0;
    n_2_1.grad = 0; n_2_2.grad = 0; n_2_3.grad = 0; 
    y1.grad = 0;
    
    if mod(epoch, 500) == 0
        lr = lr * 0.8;
    end

    % 遍历数据点
    for i = 1 : N
        x1 = X_train(i, 1);
        x2 = X_train(i, 2);
        y_target = Y_train(i);
    
        % ------------ 前向传播 --------------
        % 乘法节点
        n_1_11.value = x1 * w_1_11.value;  % n1
        n_1_21.value = x2 * w_1_21.value;
        
        n_1_12.value = x1 * w_1_12.value;  % n2
        n_1_22.value = x2 * w_1_22.value;
        
        n_1_13.value = x1 * w_1_13.value;  % n3
        n_1_23.value = x2 * w_1_23.value;

        % 加法节点
        n_1.value = n_1_11.value + n_1_21.value + b_1_1.value;
        n_2.value = n_1_12.value + n_1_22.value + b_1_2.value;
        n_3.value = n_1_13.value + n_1_23.value + b_1_3.value;

        % 激活函数节点
        t_1.value = max(n_1.value, 0);
        t_2.value = max(n_2.value, 0);
        t_3.value = max(n_3.value, 0);

        % 第二层乘法节点
        n_2_1.value = t_1.value * w_2_1.value;
        n_2_2.value = t_2.value * w_2_2.value;
        n_2_3.value = t_3.value * w_2_3.value; 

        % 第二层加法节点，最终输出
        y1.value = n_2_1.value + n_2_2.value + n_2_3.value + b_2_1.value;
        
        % 累加到当前的训练误差中
        train_loss(epoch) = train_loss(epoch) + (y1.value - f_target(x1, x2)).^2;
        

        % ------------ 反向传播 --------------

        % 输出层的梯度
        d_y1 = 2 .* (y1.value - f_target(x1, x2));
        
        
        % 第二层乘法节点梯度
        d_n_2_1 = d_y1; 
        d_n_2_2 = d_y1; 
        d_n_2_3 = d_y1; 

        % 第二层偏置梯度
        d_b_2_1 = d_y1; 

        % 第二层激活函数节点梯度
        d_t_1 = w_2_1.value * d_n_2_1; 
        d_t_2 = w_2_2.value * d_n_2_2; 
        d_t_3 = w_2_3.value * d_n_2_3; 
        % 第二层权重梯度
        d_w_2_1 = t_1.value * d_n_2_1; 
        d_w_2_2 = t_2.value * d_n_2_2; 
        d_w_2_3 = t_3.value * d_n_2_3; 

        % 第二层加法节点梯度
        d_n_1 = double(n_1.value > 0) * d_t_1; 
        d_n_2 = double(n_2.value > 0) * d_t_2;  
        d_n_3 = double(n_3.value > 0) * d_t_3; 

        % 第一层偏置梯度
        d_b_1_1 = d_n_1;
        d_b_1_2 = d_n_2;
        d_b_1_3 = d_n_3;

        % 第一层乘法节点梯度
        d_n_1_11 = d_n_1; 
        d_n_1_21 = d_n_1; 
        d_n_1_12 = d_n_2; 
        d_n_1_22 = d_n_2;
        d_n_1_13 = d_n_3; 
        d_n_1_23 = d_n_3; 

        % 第一层权重梯度
        d_w_1_11 = x1 * d_n_1_11; 
        d_w_1_21 = x2 * d_n_1_21; 
    
        d_w_1_12 = x1 * d_n_1_12;  
        d_w_1_22 = x2 * d_n_1_22;
    
        d_w_1_13 = x1 * d_n_1_13;  
        d_w_1_23 = x2 * d_n_1_23; 


        % 累计梯度
        w_1_11.grad = w_1_11.grad + d_w_1_11; 
        w_1_21.grad = w_1_21.grad + d_w_1_21; 
        b_1_1.grad = b_1_1.grad + d_b_1_1; 
    
        w_1_12.grad = w_1_12.grad + d_w_1_12; 
        w_1_22.grad = w_1_22.grad + d_w_1_22; 
        b_1_2.grad = b_1_2.grad + d_b_1_2; 
    
        w_1_13.grad = w_1_13.grad + d_w_1_13; 
        w_1_23.grad = w_1_23.grad + d_w_1_23; 
        b_1_3.grad = b_1_3.grad + d_b_1_3; 
        
        w_2_1.grad = w_2_1.grad + d_w_2_1; 
        w_2_2.grad = w_2_2.grad + d_w_2_2; 
        w_2_3.grad = w_2_3.grad + d_w_2_3;
    
        b_2_1.grad = b_2_1.grad + d_b_2_1;
    end

    train_loss(epoch) = train_loss(epoch) / N;
    
    

    % ------------ 参数更新 --------------
    w_1_11.value = w_1_11.value - lr * w_1_11.grad / N; 
    w_1_21.value = w_1_21.value - lr * w_1_21.grad / N; 
    b_1_1.value = b_1_1.value - lr * b_1_1.grad / N; 

    w_1_12.value = w_1_12.value - lr * w_1_12.grad / N; 
    w_1_22.value = w_1_22.value - lr * w_1_22.grad / N; 
    b_1_2.value = b_1_2.value - lr * b_1_2.grad / N; 

    w_1_13.value = w_1_13.value - lr * w_1_13.grad / N; 
    w_1_23.value = w_1_23.value - lr * w_1_23.grad / N; 
    b_1_3.value = b_1_3.value - lr * b_1_3.grad / N;  
    
    w_2_1.value = w_2_1.value - lr * w_2_1.grad / N; 
    w_2_2.value = w_2_2.value - lr * w_2_2.grad / N; 
    w_2_3.value = w_2_3.value - lr * w_2_3.grad / N; 

    b_2_1.value = b_2_1.value - lr * b_2_1.grad / N; 
end




% 计算结果并绘图

net_pre = zeros(21, 21);
for x1 = -1:0.1:1
    for x2 = -1:0.1:1
        % 乘法节点
        n_1_11.value = x1 * w_1_11.value;  % n1
        n_1_21.value = x2 * w_1_21.value;
        
        n_1_12.value = x1 * w_1_12.value;  % n2
        n_1_22.value = x2 * w_1_22.value;
        
        n_1_13.value = x1 * w_1_13.value;  % n3
        n_1_23.value = x2 * w_1_23.value;

        % 加法节点
        n_1.value = n_1_11.value + n_1_21.value + b_1_1.value;
        n_2.value = n_1_12.value + n_1_22.value + b_1_2.value;
        n_3.value = n_1_13.value + n_1_23.value + b_1_3.value;

        % 激活函数节点
        t_1.value = max(n_1.value, 0);
        t_2.value = max(n_2.value, 0);
        t_3.value = max(n_3.value, 0);
        
        % 第二层乘法节点
        n_2_1.value = t_1.value * w_2_1.value;
        n_2_2.value = t_2.value * w_2_2.value;
        n_2_3.value = t_3.value * w_2_3.value; 

        % 第二层加法节点，最终输出
        y1.value = n_2_1.value + n_2_2.value + n_2_3.value + b_2_1.value;
        
        % 注意行列
        net_pre(round((x2 + 1.1) / 0.1), round((x1 + 1.1) / 0.1)) = y1.value;
    end
end



x = -1:0.1:1;
y = -1:0.1:1;

[X, Y] = meshgrid(x, y);

f_true = f_target(X, Y);



figure(1)
semilogy(train_loss)
xlabel("epoch");
ylabel("Train loss");


figure('Position', [100, 300, 1400, 300])

subplot(1, 3, 1)
heatmap(x, y, f_true, "Colormap", turbo)
title("target function");
xlabel("x_1");
ylabel("x_2");

subplot(1, 3, 2)
heatmap(x, y, net_pre, "Colormap", turbo)
title("prediction");
xlabel("x_1");
ylabel("x_2");

subplot(1, 3, 3)
heatmap(x, y, abs(net_pre - f_true), "Colormap", turbo)
title("absolute error");
xlabel("x_1");
ylabel("x_2");

