function data = generate_stamps(N)
    % 生成多维多分类的高斯分布数据
    % N: 总样本数
    
    % 参数设置
    M = 3;  % 特征维度
    K = 3;  % 类别数
    
    % 设置各类别的均值向量
    mu = [0, 0, 0;      % 类别0
          3, 0, 0;      % 类别1  
          0, 3, 0];     % 类别2
    
    % 共同协方差矩阵（正定）
    Sigma = [1.0, 0.3, 0.1;
             0.3, 1.0, 0.2;
             0.1, 0.2, 1.0];

    samples_per_class = floor(N / K);
    
    data = [];
    
    for k = 1:K
        X_k = mvnrnd(mu(k, :), Sigma, samples_per_class);
        Y_k = (k-1) * ones(samples_per_class, 1);  
        
        data = [data; X_k, Y_k];
    end

    perm = randperm(size(data, 1));
    data = data(perm, :);
end