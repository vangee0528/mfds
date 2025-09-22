function param = fit_lda(data)
    % 多维多分类LDA参数估计
    % data: N×(M+1)矩阵，前M列为特征，最后一列为标签
    % 返回: 包含估计参数的结构体
    
    % 分离特征和标签
    X = data(:, 1:end-1);  % N×M特征矩阵
    Y = data(:, end);      % N×1标签向量
    
    [N, M] = size(X);
    K = length(unique(Y));  % 类别数
    
    % 初始化参数
    param = struct();
    param.mu = zeros(K, M);     % K×M均值矩阵
    param.p = zeros(K, 1);      % K×1先验概率向量
    param.Sigma = zeros(M, M);  % M×M共同协方差矩阵
    param.K = K;
    param.M = M;
    
    % 计算各类别的样本数和先验概率
    class_labels = unique(Y);
    N_k = zeros(K, 1);
    
    for k = 1:K
        label = class_labels(k);
        mask = (Y == label);
        N_k(k) = sum(mask);
        
        % 估计均值 μ_k
        param.mu(k, :) = mean(X(mask, :), 1);
        
        % 估计先验概率 p_k
        param.p(k) = N_k(k) / N;
    end
    
    % 估计共同协方差矩阵 Σ
    S_pooled = zeros(M, M);
    for k = 1:K
        label = class_labels(k);
        mask = (Y == label);
        X_k = X(mask, :);
        
        % 计算类内散布矩阵
        mu_k = param.mu(k, :);
        X_centered = X_k - mu_k;  % 中心化
        S_k = X_centered' * X_centered;  % M×M散布矩阵
        
        S_pooled = S_pooled + S_k;
    end
    
    % pooled协方差矩阵
    param.Sigma = S_pooled / (N - K);
    
    % 确保协方差矩阵的数值稳定性
    param.Sigma = param.Sigma + 1e-6 * eye(M);
    
    % 计算判别函数的预计算参数
    param.Sigma_inv = inv(param.Sigma);
    
    % 预计算各类别判别函数的常数项
    param.const_terms = zeros(K, 1);
    for k = 1:K
        mu_k = param.mu(k, :)';
        param.const_terms(k) = -0.5 * mu_k' * param.Sigma_inv * mu_k + log(param.p(k));
    end
    
    % 预计算线性项系数
    param.linear_coef = (param.Sigma_inv * param.mu')';  % K×M矩阵
    
    % 存储类别标签映射
    param.class_labels = class_labels;
    
    fprintf('LDA模型训练完成:\n');
    fprintf('  特征维度: %d\n', M);
    fprintf('  类别数: %d\n', K);
    fprintf('  样本数: %d\n', N);
    for k = 1:K
        fprintf('  类别 %d: %d 个样本 (%.2f%%)\n', class_labels(k), N_k(k), param.p(k)*100);
    end
end