function pred = test_lda(X, param, K)
    % X: 测试数据特征
    % param: fit_lda返回的模型参数
    % K: 类别数

    N = size(X, 1);
    scores = zeros(N, K);
    Sigma_inv = inv(param.Sigma);

    for k = 0:K-1
        mu_k = param.mu(k+1, :);
        p_k = param.p(k+1);
        
        % 计算判别函数 delta_k(x)
        % delta_k(x) = x' * inv(Sigma) * mu_k - 0.5 * mu_k' * inv(Sigma) * mu_k + log(p_k)
        term1 = X * Sigma_inv * mu_k';
        term2 = -0.5 * mu_k * Sigma_inv * mu_k';
        term3 = log(p_k);
        
        scores(:, k+1) = term1 + term2 + term3;
    end
    
    % 找到每个样本得分最高的类别
    [~, pred] = max(scores, [], 2);
    pred = pred - 1; % 类别标签从0开始
end