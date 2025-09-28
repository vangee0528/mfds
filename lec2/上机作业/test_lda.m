function pred = test_lda(X, param)
    % 多维多分类LDA预测函数
    % X: N×M特征矩阵
    % param: fit_lda返回的参数结构体
    
    [N, M] = size(X);
    K = param.K;
    pred = zeros(N, 1);
    
    Sigma_inv = inv(param.Sigma);
    
    for i = 1:N
        x = X(i, :)';  % 当前样本
        scores = zeros(K, 1);
        
        for k = 1:K
            mu_k = param.mu(k, :)';
            % 判别函数 δ_k(x)
            scores(k) = x' * Sigma_inv * mu_k - 0.5 * mu_k' * Sigma_inv * mu_k + log(param.p(k));
        end
        
        [~, max_idx] = max(scores);
        pred(i) = max_idx - 1;  % 标签从0开始
    end
end