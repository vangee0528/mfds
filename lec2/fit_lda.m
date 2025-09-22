function param = fit_lda(data, K)
    % data: 训练数据，最后一列是标签
    % K: 类别数
    
    X = data(:, 1:end-1);
    Y = data(:, end);
    [N, M] = size(X);
    
    mu_hat = zeros(K, M);
    p_hat = zeros(K, 1);
    
    for k = 0:K-1
        X_k = X(Y == k, :);
        N_k = size(X_k, 1);
        mu_hat(k+1, :) = mean(X_k);
        p_hat(k+1) = N_k / N;
    end
    
    Sigma_hat = zeros(M, M);
    for k = 0:K-1
        X_k = X(Y == k, :);
        mu_k = mu_hat(k+1, :);
        for i = 1:size(X_k, 1)
            Sigma_hat = Sigma_hat + (X_k(i, :) - mu_k)' * (X_k(i, :) - mu_k);
        end
    end
    Sigma_hat = Sigma_hat / (N - K);
    
    param.mu = mu_hat;
    param.p = p_hat;
    param.Sigma = Sigma_hat;
end