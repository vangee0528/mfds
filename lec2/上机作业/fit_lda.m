function param = fit_lda(data)
    % 多维多分类LDA参数估计
    X = data(:, 1:end-1);  
    Y = data(:, end);     
    
    [N, M] = size(X);
    K = length(unique(Y));  
    
    % 估计各类别均值
    mu = zeros(K, M);
    p = zeros(K, 1);
    
    for k = 1:K
        class_k = (Y == k-1);  
        mu(k, :) = mean(X(class_k, :));
        p(k) = sum(class_k) / N;  
    end
    
    % 估计共同协方差矩阵
    S = zeros(M, M);
    for k = 1:K
        class_k = (Y == k-1);
        X_k = X(class_k, :);
        X_centered = X_k - mu(k, :);
        S = S + X_centered' * X_centered;
    end
    Sigma = S / (N - K);
    
    param.mu = mu;
    param.Sigma = Sigma;
    param.p = p;
    param.K = K;
    param.M = M;
end