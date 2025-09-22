function data = generate_stamps(N, M, K)
    % N: 样本总数
    % M: 特征维度
    % K: 类别数

    % 为每个类别生成均值向量
    mus = randn(K, M) * 2; % 随机生成K个M维的均值向量
    
    % 生成一个共享的协方差矩阵
    A = randn(M, M);
    sigma = A * A' + eye(M); % 确保协方差矩阵是正定的

    data_X = [];
    data_Y = [];
    
    Nk = fix(N / K); % 每个类别的样本数

    for k = 1:K
        % 从多元正态分布中生成数据
        X_k = mvnrnd(mus(k,:), sigma, Nk);
        Y_k = ones(Nk, 1) * (k-1); % 类别标签从0开始
        
        data_X = [data_X; X_k];
        data_Y = [data_Y; Y_k];
    end
    
    % 将特征和标签合并
    data = [data_X, data_Y];
end