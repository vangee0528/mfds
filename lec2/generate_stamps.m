function data = generate_stamps(N, M, K, mu_list, Sigma)
    % 生成多维多分类的高斯分布数据
    % N: 总样本数
    % M: 特征维度 (默认为2)
    % K: 类别数 (默认为3)
    % mu_list: K×M矩阵，每行为一个类别的均值向量 (可选)
    % Sigma: M×M协方差矩阵，所有类别共享 (可选)
    
    % 设置默认参数
    if nargin < 2
        M = 2; % 默认2维特征
    end
    if nargin < 3
        K = 3; % 默认3个类别
    end
    
    % 如果没有提供均值向量，随机生成
    if nargin < 4 || isempty(mu_list)
        mu_list = zeros(K, M);
        for k = 1:K
            % 在不同区域生成均值，确保类别间有区分度
            angle = 2 * pi * (k-1) / K;
            radius = 3; % 类别中心间的距离
            mu_list(k, :) = radius * [cos(angle), sin(angle)];
            if M > 2
                % 对于高维情况，其他维度随机生成
                mu_list(k, 3:M) = randn(1, M-2) * 2;
            end
        end
    end
    
    % 如果没有提供协方差矩阵，生成一个正定矩阵
    if nargin < 5 || isempty(Sigma)
        Sigma = generate_pos_def_matrix(M, 1.0);
    end
    
    % 为每个类别分配样本数（尽量均匀分配）
    samples_per_class = floor(N / K);
    remaining_samples = N - samples_per_class * K;
    
    % 初始化数据矩阵
    X = zeros(N, M);
    Y = zeros(N, 1);
    
    current_idx = 1;
    
    for k = 1:K
        % 当前类别的样本数
        if k <= remaining_samples
            n_k = samples_per_class + 1;
        else
            n_k = samples_per_class;
        end
        
        % 生成多元高斯分布数据
        % X|Y=k ~ N(mu_k, Sigma)
        X_k = mvnrnd(mu_list(k, :), Sigma, n_k);
        
        % 存储数据和标签
        end_idx = current_idx + n_k - 1;
        X(current_idx:end_idx, :) = X_k;
        Y(current_idx:end_idx) = k - 1; % 标签从0开始
        
        current_idx = end_idx + 1;
    end
    
    % 随机打乱数据顺序
    perm = randperm(N);
    X = X(perm, :);
    Y = Y(perm);
    
    % 返回数据矩阵 [X, Y]
    data = [X, Y];
end