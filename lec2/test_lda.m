function pred = test_lda(X, param)
    % 多维多分类LDA预测函数
    % X: N×M特征矩阵，N个样本，M维特征
    % param: fit_lda函数返回的参数结构体
    % 返回: N×1预测标签向量
    
    [N, M] = size(X);
    K = param.K;
    
    % 初始化预测结果
    pred = zeros(N, 1);
    
    % 计算每个样本的判别函数值
    scores = zeros(N, K);
    
    for i = 1:N
        x = X(i, :)';  % 当前样本，转为列向量
        
        for k = 1:K
            % 计算判别函数 δ_k(x) = x^T Σ^(-1) μ_k - (1/2) μ_k^T Σ^(-1) μ_k + ln(p_k)
            % 使用预计算的参数提高效率
            linear_term = param.linear_coef(k, :) * x;  % x^T Σ^(-1) μ_k
            scores(i, k) = linear_term + param.const_terms(k);
        end
        
        % 选择得分最高的类别
        [~, max_idx] = max(scores(i, :));
        pred(i) = param.class_labels(max_idx);
    end
    
    % 可选：返回每个样本的所有类别得分（用于概率估计）
    % 如果需要概率输出，可以考虑添加softmax变换
end