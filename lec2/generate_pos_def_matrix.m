function Sigma = generate_pos_def_matrix(M, scale)
    % 生成M维正定协方差矩阵
    % M: 矩阵维度
    % scale: 缩放因子，控制协方差矩阵的大小
    
    if nargin < 2
        scale = 1.0;
    end
    
    % 方法1：使用随机矩阵的Gram矩阵
    A = randn(M, M);
    Sigma = scale * (A * A');
    
    % 添加小的对角项确保正定性
    Sigma = Sigma + 0.01 * eye(M);
    
    % 方法2（备选）：使用特征值分解构造
    % eigenvalues = rand(M, 1) * scale + 0.1; % 确保所有特征值为正
    % [Q, ~] = qr(randn(M, M)); % 正交矩阵
    % Sigma = Q * diag(eigenvalues) * Q';
end