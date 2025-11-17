function mse = compute_mse(y_true, y_pred)
%COMPUTE_MSE 计算均方误差。
    residual = y_true - y_pred;
    mse = mean(residual .^ 2);
end
