function y_pred = predict_regression(model, X)
%PREDICT_REGRESSION 使用统一的线性模型结构进行预测。
    y_pred = model.intercept + X * model.beta;
end
