function tests_passed = test_ridge_reg()
%TEST_RIDGE_REG Validate ridge solution matches manual closed-form result.
    X = (1:4)';
    y = 2 * X + 1;
    lambda = 0.5;

    model = ridge_regression(X, y, lambda);

    X_aug = [ones(size(X, 1), 1), X];
    reg = lambda * eye(2); reg(1, 1) = 0;
    theta = (X_aug' * X_aug + reg) \ (X_aug' * y);

    assert(abs(model.intercept - theta(1)) < 1e-10, 'Ridge intercept mismatch');
    assert(abs(model.beta - theta(2)) < 1e-10, 'Ridge slope mismatch');

    y_pred = model.predict(X);
    assert(all(abs(y_pred - (model.intercept + model.beta * X)) < 1e-12), ...
        'Predict handle inconsistent with closed form');

    tests_passed = true;
end
