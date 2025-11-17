function tests_passed = test_linear_reg()
%TEST_LINEAR_REG Basic sanity checks for the OLS implementation.
    X = (1:5)';
    y = 3 * X + 2;

    model = linear_regression(X, y);
    y_pred = model.predict(X);

    assert(abs(model.intercept - 2) < 1e-10, 'Incorrect intercept for perfect linear data');
    assert(norm(model.beta - 3) < 1e-10, 'Incorrect slope for perfect linear data');
    assert(max(abs(y - y_pred)) < 1e-10, 'Predictions should perfectly fit training data');

    tests_passed = true;
end
