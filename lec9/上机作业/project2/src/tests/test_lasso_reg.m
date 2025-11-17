function tests_passed = test_lasso_reg()
%TEST_LASSO_REG Ensure Lasso penalization shrinks coefficients with large lambda.
    rng(1);
    X = randn(200, 4);
    true_beta = [3; -2; 0; 0];
    y = X * true_beta + 0.1 * randn(size(X, 1), 1);

    small_lambda_model = lasso_regression(X, y, 1e-3);
    large_lambda_model = lasso_regression(X, y, 1e2);

    assert(norm(large_lambda_model.beta, 1) < norm(small_lambda_model.beta, 1), ...
        'Lasso with large lambda should yield sparser (smaller) coefficients');

    assert(~isempty(large_lambda_model.predict(X)), 'Predict handle should return values');
    tests_passed = true;
end
