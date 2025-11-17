function split = data_preprocessor(data, params)
%DATA_PREPROCESSOR Standardize features and create a train/test split.
%
%   INPUTS:
%       data   - struct from data_loader.
%       params - struct from project_parameters.
%
%   OUTPUTS:
%       split struct containing:
%           X_train, y_train, X_test, y_test
%           mu, sigma - standardization statistics
%           feature_names, target_name

    arguments
        data struct
        params struct
    end

    rng(params.random_seed);
    n_samples = size(data.X, 1);
    indices = randperm(n_samples);

    n_train = round(params.train_ratio * n_samples);
    train_idx = indices(1:n_train);
    test_idx = indices(n_train + 1:end);

    X_train = data.X(train_idx, :);
    y_train = data.y(train_idx, :);
    X_test = data.X(test_idx, :);
    y_test = data.y(test_idx, :);

    mu = mean(X_train, 1);
    sigma = std(X_train, 0, 1);
    sigma(sigma == 0) = 1; % avoid divide-by-zero

    if params.standardize
        X_train = (X_train - mu) ./ sigma;
        X_test = (X_test - mu) ./ sigma;
    end

    split = struct();
    split.X_train = X_train;
    split.y_train = y_train;
    split.X_test = X_test;
    split.y_test = y_test;
    split.mu = mu;
    split.sigma = sigma;
    split.feature_names = data.feature_names;
    split.target_name = data.target_name;

    if isfield(params, 'files') && isfield(params.files, 'train_test_split')
        train_test_split = split; %#ok<NASGU>
        save(params.files.train_test_split, 'train_test_split');
    end

    if isfield(params, 'files') && isfield(params.files, 'cleaned_data')
        cleaned_data = struct('X', data.X, 'y', data.y, ...
                              'feature_names', data.feature_names, ...
                              'target_name', data.target_name); %#ok<NASGU>
        save(params.files.cleaned_data, 'cleaned_data');
    end
end
