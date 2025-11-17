function data = prepare_diabetes_data(dataPath, trainRatio, randomSeed)
%PREPARE_DIABETES_DATA 加载并预处理糖尿病数据集。
%   data = PREPARE_DIABETES_DATA(dataPath, trainRatio, randomSeed)
%   - dataPath: CSV文件路径
%   - trainRatio: 训练集占比（默认0.7）
%   - randomSeed: 随机种子（默认42）

    if nargin < 2 || isempty(trainRatio)
        trainRatio = 0.7;
    end
    if nargin < 3 || isempty(randomSeed)
        randomSeed = 42;
    end

    if ~isfile(dataPath)
        error('未找到数据文件：%s', dataPath);
    end

    rawTable = readtable(dataPath);
    variableNames = rawTable.Properties.VariableNames;
    featureNames = variableNames(1:end-1);

    X = rawTable{:, featureNames};
    y = rawTable{:, end};

    rng(randomSeed);
    nSamples = size(X, 1);
    permutation = randperm(nSamples);
    nTrain = floor(trainRatio * nSamples);

    trainIdx = permutation(1:nTrain);
    testIdx = permutation(nTrain+1:end);

    mu = mean(X(trainIdx, :), 1);
    sigma = std(X(trainIdx, :), 0, 1);
    sigma(sigma == 0) = 1;

    X_train = (X(trainIdx, :) - mu) ./ sigma;
    X_test = (X(testIdx, :) - mu) ./ sigma;

    y_train = y(trainIdx);
    y_test = y(testIdx);

    data = struct();
    data.X_train = X_train;
    data.X_test = X_test;
    data.y_train = y_train;
    data.y_test = y_test;
    data.mu = mu;
    data.sigma = sigma;
    data.feature_names = featureNames;
    data.train_indices = trainIdx;
    data.test_indices = testIdx;
    data.random_seed = randomSeed;
end
