c = 0.5;  % 波速

% 边界条件: u(0,t) = 0, u(1,t) = 0
numBoundaryConditionPoints = [30 30];

x0BC1 = zeros(1,numBoundaryConditionPoints(1));
x0BC2 = ones(1,numBoundaryConditionPoints(2));

t0BC1 = linspace(0,2,numBoundaryConditionPoints(1));
t0BC2 = linspace(0,2,numBoundaryConditionPoints(2));

u0BC1 = zeros(1,numBoundaryConditionPoints(1));
u0BC2 = zeros(1,numBoundaryConditionPoints(2));

% 初始条件: u(x,0) = sin(pi*x)
numInitialConditionPoints = 50;

x0IC = linspace(0,1,numInitialConditionPoints);
t0IC = zeros(1,numInitialConditionPoints);
u0IC = sin(pi*x0IC);

X0 = [x0IC x0BC1 x0BC2];
T0 = [t0IC t0BC1 t0BC2];
U0 = [u0IC u0BC1 u0BC2];

numInternalCollocationPoints = 8000;

points = rand(numInternalCollocationPoints,2);

dataX = points(:,1);  % x in [0,1]
dataT = 2*points(:,2);  % t in [0,2]
numBlocks = 6;
fcOutputSize = 32;
fcBlock = [
    fullyConnectedLayer(fcOutputSize)
    tanhLayer];

layers = [
    featureInputLayer(2)
    repmat(fcBlock,[numBlocks 1])
    fullyConnectedLayer(1)]
net = dlnetwork(layers)
function [loss,gradients] = modelLoss(net,X,T,X0,T0,U0,c)

% 使用内点进行预测
XT = cat(1,X,T);
U = forward(net,XT);

% 计算关于 X 和 T 的导数
X = stripdims(X);
T = stripdims(T);
U = stripdims(U);

%  一阶导数
Ux = dljacobian(U,X,1);
Ut = dljacobian(U,T,1);

% 二阶导数
Uxx = dldivergence(Ux,X,1);
Utt = dldivergence(Ut,T,1);

% 计算 mseF - 强制波动方程: Utt - c^2*Uxx = 0
f = Utt - (c^2).*Uxx;
mseF = mean(f.^2);

% 计算 mseU - 强制初始和边界条件
XT0 = cat(1,X0,T0);
U0Pred = forward(net,XT0);
mseU = l2loss(U0Pred,U0);

loss = mseF + mseU;

% 计算梯度
gradients = dlgradient(loss,net.Learnables);

end

solverState = lbfgsState;
maxIterations = 2000;
gradientTolerance = 1e-5;
stepTolerance = 1e-5;

X = dlarray(dataX,"BC");
T = dlarray(dataT,"BC");
X0 = dlarray(X0,"CB");
T0 = dlarray(T0,"CB");
U0 = dlarray(U0,"CB");

accfun = dlaccelerate(@modelLoss);

lossFcn = @(net) dlfeval(accfun,net,X,T,X0,T0,U0,c);

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info=["Iteration" "GradientsNorm" "StepNorm"], ...
    XLabel="Iteration");

iteration = 0;
while iteration < maxIterations && ~monitor.Stop
    iteration = iteration + 1;
    [net, solverState] = lbfgsupdate(net,lossFcn,solverState);

    updateInfo(monitor, ...
        Iteration=iteration, ...
        GradientsNorm=solverState.GradientsNorm, ...
        StepNorm=solverState.StepNorm);

    recordMetrics(monitor,iteration,TrainingLoss=solverState.Loss);

    monitor.Progress = 100*iteration/maxIterations;

    if solverState.GradientsNorm < gradientTolerance || ...
            solverState.StepNorm < stepTolerance || ...
            solverState.LineSearchStatus == "failed"
        break
    end

end

% 生成测试数据
x = 0:0.01:1;  % x 从 0 到 1
t = 0:0.01:2;  % t 从 0 到 2
[X, T] = meshgrid(x, t);

% 获取预测结果
szXTest = numel(x);
szTTest = numel(t);

UPred = zeros(szTTest, szXTest);
UAnalytical = zeros(szTTest, szXTest);

% 创建测试网格的 dlarray 格式
for i = 1:szTTest
    t_val = t(i);
    X_test_row = dlarray(x, "CB");
    T_test_row = dlarray(repmat(t_val, 1, szXTest), "CB");
    XT_test = cat(1, X_test_row, T_test_row);
    
    UPred(i,:) = extractdata(forward(net, XT_test))';
    
    % 解析解: u(x,t) = sin(pi*x)*cos(pi*c*t)
    UAnalytical(i,:) = sin(pi*x).*cos(pi*c*t_val);
end

% 计算误差
err = norm(UPred - UAnalytical) / norm(UAnalytical);
fprintf('相对误差: %.6e\n', err);

% 保存预测结果为工作区变量和 mat 文件
save('3220102895.mat', 'UPred');

% 创建可视化图形
figure('Position', [100 100 1200 800]);
tiledlayout(2, 2, 'TileSpacing', 'compact');

% 图1: 不同时刻的预测值
nexttile
hold on
for i = 1:4:size(UPred,1)
    plot(x, UPred(i,:), 'LineWidth', 1.5, 'DisplayName', sprintf('t=%.1f', t(i)));
end
hold off
xlabel('x');
ylabel('u(x,t)');
title('PINN Predictions at Different Times');
% legend;
grid on;

% 图2: 不同时刻的解析解
nexttile
hold on
for i = 1:4:size(UAnalytical,1)
    plot(x, UAnalytical(i,:), 'LineWidth', 1.5, 'DisplayName', sprintf('t=%.1f', t(i)));
end
hold off
xlabel('x');
ylabel('u(x,t)');
title('Analytical Solution at Different Times');
% legend;
grid on;

% 图3: PINN 预测的热图
nexttile
imagesc(x, t, UPred);
colorbar;
xlabel('x');
ylabel('t');
title('PINN Predictions (Heatmap)');
axis xy;

% 图4: 误差热图
nexttile
error_map = abs(UPred - UAnalytical);
imagesc(x, t, error_map);
colorbar;
xlabel('x');
ylabel('t');
title('Prediction Error (Heatmap)');
axis xy;

% 保存图形
saveas(gcf, 'wave_equation_results.png');
fprintf('图形已保存为 wave_equation_results.png\n');
fprintf('结果已保存为 3220102895.mat\n');