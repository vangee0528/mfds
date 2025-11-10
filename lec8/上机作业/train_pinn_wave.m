%% PINN training for 1D wave equation with c = 0.5
% Solves u_tt - c^2 u_xx = 0 on x in [0,1], t in [0,2]
clear; clc; close all;

rng(42, 'twister');

%% Problem setup
c = 0.5;
numCollocation = 8000;
numIC = 128;
numICVelocity = numIC;
numBC = 128;
maxEpochs = 4000;
learningRate = 1e-3;
displayPeriod = 100;

%% Generate collocation points (interior)
xColloc = rand(1, numCollocation);
tColloc = 2 * rand(1, numCollocation);

%% Initial condition points for displacement u(x,0) = sin(pi x)
xIC = linspace(0, 1, numIC);
tIC = zeros(1, numIC);
uIC = sin(pi * xIC);

%% Initial condition points for velocity u_t(x,0) = 0
xICVel = linspace(0, 1, numICVelocity);
tICVel = zeros(1, numICVelocity);

%% Boundary condition points u(0,t) = u(1,t) = 0
% sample the same number of points on each boundary edge
xBC = [zeros(1, numBC), ones(1, numBC)];
tBC = linspace(0, 2, numBC);
tBC = [tBC, tBC];

%% Convert data to dlarray format
xCollocDL = dlarray(xColloc, 'CB');
tCollocDL = dlarray(tColloc, 'CB');

xICDL = dlarray(xIC, 'CB');
tICDL = dlarray(tIC, 'CB');
uICDL = dlarray(uIC, 'CB');

xICVelDL = dlarray(xICVel, 'CB');
tICVelDL = dlarray(tICVel, 'CB');

xBCDL = dlarray(xBC, 'CB');
tBCDL = dlarray(tBC, 'CB');
uBCDL = dlarray(zeros(size(xBC)), 'CB');

%% Build neural network
numHiddenUnits = 40;
numHiddenLayers = 6;

layers = [
    featureInputLayer(2, Normalization='none', Name='input')
];

for k = 1:numHiddenLayers
    layers = [layers;
        fullyConnectedLayer(numHiddenUnits, Name=sprintf('fc_%d', k));
        tanhLayer(Name=sprintf('tanh_%d', k));
    ];
end

layers = [layers;
    fullyConnectedLayer(1, Name='output')
];

net = dlnetwork(layers);
net = dlupdate(@double, net);

%% Training loop (Adam)
trailingAvg = [];
trailingAvgSq = [];

monitorFig = figure('Name', 'Training Progress', 'NumberTitle', 'off');
lossAxis = axes('Parent', monitorFig);
hold(lossAxis, 'on'); grid(lossAxis, 'on');
xlabel(lossAxis, 'Epoch'); ylabel(lossAxis, 'Loss value');

lossHistory = zeros(maxEpochs, 4);

for epoch = 1:maxEpochs
    [lossVal, gradients, componentLoss] = dlfeval(@modelLoss, net, ...
        xCollocDL, tCollocDL, ...
        xICDL, tICDL, uICDL, ...
        xICVelDL, tICVelDL, ...
        xBCDL, tBCDL, uBCDL, c);

    [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, ...
        trailingAvg, trailingAvgSq, epoch, learningRate, 0.9, 0.999);

    lossHistory(epoch, :) = gather([extractdata(lossVal), ...
        extractdata(componentLoss.pde), ...
        extractdata(componentLoss.ic), ...
        extractdata(componentLoss.bc + componentLoss.icVel)]);

    if mod(epoch, displayPeriod) == 0 || epoch == 1
        fprintf('Epoch %4d | Loss %.3e | PDE %.3e | IC %.3e | IC_t %.3e | BC %.3e\n', ...
            epoch, extractdata(lossVal), extractdata(componentLoss.pde), ...
            extractdata(componentLoss.ic), extractdata(componentLoss.icVel), ...
            extractdata(componentLoss.bc));
    end

    if isgraphics(monitorFig)
        plot(lossAxis, epoch, extractdata(lossVal), '.k');
        drawnow limitrate;
    end
end

%% Evaluate on test grid and save results
xVec = 0:0.01:1;
tVec = 0:0.01:2;
[XGrid, TGrid] = meshgrid(xVec, tVec);

XTTest = dlarray([XGrid(:)'; TGrid(:)'], 'CB');
UPred = forward(net, XTTest);
UPred = reshape(extractdata(UPred), size(XGrid));
UPred = double(UPred);

save('3220102895.mat', 'UPred');

%% Optional: compare against analytical solution
UTrue = sin(pi * XGrid) .* cos(pi * c * TGrid);
relErr = norm(UPred(:) - UTrue(:)) / norm(UTrue(:));
fprintf('Relative L2 error on evaluation grid: %.3e\n', relErr);

figure('Name', 'Prediction vs Truth');
subplot(1,3,1);
imagesc(xVec, tVec, UTrue);
set(gca, 'YDir', 'normal');
colorbar; title('Analytical solution'); xlabel('x'); ylabel('t');

subplot(1,3,2);
imagesc(xVec, tVec, UPred);
set(gca, 'YDir', 'normal');
colorbar; title('PINN prediction'); xlabel('x');

subplot(1,3,3);
imagesc(xVec, tVec, UPred - UTrue);
set(gca, 'YDir', 'normal');
colorbar; title('Prediction error'); xlabel('x');

grams = gather(lossHistory);
figure('Name', 'Loss Breakdown');
plot(1:maxEpochs, grams(:,1), 'k', 'LineWidth', 1.2); hold on;
plot(1:maxEpochs, grams(:,2), 'r--', 'LineWidth', 1.0);
plot(1:maxEpochs, grams(:,3), 'b-.', 'LineWidth', 1.0);
plot(1:maxEpochs, grams(:,4), 'g:', 'LineWidth', 1.0);
legend('Total', 'PDE', 'IC', 'IC_t + BC'); grid on;
xlabel('Epoch'); ylabel('Loss components');

%% Local functions
function [lossVal, gradients, compLoss] = modelLoss(net, xInt, tInt, ...
    xIC, tIC, uIC, xICVel, tICVel, xBC, tBC, uBC, c)

    residual = waveResidual(net, xInt, tInt, c);
    msePDE = mean(residual .^ 2, 'all');

    uPredIC = forward(net, [xIC; tIC]);
    mseIC = mean((uPredIC - uIC) .^ 2, 'all');

    [~, ~, utIC] = modelGradients(net, xICVel, tICVel);
    mseICVel = mean(utIC .^ 2, 'all');

    uPredBC = forward(net, [xBC; tBC]);
    mseBC = mean((uPredBC - uBC) .^ 2, 'all');

    lossVal = msePDE + mseIC + mseICVel + mseBC;
    gradients = dlgradient(lossVal, net.Learnables);

    compLoss = struct('pde', msePDE, 'ic', mseIC, 'icVel', mseICVel, 'bc', mseBC);
end

function residual = waveResidual(net, x, t, c)
    [~, ~, ~, uxx, utt] = modelGradients(net, x, t);
    residual = utt - (c^2) * uxx;
end

function [u, ux, ut, uxx, utt] = modelGradients(net, x, t)
    XT = [x; t];
    u = forward(net, XT);

    sumU = sum(u, 'all');
    ux = dlgradient(sumU, x, 'EnableHigherDerivatives', true);
    ut = dlgradient(sumU, t, 'EnableHigherDerivatives', true);

    sumUx = sum(ux, 'all');
    uxx = dlgradient(sumUx, x);

    sumUt = sum(ut, 'all');
    utt = dlgradient(sumUt, t);
end
