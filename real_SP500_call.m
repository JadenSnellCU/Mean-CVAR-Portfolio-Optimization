clear all;
close all;
clc;
% Read in combined stock and options data
data = readtable('sp500_filtered.csv');

% Separate data for stocks and options
stockAAPL = data.Adj_Close(contains(data.contractSymbol, "AAPL"));
stockNVDA = data.Adj_Close(contains(data.contractSymbol, "NVDA"));
strikeAAPL = data.strike(contains(data.contractSymbol, "AAPL"));
strikeNVDA = data.strike(contains(data.contractSymbol, "NVDA"));
midPriceAAPL = (data.bid(contains(data.contractSymbol, "AAPL")) + data.ask(contains(data.contractSymbol, "AAPL"))) / 2;
midPriceNVDA = (data.bid(contains(data.contractSymbol, "NVDA")) + data.ask(contains(data.contractSymbol, "NVDA"))) / 2;

% Initialize Parameters
rf = 0.005; % Risk-free rate
a = 0.95; % Confidence level
T = 1; % Time to maturity

% Simulate Stock Returns
stockReturnsAAPL = diff(stockAAPL) ./ stockAAPL(1:end-1); % AAPL stock daily returns
stockReturnsNVDA = diff(stockNVDA) ./ stockNVDA(1:end-1); % NVDA stock daily returns

% Simulate Option Returns
optionPayoffAAPL = max(0, stockAAPL(end) - strikeAAPL); % AAPL call payoff at maturity
optionPayoffNVDA = max(0, stockNVDA(end) - strikeNVDA); % NVDA call payoff at maturity
optionReturnsAAPL = (optionPayoffAAPL - midPriceAAPL) ./ midPriceAAPL; % AAPL call returns
optionReturnsNVDA = (optionPayoffNVDA - midPriceNVDA) ./ midPriceNVDA; % NVDA call returns

% Align stock and option data length
minLength = min([length(stockReturnsAAPL), length(stockReturnsNVDA), length(optionReturnsAAPL), length(optionReturnsNVDA)]);
stockReturnsAAPL = stockReturnsAAPL(1:minLength);
stockReturnsNVDA = stockReturnsNVDA(1:minLength);
optionReturnsAAPL = optionReturnsAAPL(1:minLength);
optionReturnsNVDA = optionReturnsNVDA(1:minLength);

% Combine returns into a matrix
returnsMatrix = [stockReturnsAAPL, stockReturnsNVDA, optionReturnsAAPL, optionReturnsNVDA];

% Expected returns
E = mean(returnsMatrix, 1);

% Number of Efficient Frontier points
numPoints = 50;

% Target returns for Efficient Frontier
minReturn = min(E);
maxReturn = max(E);
targetReturns = linspace(minReturn, maxReturn, numPoints);

% Initialize portfolio stats
portReturn = zeros(numPoints, 1);
portCVAR = zeros(numPoints, 1);
weightMatrix = zeros(numPoints, size(returnsMatrix, 2));

% Optimization options
options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp');

% Construct Efficient Frontier
for i = 1:numPoints
    targetRet = targetReturns(i);

    % Objective: Minimize CVaR
    objective = @(w) calculateCVAR(returnsMatrix * w, a);

    % Constraints: Target return, sum of weights = 1
    Aeq = [mean(returnsMatrix); ones(1, size(returnsMatrix, 2))];
    beq = [targetRet; 1];
    lb = zeros(size(returnsMatrix, 2), 1); % No short selling
    ub = ones(size(returnsMatrix, 2), 1); % Maximum weight of 1
    x_initial = ones(size(returnsMatrix, 2), 1) / size(returnsMatrix, 2);

    % Solve optimization problem
    [optWeights, optCVAR, exitflag] = fmincon(objective, x_initial, [], [], Aeq, beq, lb, ub, [], options);

    % Check for successful optimization
    if exitflag <= 0
        warning('Optimization did not converge at target return %.4f.', targetRet);
        continue;
    end

    % Store results
    portReturn(i) = targetRet;
    portCVAR(i) = optCVAR;
    weightMatrix(i, :) = optWeights';
end

% Remove zero entries where optimization failed
validIndices = portReturn > 0;
portReturn = portReturn(validIndices);
portCVAR = portCVAR(validIndices);
weightMatrix = weightMatrix(validIndices, :);

% Calculate Sharpe Ratios
sharpeRatios = (portReturn - rf) ./ portCVAR;

% Tangency Portfolio
[~, idxMaxSharpe] = max(sharpeRatios);
tanRet = portReturn(idxMaxSharpe);
tanCVAR = portCVAR(idxMaxSharpe);
tanWeights = weightMatrix(idxMaxSharpe, :);

% Plot Efficient Frontier
figure;
plot(portCVAR, portReturn, 'b-', 'LineWidth', 2);
xlabel('CVaR');
ylabel('Expected Return');
title('Efficient Frontier with AAPL and NVDA Stocks and Options');
grid on;
hold on;

% Plot Tangency Portfolio
plot(tanCVAR, tanRet, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

% Capital Market Line (CML)
cmlCVAR = linspace(0, max(portCVAR), numPoints);
cmlSlope = (tanRet - rf) / tanCVAR;
cmlReturns = rf + cmlSlope * cmlCVAR;
plot(cmlCVAR, cmlReturns, 'k--', 'LineWidth', 2);
legend('Efficient Frontier', 'Tangency Portfolio', 'CML', 'Location', 'Best');

% Display Tangency Portfolio Weights
fprintf('Tangency Portfolio Weights:\n');
fprintf('AAPL Stock: %.2f%%\n', tanWeights(1) * 100);
fprintf('NVDA Stock: %.2f%%\n', tanWeights(2) * 100);
fprintf('AAPL Call Option: %.2f%%\n', tanWeights(3) * 100);
fprintf('NVDA Call Option: %.2f%%\n', tanWeights(4) * 100);
hold off;

%% Function Definition
function CVAR = calculateCVAR(portfolioReturns, a)
    portfolioLosses = -portfolioReturns; % Treat returns as losses for CVaR calculation
    VaR = quantile(portfolioLosses, a); % Value at Risk
    CVAR = mean(portfolioLosses(portfolioLosses >= VaR)); % Conditional VaR
end
