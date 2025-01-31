% read in combined stock and options data
data = readtable('data/sp500_combined_monthly.csv');

% ensure contract symbols are strings
data.contractSymbol = string(data.contractSymbol);

% separate stock and option data
stockAAPL = data.Adj_Close(contains(data.contractSymbol, "AAPL"));
stockNVDA = data.Adj_Close(contains(data.contractSymbol, "NVDA"));
strikeAAPL = data.strike(contains(data.contractSymbol, "AAPL"));
strikeNVDA = data.strike(contains(data.contractSymbol, "NVDA"));
midPriceAAPL = (data.bid(contains(data.contractSymbol, "AAPL")) + data.ask(contains(data.contractSymbol, "AAPL"))) / 2;
midPriceNVDA = (data.bid(contains(data.contractSymbol, "NVDA")) + data.ask(contains(data.contractSymbol, "NVDA"))) / 2;

% initialize Parameters
rf = 0.05; % risk-free rate
a = 0.95; % confidence level
T = 1; % ttm

% simulate stock returns
stockReturnsAAPL = diff(stockAAPL) ./ stockAAPL(1:end-1); % AAPL stock daily returns
stockReturnsNVDA = diff(stockNVDA) ./ stockNVDA(1:end-1); % NVDA stock daily returns

% simulate option returns
optionPayoffAAPL = max(0, stockAAPL(end) - strikeAAPL); % AAPL call payoff at maturity
optionPayoffNVDA = max(0, stockNVDA(end) - strikeNVDA); % NVDA call payoff at maturity
optionReturnsAAPL = (optionPayoffAAPL - midPriceAAPL) ./ midPriceAAPL; % AAPL call returns
optionReturnsNVDA = (optionPayoffNVDA - midPriceNVDA) ./ midPriceNVDA; % NVDA call returns

% include put options
putPayoffAAPL = max(0, strikeAAPL - stockAAPL(end)); % AAPL put payoff
putPayoffNVDA = max(0, strikeNVDA - stockNVDA(end)); % NVDA put payoff
putReturnsAAPL = (putPayoffAAPL - midPriceAAPL) ./ midPriceAAPL; % AAPL put returns
putReturnsNVDA = (putPayoffNVDA - midPriceNVDA) ./ midPriceNVDA; % NVDA put returns

% add short options
shortCallReturnsAAPL = -optionReturnsAAPL;
shortCallReturnsNVDA = -optionReturnsNVDA;
shortPutReturnsAAPL = -putReturnsAAPL;
shortPutReturnsNVDA = -putReturnsNVDA;

% align stock and option data length
minLength = min([length(stockReturnsAAPL), length(stockReturnsNVDA), ...
                 length(optionReturnsAAPL), length(optionReturnsNVDA), ...
                 length(putReturnsAAPL), length(putReturnsNVDA)]);
stockReturnsAAPL = stockReturnsAAPL(1:minLength);
stockReturnsNVDA = stockReturnsNVDA(1:minLength);
optionReturnsAAPL = optionReturnsAAPL(1:minLength);
optionReturnsNVDA = optionReturnsNVDA(1:minLength);
putReturnsAAPL = putReturnsAAPL(1:minLength);
putReturnsNVDA = putReturnsNVDA(1:minLength);
shortCallReturnsAAPL = shortCallReturnsAAPL(1:minLength);
shortCallReturnsNVDA = shortCallReturnsNVDA(1:minLength);
shortPutReturnsAAPL = shortPutReturnsAAPL(1:minLength);
shortPutReturnsNVDA = shortPutReturnsNVDA(1:minLength);

% combine returns into a matrix
returnsMatrix = [stockReturnsAAPL, stockReturnsNVDA, ...
                 optionReturnsAAPL, optionReturnsNVDA, ...
                 shortCallReturnsAAPL, shortCallReturnsNVDA, ...
                 putReturnsAAPL, putReturnsNVDA, ...
                 shortPutReturnsAAPL, shortPutReturnsNVDA];

% remove rows with any NaN or Inf values
returnsMatrix = returnsMatrix(~any(isnan(returnsMatrix) | isinf(returnsMatrix), 2), :);

threshold = 5; 
returnsMatrix = returnsMatrix(all(abs(returnsMatrix) < threshold, 2), :);


% expected returns
E = mean(returnsMatrix, 1);

numPoints = 50;

% target returns forEF
minReturn = min(E);
maxReturn = max(E);
targetReturns = linspace(minReturn, maxReturn, numPoints);

%initialize portfolio stats
portReturn = zeros(numPoints, 1);
portCVAR = zeros(numPoints, 1);
weightMatrix = zeros(numPoints, size(returnsMatrix, 2));

% optimization options
options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp');

% construct EF
for i = 1:numPoints
    targetRet = targetReturns(i);

    objective = @(w) calculateCVAR(returnsMatrix * w, a);

    Aeq = [mean(returnsMatrix); ones(1, size(returnsMatrix, 2))];
    beq = [targetRet; 1];
    lb = zeros(size(returnsMatrix, 2), 1); % No short selling
    ub = ones(size(returnsMatrix, 2), 1); % Maximum weight of 1
    x_initial = ones(size(returnsMatrix, 2), 1) / size(returnsMatrix, 2);


    [optWeights, optCVAR, exitflag] = fmincon(objective, x_initial, [], [], Aeq, beq, lb, ub, [], options);

    % check for successful optimization
    if exitflag <= 0
        warning('Optimization did not converge at target return %.4f.', targetRet);
        continue;
    end

    % store results
    portReturn(i) = targetRet;
    portCVAR(i) = optCVAR;
    weightMatrix(i, :) = optWeights';
end

% remove zero entries where optimization failed
validIndices = portReturn > 0;
portReturn = portReturn(validIndices);
portCVAR = portCVAR(validIndices);
weightMatrix = weightMatrix(validIndices, :);

% calculate Sharpe ratios
sharpeRatios = (portReturn - rf) ./ portCVAR;

% tangency portfolio
[~, idxMaxSharpe] = max(sharpeRatios);
tanRet = portReturn(idxMaxSharpe);
tanCVAR = portCVAR(idxMaxSharpe);
tanWeights = weightMatrix(idxMaxSharpe, :);

% plot EF
figure;
plot(portCVAR, portReturn, 'b-', 'LineWidth', 2);
xlabel('CVaR');
ylabel('Expected Return');
title('Efficient Frontier with AAPL and NVDA Stocks and Options');
grid on;
hold on;

% plot Tangency Portfolio
plot(tanCVAR, tanRet, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

% cml
cmlCVAR = linspace(0, max(portCVAR), numPoints);
cmlSlope = (tanRet - rf) / tanCVAR;
cmlReturns = rf + cmlSlope * cmlCVAR;
plot(cmlCVAR, cmlReturns, 'k--', 'LineWidth', 2);
legend('Efficient Frontier', 'Tangency Portfolio', 'CML', 'Location', 'Best');
ylim([0,max(portReturn)+0.25]);
% Display Tangency Portfolio Weights
fprintf('Tangency Portfolio Weights:\n');
fprintf('AAPL Stock: %.2f%%\n', tanWeights(1) * 100);
fprintf('NVDA Stock: %.2f%%\n', tanWeights(2) * 100);
fprintf('AAPL Call Option: %.2f%%\n', tanWeights(3) * 100);
fprintf('NVDA Call Option: %.2f%%\n', tanWeights(4) * 100);
fprintf('AAPL Short Call Option: %.2f%%\n', tanWeights(5) * 100);
fprintf('NVDA Short Call Option: %.2f%%\n', tanWeights(6) * 100);
fprintf('AAPL Put Option: %.2f%%\n', tanWeights(7) * 100);
fprintf('NVDA Put Option: %.2f%%\n', tanWeights(8) * 100);
fprintf('AAPL Short Put Option: %.2f%%\n', tanWeights(9) * 100);
fprintf('NVDA Short Put Option: %.2f%%\n', tanWeights(10) * 100);
hold off;

%% Function Definition
function CVAR = calculateCVAR(portfolioReturns, a)
    portfolioLosses = -portfolioReturns; % Treat returns as losses for CVaR calculation
    VaR = quantile(portfolioLosses, a); % Value at Risk
    CVAR = mean(portfolioLosses(portfolioLosses >= VaR)); % Conditional VaR
end
