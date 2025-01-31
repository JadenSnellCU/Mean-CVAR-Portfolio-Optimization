%% Mean-CVaR Portfolio Optimization with Stocks and Options
clear all;
close all;
clc;
% initialize Parameters
rf = 0.05; 
S1 = 100; S2 = 100; % stock prices
K1 = 100; K2 = 100; % strike prices of options
T = 1; % ttm
sigma1 = 0.2; sigma2 = 0.25; % volatility
mu1 = 0.1; mu2 = 0.2; % expected returns of the stocks
rho = 0.02; % correlation between the stocks
a = 0.9; % confidence level
initial = 10000; % initial investment amount

% calculate option prices using Black-Scholes Formula
[c1,p1] = blsprice(S1, K1, rf, T, sigma1);
[c2,p2] = blsprice(S2, K2, rf, T, sigma2);

% generate correlated random returns for stocks
N = 10000; % number of simulations
mu_vec = [0; 0];
Sigma = [1, rho; rho, 1];
R = mvnrnd(mu_vec, Sigma, N);
Z1 = R(:,1);
Z2 = R(:,2);

% simulate stock prices and calculate returns
S_T1 = S1 * exp((mu1 - 0.5 * sigma1^2) * T + sigma1 * sqrt(T) .* Z1); %geometric Brownian motion (dS_t = muS_tdt + sigmaS_tdW_t)
S_T2 = S2 * exp((mu2 - 0.5 * sigma2^2) * T + sigma2 * sqrt(T) .* Z2);
stock_return1 = (S_T1 - S1) / S1;
stock_return2 = (S_T2 - S2) / S2;

%simulatie options and retuns
long_call_payoff1 = max(0, S_T1 - K1);
long_call_return1 = (long_call_payoff1 - c1) / c1;
long_call_payoff2 = max(0, S_T2 - K2);
long_call_return2 = (long_call_payoff2 - c2) / c2;

short_call_payoff1 = -max(0, S_T1 - K1);
short_call_return1 = -long_call_return1;
short_call_payoff2 = -max(0, S_T2 - K2);
short_call_return2 = -long_call_return2;

long_put_payoff1 = max(0, K1 - S_T1);
long_put_return1 = (long_put_payoff1 - p1) / p1;
long_put_payoff2 = max(0, K2 - S_T2);
long_put_return2 = (long_put_payoff2 - p2) / p2;

short_put_payoff1 = -max(0, K1 - S_T1);
short_put_return1 = -long_put_return1;
short_put_payoff2 = -max(0, K2 - S_T2);
short_put_return2 = -long_put_return2;


% expected returns array
E = [mean(stock_return1), mean(stock_return2), mean(long_call_return1), mean(long_call_return2),mean(short_call_return1), mean(short_call_return2),mean(long_put_return1), mean(long_put_return2),mean(short_put_return1), mean(short_put_return2), rf];

% number of efficient frontier points
numPoints = 50;

%targetr returns for EF
disp(E(1:10));
minReturn = min(E(1:10)); 
maxReturn = max(E(1:10)); 
targetReturns = linspace(minReturn, maxReturn, numPoints);

% port stats
portReturn = zeros(numPoints, 1);
portCVAR = zeros(numPoints, 1);
weightMatrix = zeros(numPoints, 10);

% returns matrix
returnsMatrix = [stock_return1, stock_return2, long_call_return1, long_call_return2, ...
    short_call_return1, short_call_return2, long_put_return1, long_put_return2, ...
    short_put_return1, short_put_return2];

disp(returnsMatrix);
options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp');

%construct EF
for i = 1:numPoints
    targetRet = targetReturns(i);
    
    objective = @(w) calculateCVAR(returnsMatrix * w, a);
    
    %constraints
    Aeq = [mean(returnsMatrix); ones(1, 10)]; % Expected return equals target, sum of weights equals 1
    beq = [targetRet; 1];
    lb = zeros(10,1);
    ub = ones(10,1);
    x_initial = ones(10,1) /10; 

    [optWeights, optCVAR, exitflag, output] = fmincon(objective, x_initial, [], [], Aeq, beq, lb, ub, [], options);
    
    % check if optimization was successful
    if exitflag <= 0
        warning('Optimization did not converge at target return %.4f.', targetRet);
        continue;
    end
    
    portReturn(i) = targetRet;
    portCVAR(i) = abs(optCVAR);
    weightMatrix(i, :) = optWeights';
end

% remove zero entries in case optimization did not converge for some points
validIndices = portReturn > 0;
portReturn = portReturn(validIndices);
portCVAR = portCVAR(validIndices);
weightMatrix = weightMatrix(validIndices, :);

sharpeRatios = (portReturn - rf) ./ portCVAR;

[~, idxMaxSharpe] = max(sharpeRatios);
tanRet = portReturn(idxMaxSharpe);
tanCVAR = portCVAR(idxMaxSharpe);
tanWeights = weightMatrix(idxMaxSharpe, :);

% plot EF
figure;
plot(portCVAR, portReturn, 'b-', 'LineWidth', 2);
xlim([0,min(max(portCVAR),2)]);
xlabel('CVaR');
ylabel('Expected Return');
title('Efficient Frontier with Two Stocks and Call Options');
grid on;
hold on;

% plot tangency portfolio
plot(tanCVAR, tanRet, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

% CML calculation and plot
cmlCVAR = linspace(0, max(portCVAR), numPoints);
cmlSlope = (tanRet - rf) / tanCVAR;
cmlReturns = rf + cmlSlope * cmlCVAR;
plot(cmlCVAR, cmlReturns, 'k--', 'LineWidth', 2);
legend('Efficient Frontier', 'Tangency Portfolio', 'CML', 'Location', 'Best');
%ylim([0,max(portReturn)]);
% displaying weights
fprintf('Tangency Portfolio Weights:\n');
fprintf('Weight in Stock 1: %.2f%%\n', tanWeights(1) * 100);
fprintf('Weight in Stock 2: %.2f%%\n', tanWeights(2) * 100);
fprintf('Weight in Long Call 1: %.2f%%\n', tanWeights(3) * 100);
fprintf('Weight in Long Call 2: %.2f%%\n', tanWeights(4) * 100);
fprintf('Weight in Short Call 1: %.2f%%\n', tanWeights(5) * 100);
fprintf('Weight in Short Call 2: %.2f%%\n', tanWeights(6) * 100);
fprintf('Weight in Long Put 1: %.2f%%\n', tanWeights(7) * 100);
fprintf('Weight in Long Put 2: %.2f%%\n', tanWeights(8) * 100);
fprintf('Weight in Short Put 1: %.2f%%\n', tanWeights(9) * 100);
fprintf('Weight in Short Put 2: %.2f%%\n', tanWeights(10) * 100);

hold off;

%% Function Definition

function CVAR = calculateCVAR(portfolioReturns, a)
    portfolioLosses = -portfolioReturns; 
    VaR = quantile(portfolioLosses, a);
    CVAR = mean(portfolioLosses(portfolioLosses >= VaR));
end
