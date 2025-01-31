% Initialize parameters
rf = 0.05; 
mu = [0.12, 0.1]; 
sigma = [0.25, 0.21];
rho = 0.3; % correlation
numPoints = 100; 
weights = linspace(0, 1, numPoints); % vary weights for efficient frontier
portReturn = zeros(1, numPoints);
portRisk = zeros(1, numPoints);

% create cov matrix
cov = [sigma(1)^2, rho * sigma(1) * sigma(2);
       rho * sigma(1) * sigma(2), sigma(2)^2];

% portoflio function
function [portReturn, portRisk] = portStats(weights, mu, cov)
    % calculate portfolio return and risk
    portReturn = weights * mu';
    portRisk = sqrt(weights * cov * weights');
end

% populate portfolio arrays
for i = 1:numPoints
    w1 = weights(i);
    w2 = 1 - w1;
    [ret, risk] = portStats([w1, w2], mu, cov);
    portReturn(i) = ret;
    portRisk(i) = risk;
end

% efficient frontier plot
figure;
plot(portRisk, portReturn, 'b-', 'LineWidth', 3);
xlabel('Risk');
ylabel('Return');
title('Two-Stock Portfolio Analysis');

% sharpe ratio function
function sharpeRatio = SharpeRatio(weights, mu, cov, rf)
    % calculate neg Sharpe ratio
    [portReturn, portRisk] = portStats(weights, mu, cov);
    sharpeRatio = -(portReturn - rf) / portRisk;
end

% weight constraints
Aeq = [1, 1];
beq = 1;
lb = [0, 0];
ub = [1, 1];
x_initial = [0.5, 0.5];

% optimize for tangency portfolio
optimize = optimoptions('fmincon', 'Display', 'off');
[tanWeights, ~] = fmincon(@(w) SharpeRatio(w, mu, cov, rf), x_initial, [], [], Aeq, beq, lb, ub, [], optimize);
[tanRet, tanRisk] = portStats(tanWeights, mu, cov);

% plot tangency portfolio
hold on;
plot(tanRisk, tanRet, 'r*', 'MarkerSize', 10);

% cml
cmlRisks = linspace(0, max(portRisk), numPoints);
cmlReturns = rf + (tanRet - rf) / tanRisk * cmlRisks;

% plot cml
plot(cmlRisks, cmlReturns, 'k--', 'LineWidth', 2);
legend('Efficient Frontier', 'Tangency Portfolio', 'CML', 'Location', 'Best');
hold off;
