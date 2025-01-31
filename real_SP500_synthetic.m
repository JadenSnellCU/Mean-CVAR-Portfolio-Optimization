clear all;
close all;
clc;

% read in combined stock and options data
data = readtable('data/synthetic_options_data_yearly.csv');

% convert data into processible data
data.Data = datetime(data.Date);
data.Expiration = datetime(data.Expiration);
data.Ticker = string(data.Ticker);
data.Type = string(data.Type);
data = sortrows(data, 'Date');

%identity unique ticker
uniqueTickers = unique(data.Ticker);

allOptionsandStocks = {};
allReturns = [];
opandStockNames = [];

for ticker = uniqueTickers'
    ticker_data = data(data.Ticker == ticker,:);

    %stock returns - puts and calls have the same adj_close
    [~,row] = unique(ticker_data.Date);
    stock_prices = ticker_data.Adj_Close(row);
    stock_dates = ticker_data.Date(row);

    %daily returns
    stock_returns = diff(stock_prices) ./ stock_prices(1:end-1);
    stock_dates = stock_dates(2:end);
  
    %disp(size(stock_returns));
    %disp(size(stock_dates));
    %disp(ticker);
    allOptionsandStocks = [allOptionsandStocks;{char(ticker+"_Stock")}];
    opandStockNames = [opandStockNames; ticker+"_Stock"];

    stockTimeTable = timetable(stock_dates,stock_returns,'VariableNames',{char(ticker+"_Stock")});
    uniqueContracts= unique([string(ticker_data.Expiration), string(ticker_data.Strike), ticker_data.Type], 'rows');
    optionTimeTable = [];
    for i = 1:size(uniqueContracts,1)
        oExp = uniqueContracts(i,1);
        oStr = uniqueContracts(i,2);
        oType = uniqueContracts(i,3);
        oData = ticker_data(string(ticker_data.Expiration)==oExp & string(ticker_data.Strike)==oStr &  string(ticker_data.Type)==oType,:);
        oData = sortrows(oData,'Date');
        %disp(length(oData.Adj_Close));
        if length(oData.Adj_Close) > 1
            %disp(size(diff(oData.Adj_Close)));
            %disp(size(oData.Adj_Close(1:end)));
            oRet = diff(oData.Adj_Close)./ oData.Adj_Close(2:end);
            oRetDate = oData.Date(2:end);

            instrName = ticker + "_" + oType + "_" + oStr + "_"+ oExp;
            opandStockNames = [opandStockNames;instrName];
   
            disp(size(opandStockNames));
            %disp(size(oRetDate));
            %disp(size(oRet));
            %disp(size(instrNames));
            oTimeTable = timetable(oRetDate,oRet,'VariableNames',{char(instrName)});

            if isempty(optionTimeTable)
                optionTimeTable = oTimeTable;
            else
                optionTimeTable = synchronize(optionTimeTable,oTimeTable);
            end
        end
    end

    if ~isempty(optionTimeTable)
        combinedTimeTable = synchronize(stockTimeTable,optionTimeTable);
    else
        combinedTimeTable = stockTimeTable;
    end

    if ~exist('fullTimeTable','var')
        fullTimeTable = combinedTimeTable;
    else
        fullTimeTable = synchronize(fullTimeTable,combinedTimeTable);
    end
end

fullTimeTable = fillmissing(fullTimeTable, 'constant',0);
disp(fullTimeTable);
returnsMatrix = fullTimeTable{:,:};


% initialize Parameters
rf = 0.005; % risk-free rate
a = 0.95; % confidence level
T = 1; % ttm

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
    %lb = -.25*ones(size(returnsMatrix, 2), 1);
    lb = zeros(size(returnsMatrix, 2), 1);
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
disp(objective);
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
ylim([0,max(portReturn)]);

% display Tangency Portfolio Weights
fprintf('Tangency Portfolio Weights:\n');
for j = 1:length(opandStockNames)
    fprintf('%s: %.2f%%\n', opandStockNames(j), tanWeights(j)*100);
end
hold off;
%% Function Definition
function CVAR = calculateCVAR(portfolioReturns, a)
    portfolioLosses = -portfolioReturns; 
    VaR = quantile(portfolioLosses, a);
    CVAR = mean(portfolioLosses(portfolioLosses >= VaR)); 
end
