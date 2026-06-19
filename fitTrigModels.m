function [best, info] = fitTrigModels(y, t_hours)
% 对 y(t) 拟合三种三角函数模型，AICc 选最优

n = length(y);
ok = isfinite(y);
nOk = sum(ok);

% 模型定义：名称 + {周期(h), 谐波k} 列表
modelNames = {'日周期', '日 + 半日谐波', '日 + 2天交替'};
modelPairs = {
    {[24, 1]};
    {[24, 1], [24, 2]};
    {[24, 1], [48, 1]};
};
nModel = 3;
results = cell(nModel, 1);

for m = 1:nModel
    pairs = modelPairs{m};
    nPair = length(pairs);
    nCoef = 1 + 2 * nPair;
    X = ones(n, nCoef);
    ampPeriod = zeros(1, nPair);
    for p = 1:nPair
        Tk = pairs{p}(1);
        kk = pairs{p}(2);
        omega = 2 * pi * kk / Tk;
        col = 2 + 2*(p-1);
        X(:, col)   = sin(omega * t_hours);
        X(:, col+1) = cos(omega * t_hours);
        ampPeriod(p) = Tk / kk;
    end

    beta = X(ok, :) \ y(ok);
    yFit = X * beta;
    res = y - yFit;
    RSS = sum(res(ok).^2);
    sigma2 = RSS / (nOk - nCoef);
    logL = -0.5 * nOk * (log(2*pi*sigma2) + 1);
    AICc = 2*nCoef - 2*logL + 2*nCoef*(nCoef+1) / max(nOk-nCoef-1, 1);

    % 各频率分量的振幅
    amp = zeros(1, nPair);
    for p = 1:nPair
        col = 2 + 2*(p-1);
        amp(p) = sqrt(beta(col)^2 + beta(col+1)^2);
    end

    TSS = sum((y(ok) - mean(y(ok))).^2);
    R2  = 1 - RSS / TSS;
    RMSE = sqrt(RSS / nOk);

    results{m} = struct(...
        'modelName', modelNames{m}, ...
        'beta', beta, ...
        'yFit', yFit, ...
        'res', res, ...
        'R2', R2, ...
        'RMSE', RMSE, ...
        'AICc', AICc, ...
        'nCoef', nCoef, ...
        'ampPeriod', ampPeriod, ...
        'amp', amp, ...
        'pairs', {pairs});
end

% AICc 选最优
aiccVals = cellfun(@(r) r.AICc, results);
[~, idxBest] = min(aiccVals);
best = results{idxBest};

% 构建对比信息表
info = table();
info.Model = modelNames';
info.nParam = cellfun(@(r) r.nCoef, results);
info.R2 = cellfun(@(r) r.R2, results);
info.RMSE = cellfun(@(r) r.RMSE, results);
info.AICc = cellfun(@(r) r.AICc, results);
info.IsBest = false(nModel, 1);
info.IsBest(idxBest) = true;
end
