%% 第一问定性分析：正态性检验 → Spearman 相关性
clear;clc;close all;
if isfile('pro1_quali.txt'), delete('pro1_quali.txt'); end
diary('pro1_quali.txt');

load('dc.mat', 'data_clean');

outDir = fullfile('img', 'pro1_quali');
if ~exist(outDir, 'dir'), mkdir(outDir); end

%% 变量定义（符号对齐论文统一符号表）
%  H: 温度 (℃)     C: 入口粉尘浓度 (g/Nm³)     Q: 流量 (Nm³/h)
%  U_i: 第 i 电场二次电压 (kV)     T_i: 第 i 电场振打周期 (s)
%  P: 总电耗 (kW)     η: 除尘效率 (%)
predVars = {'Temp_C','C_in_gNm3','Q_Nm3h', ...
            'U1_kV','U2_kV','U3_kV','U4_kV', ...
            'T1_s','T2_s','T3_s','T4_s','P_total_kW'};
varSymbols = {'H','C','Q', ...
              'U_1','U_2','U_3','U_4', ...
              'T_1','T_2','T_3','T_4','P'};
targetVar = 'eff';
targetSymbol = '\eta';
% 文件名安全版本（去除 LaTeX 转义符）
fileSafeSymbols = {'H','C','Q','U1','U2','U3','U4','T1','T2','T3','T4','P','eta'};

allVars = [predVars, {targetVar}];
allSymbols = [varSymbols, {targetSymbol}];
nVars = length(allVars);
nPred = length(predVars);

%% 1. 正态性检验（Jarque-Bera + 偏度/峰度）
fprintf('正态性检验 (Jarque-Bera, H0: 正态分布, alpha=0.05)\n\n');
fprintf('%-6s %8s %8s %8s %6s  %s\n', '变量', '偏度', '峰度', 'p值', 'h', '结论');
fprintf('%s\n', repmat('-', 1, 58));

skewVals = zeros(nVars, 1);
kurtVals = zeros(nVars, 1);
jbPVals  = zeros(nVars, 1);
isNormal = zeros(nVars, 1);
X_all    = zeros(height(data_clean), nVars);

for i = 1:nVars
    vn = allVars{i};
    x = data_clean.(vn);
    ok = isfinite(x);
    x_ok = x(ok);
    X_all(ok, i) = x_ok;

    skewVals(i) = skewness(x_ok);
    kurtVals(i) = kurtosis(x_ok);
    [h, p] = jbtest(x_ok);
    jbPVals(i) = p;
    isNormal(i) = ~h;

    if h, verdict = '非正态'; else, verdict = '正态'; end
    fprintf('%-6s %+8.3f %+8.3f %8.4f %6d  %s\n', ...
        allSymbols{i}, skewVals(i), kurtVals(i), p, h, verdict);
end

nNormal = sum(isNormal);
fprintf('\n%d/%d 个变量通过正态性检验\n', nNormal, nVars);
if nNormal < nVars
    fprintf('大样本下 JB 检验过度敏感，改用 Spearman 秩相关（无分布假设）\n');
end

%% 图: 各变量直方图 + 正态叠加（每变量独立图窗）
for i = 1:nVars
    x = X_all(:, i);
    x = x(isfinite(x));
    fig = figure('Name', sprintf('直方图 - %s', allSymbols{i}));
    histogram(x, 40, 'Normalization', 'pdf', 'EdgeAlpha', 0.3, 'FaceAlpha', 0.6);
    hold on;
    xg = linspace(min(x), max(x), 200);
    mu = mean(x); sg = std(x);
    plot(xg, normpdf(xg, mu, sg), 'r-', 'LineWidth', 1.5);
    xlabel(allSymbols{i}); ylabel('概率密度');
    grid on;
    saveCurrent(fig, sprintf('hist_%s', fileSafeSymbols{i}), outDir);
end

%% 图: 直方图汇总大图（Rule 1: 4行, P η / H C Q / U_1-4 / T_1-4）
gridCols1 = [12, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];      % X_all 列: P,η,H,C,Q,U_1-4,T_1-4
gridPos1  = [ 2,  3, 5, 6, 7, 9,10,11,12,13,14,15,16];       % subplot 位
figHistGrid = figure('Name', '直方图汇总');
for p = 1:13
    subplot(4, 4, gridPos1(p));
    x = X_all(:, gridCols1(p));
    x = x(isfinite(x));
    histogram(x, 40, 'Normalization', 'pdf', 'EdgeAlpha', 0.3, 'FaceAlpha', 0.6);
    hold on;
    xg = linspace(min(x), max(x), 200);
    mu = mean(x); sg = std(x);
    plot(xg, normpdf(xg, mu, sg), 'r-', 'LineWidth', 1);
    xlabel(allSymbols{gridCols1(p)}); ylabel('');
    grid on;
end
saveCurrent(figHistGrid, 'hist_grid', outDir);

%% 2. Spearman 相关性分析

okRows = all(isfinite(X_all), 2);
X_comp = X_all(okRows, :);
fprintf('\n完整观测数: %d (去除 NaN 后)\n', sum(okRows));

[R_spearman, P_spearman] = corr(X_comp, 'type', 'Spearman');

%% 终端输出: 各变量与 η 的相关性（按 |rho| 降序）
fprintf('\n与 %s 的 Spearman 相关\n\n', targetSymbol);
fprintf('%-6s %10s %10s\n', '变量', 'Spearman rho', 'p值');
fprintf('%s\n', repmat('-', 1, 30));

spearmanWithEff = R_spearman(1:nPred, end);
[~, sortIdx] = sort(abs(spearmanWithEff), 'descend');

for k = 1:nPred
    i = sortIdx(k);
    rS = R_spearman(i, end);
    pS = P_spearman(i, end);

    fprintf('%-6s %+10.4f %10.2e  %s\n', ...
        varSymbols{i}, rS, pS, ...
        significanceStars(pS));
end

fprintf('\nSpearman: 单调依赖强度 (无分布假设)\n');
fprintf('显著性: *** p<0.001  ** p<0.01  * p<0.05  n.s. p>=0.05\n');

%% 图: Spearman 相关系数热力图
figSpearman = figure('Name', 'Spearman 秩相关系数矩阵');
imagesc(R_spearman);
colormap(jet); colorbar; caxis([-1 1]);
set(gca, 'XTick', 1:nVars, 'XTickLabel', allSymbols, ...
         'YTick', 1:nVars, 'YTickLabel', allSymbols);
xtickangle(45);
axis equal tight;
for ii = 1:nVars
    for jj = 1:nVars
        if ii ~= jj
            text(jj, ii, sprintf('%.2f', R_spearman(ii,jj)), ...
                'HorizontalAlignment', 'center', 'FontSize', 7);
        end
    end
end
saveCurrent(figSpearman, 'corr_spearman', outDir);

%% 3. 变量间自相关诊断
fprintf('\n变量间高相关对 (|rho|>0.7)\n\n');
highCorrFound = false;
for ii = 1:nPred
    for jj = ii+1:nPred
        r = R_spearman(ii, jj);
        if abs(r) > 0.7
            highCorrFound = true;
            fprintf('  %s <-> %s: rho = %+.3f\n', varSymbols{ii}, varSymbols{jj}, r);
        end
    end
end
if ~highCorrFound
    fprintf('  未发现 |rho| > 0.7 的强相关对\n');
end
fprintf('注: 若存在强相关对，建立回归模型时需考虑共线性\n');

%% 4. 导出 Spearman 相关系数矩阵为 CSV

writeCorrCSV(R_spearman, 'pro1_quali_spearman.csv', allSymbols);
writeCorrCSV(P_spearman, 'pro1_quali_spearman_p.csv', allSymbols);
fprintf('已导出 pro1_quali_spearman.csv 和 pro1_quali_spearman_p.csv\n');

fprintf('\n图片已保存到 %s\n', outDir);
diary off;
