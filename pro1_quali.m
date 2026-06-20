%% 第一问定性分析：正态性检验 → Pearson/Spearman 相关性
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
    fprintf('大样本下 JB 检验过度敏感，Pearson r 作为线性描述统计量仍有效\n');
    fprintf('但显著性应参考 Spearman（无分布假设）\n');
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

%% 图: 各变量 Q-Q 图（每变量独立图窗）
for i = 1:nVars
    fig = figure('Name', sprintf('Q-Q 图 - %s', allSymbols{i}));
    x = X_all(:, i);
    x = x(isfinite(x));
    qqplot(x);
    xlabel(''); ylabel('');
    grid on;
    saveCurrent(fig, sprintf('qq_%s', fileSafeSymbols{i}), outDir);
end

%% 图: Q-Q 图汇总大图（Rule 1: 4行, P η / H C Q / U_1-4 / T_1-4）
figQQGrid = figure('Name', 'Q-Q 图汇总');
for p = 1:13
    subplot(4, 4, gridPos1(p));
    x = X_all(:, gridCols1(p));
    x = x(isfinite(x));
    qqplot(x);
    xlabel(''); ylabel('');
    title(allSymbols{gridCols1(p)});
    grid on;
end
saveCurrent(figQQGrid, 'qq_grid', outDir);

%% 2. Pearson & Spearman 相关性分析

okRows = all(isfinite(X_all), 2);
X_comp = X_all(okRows, :);
fprintf('\n完整观测数: %d (去除 NaN 后)\n', sum(okRows));

[R_pearson, P_pearson] = corr(X_comp, 'type', 'Pearson');
[R_spearman, P_spearman] = corr(X_comp, 'type', 'Spearman');

%% 终端输出: 各变量与 η 的相关性（按 Spearman |rho| 降序）
fprintf('\n与 %s 的相关性 (Pearson vs Spearman)\n\n', targetSymbol);
fprintf('%-6s %10s %10s %10s %10s\n', '变量', 'Pearson r', 'p值', 'Spearman rho', 'p值');
fprintf('%s\n', repmat('-', 1, 55));

spearmanWithEff = R_spearman(1:nPred, end);
[~, sortIdx] = sort(abs(spearmanWithEff), 'descend');

for k = 1:nPred
    i = sortIdx(k);
    rP = R_pearson(i, end);
    pP = P_pearson(i, end);
    rS = R_spearman(i, end);
    pS = P_spearman(i, end);

    fprintf('%-6s %+10.4f %10.2e %+10.4f %10.2e  %s  %s\n', ...
        varSymbols{i}, rP, pP, rS, pS, ...
        significanceStars(pP), significanceStars(pS));
end

fprintf('\nPearson: 线性依赖强度 (计算不依赖正态假设)\n');
fprintf('Spearman: 单调依赖强度 (无分布假设)\n');
fprintf('显著性: *** p<0.001  ** p<0.01  * p<0.05  n.s. p>=0.05\n');
fprintf('\n诊断: Pearson r≈0 而 Spearman rho≠0 → 关系存在但非线性\n');

%% 图: Pearson 相关系数热力图
figPearson = figure('Name', 'Pearson 相关系数矩阵');
imagesc(R_pearson);
colormap(jet); colorbar; caxis([-1 1]);
set(gca, 'XTick', 1:nVars, 'XTickLabel', allSymbols, ...
         'YTick', 1:nVars, 'YTickLabel', allSymbols);
xtickangle(45);
axis equal tight;
for ii = 1:nVars
    for jj = 1:nVars
        if ii ~= jj
            text(jj, ii, sprintf('%.2f', R_pearson(ii,jj)), ...
                'HorizontalAlignment', 'center', 'FontSize', 7);
        end
    end
end
saveCurrent(figPearson, 'corr_pearson', outDir);

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

%% 图: 相关系数热力图汇总（Rule 3: Pearson + Spearman 紧密相连，并排）
figCorrGrid = figure('Name', '相关系数热力图汇总');
subplot(1, 2, 1);
imagesc(R_pearson);
colormap(jet); colorbar; caxis([-1 1]);
set(gca, 'XTick', 1:nVars, 'XTickLabel', allSymbols, ...
         'YTick', 1:nVars, 'YTickLabel', allSymbols);
xtickangle(45); axis equal tight;
title('Pearson');
for ii = 1:nVars
    for jj = 1:nVars
        if ii ~= jj
            text(jj, ii, sprintf('%.2f', R_pearson(ii,jj)), ...
                'HorizontalAlignment', 'center', 'FontSize', 6);
        end
    end
end
subplot(1, 2, 2);
imagesc(R_spearman);
colormap(jet); colorbar; caxis([-1 1]);
set(gca, 'XTick', 1:nVars, 'XTickLabel', allSymbols, ...
         'YTick', 1:nVars, 'YTickLabel', allSymbols);
xtickangle(45); axis equal tight;
title('Spearman');
for ii = 1:nVars
    for jj = 1:nVars
        if ii ~= jj
            text(jj, ii, sprintf('%.2f', R_spearman(ii,jj)), ...
                'HorizontalAlignment', 'center', 'FontSize', 6);
        end
    end
end
saveCurrent(figCorrGrid, 'corr_heatmap', outDir);

%% 图: 各预测变量 vs η 散点图（按 Spearman |rho| 降序，每变量独立图窗）
for k = 1:nPred
    i = sortIdx(k);
    x = X_comp(:, i);
    y = X_comp(:, end);
    rP = R_pearson(i, end);
    rS = R_spearman(i, end);

    fig = figure('Name', sprintf('%s vs %s', varSymbols{i}, targetSymbol));
    scatter(x, y, 2, 'filled', 'MarkerFaceAlpha', 0.3);
    [xs, idxSort] = sort(x);
    ys = smoothdata(y(idxSort), 'loess', 500);
    hold on;
    plot(xs, ys, 'r-', 'LineWidth', 1.5);
    xlabel(varSymbols{i}); ylabel(targetSymbol);
    grid on;
    % 在图窗 Name 中标注相关系数（作为图窗标题）
    set(fig, 'Name', sprintf('%s vs %s  (Pearson: %+.3f, Spearman: %+.3f)', ...
        varSymbols{i}, targetSymbol, rP, rS));
    saveCurrent(fig, sprintf('scatter_%s_vs_eta', varSymbols{i}), outDir);
end

%% 图: 散点图汇总大图（Rule 2: 3行, P H C Q / U_1-4 / T_1-4）
% X_all 列: P=12, H=1, C=2, Q=3 / U_1-4=4:7 / T_1-4=8:11
gridCols2 = [12, 1, 2, 3,  4, 5, 6, 7,  8, 9, 10, 11];
figScatterGrid = figure('Name', '预测变量 vs η 汇总');
for p = 1:12
    subplot(3, 4, p);
    col = gridCols2(p);
    x = X_comp(:, col);
    y = X_comp(:, end);
    scatter(x, y, 2, 'filled', 'MarkerFaceAlpha', 0.3);
    [xs, idxSort] = sort(x);
    ys = smoothdata(y(idxSort), 'loess', 500);
    hold on;
    plot(xs, ys, 'r-', 'LineWidth', 1.5);
    xlabel(varSymbols{col}); ylabel(targetSymbol);
    rP = R_pearson(col, end);
    rS = R_spearman(col, end);
    title(sprintf('r=%.3f  rho=%.3f', rP, rS), 'FontSize', 8);
    grid on;
end
saveCurrent(figScatterGrid, 'scatter_grid', outDir);

%% 3. 变量间自相关诊断
fprintf('\n变量间高相关对 (|r|>0.7)\n\n');
highCorrFound = false;
for ii = 1:nPred
    for jj = ii+1:nPred
        r = R_pearson(ii, jj);
        if abs(r) > 0.7
            highCorrFound = true;
            fprintf('  %s <-> %s: r = %+.3f\n', varSymbols{ii}, varSymbols{jj}, r);
        end
    end
end
if ~highCorrFound
    fprintf('  未发现 |r| > 0.7 的强相关对\n');
end
fprintf('注: 若存在强相关对，建立回归模型时需考虑共线性\n');

%% 4. 导出相关系数矩阵为 CSV

writeCorrCSV(R_pearson, 'pro1_quali_pearson.csv', allSymbols);
writeCorrCSV(R_spearman, 'pro1_quali_spearman.csv', allSymbols);
fprintf('已导出 pro1_quali_pearson.csv 和 pro1_quali_spearman.csv\n');

fprintf('\n图片已保存到 %s\n', outDir);
diary off;
