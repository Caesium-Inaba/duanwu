%% 第二问：分工况 Spearman 相关性分析
clear;clc;close all;
if isfile('pro2.txt'), delete('pro2.txt'); end
diary('pro2.txt');

load('dc.mat', 'data_clean');
load('dkmeans.mat', 'idx', 'kOpt', 'centers_H', 'centers_C');

outDir = fullfile('img', 'pro2');
if ~exist(outDir, 'dir'), mkdir(outDir); end

%% 变量定义（符号对齐论文统一符号表）
predVars = {'Temp_C','C_in_gNm3','Q_Nm3h', ...
            'U1_kV','U2_kV','U3_kV','U4_kV', ...
            'T1_s','T2_s','T3_s','T4_s','P_total_kW'};
varSymbols = {'H','C','Q', ...
              'U_1','U_2','U_3','U_4', ...
              'T_1','T_2','T_3','T_4','P'};
targetVar = 'eff';
targetSymbol = '\eta';

allVars = [predVars, {targetVar}];
allSymbols = [varSymbols, {targetSymbol}];
nVars = length(allVars);
nPred = length(predVars);

%% 构造完整数据矩阵
X_all = NaN(height(data_clean), nVars);
for i = 1:nVars
    x = data_clean.(allVars{i});
    ok = isfinite(x);
    X_all(ok, i) = x(ok);
end

%% 分工况 Spearman 相关
fprintf('分工况 Spearman 秩相关系数分析 (k=%d)\n\n', kOpt);
fprintf('聚类中心:\n');
for j = 1:kOpt
    fprintf('  工况 %d: H=%.1f C, C=%.2f g/Nm^3\n', j, centers_H(j), centers_C(j));
end
fprintf('\n');

R_clusters = cell(kOpt, 1);  % 各工况 Spearman rho 矩阵
P_clusters = cell(kOpt, 1);  % 各工况 p 值矩阵

for cluster = 1:kOpt
    mask = idx == cluster;
    X_cluster = X_all(mask, :);
    % 仅保留完整观测行
    okRows = all(isfinite(X_cluster), 2);
    X_cluster = X_cluster(okRows, :);
    n = size(X_cluster, 1);

    fprintf('-- 工况 %d (n=%d) --\n', cluster, n);
    fprintf('  中心: H=%.1f C, C=%.2f g/Nm^3\n', centers_H(cluster), centers_C(cluster));

    if n < 20
        fprintf('  样本量过小，跳过相关分析\n\n');
        continue;
    end

    [R_spearman, P_spearman] = corr(X_cluster, 'type', 'Spearman');
    R_clusters{cluster} = R_spearman;
    P_clusters{cluster} = P_spearman;

    % 输出与 eta 的 Spearman rho（按 |rho| 降序）
    rhoWithEta = R_spearman(1:nPred, end);
    [~, sortIdxLocal] = sort(abs(rhoWithEta), 'descend');

    fprintf('  与 %s 的 Spearman 相关 (|rho| 降序):\n', targetSymbol);
    fprintf('  %-6s %10s %10s\n', '变量', 'rho', 'p值');
    for k = 1:nPred
        i = sortIdxLocal(k);
        rho = R_spearman(i, end);
        p = P_spearman(i, end);
        fprintf('  %-6s %+10.4f %10.2e  %s\n', varSymbols{i}, rho, p, significanceStars(p));
    end

    % 导出 CSV（rho 矩阵 + p 值矩阵）
    csvRho = sprintf('pro2_spearman_cluster%d_rho.csv', cluster);
    csvPval = sprintf('pro2_spearman_cluster%d_pval.csv', cluster);
    writeCorrCSV(R_spearman, csvRho, allSymbols);
    writeCorrCSV(P_spearman, csvPval, allSymbols);
    fprintf('  已导出 %s, %s\n', csvRho, csvPval);

    % 热力图：显著格子 jet 色阶，不显著 (p>0.05) 涂灰
    cmap = jet(256);
    nV = nVars;
    img = zeros(nV, nV, 3);
    for ii = 1:nV
        for jj = 1:nV
            if P_spearman(ii,jj) > 0.05
                img(ii, jj, :) = [0.65, 0.65, 0.65];
            else
                cidx = round((R_spearman(ii,jj) + 1) / 2 * 255) + 1;
                cidx = max(1, min(256, cidx));
                img(ii, jj, :) = cmap(cidx, :);
            end
        end
    end
    fig = figure('Name', sprintf('Spearman - 工况 %d (n=%d)', cluster, n), ...
        'Position', [50, 50, 750, 650]);
    image(img);
    set(gca, 'XTick', 1:nVars, 'XTickLabel', allSymbols, ...
             'YTick', 1:nVars, 'YTickLabel', allSymbols);
    xtickangle(45);
    axis equal tight;
    % 手工 colorbar
    colormap(jet); caxis([-1 1]); colorbar('Position', [0.92, 0.15, 0.03, 0.70]);
    for ii = 1:nVars
        for jj = 1:nVars
            if ii ~= jj
                t = sprintf('%.2f', R_spearman(ii,jj));
                if P_spearman(ii,jj) <= 0.05
                    t = [t, newline, significanceStars(P_spearman(ii,jj))];
                else
                    t = [t, newline, 'n.s.'];
                end
                text(jj, ii, t, 'HorizontalAlignment', 'center', 'FontSize', 8);
            end
        end
    end
    saveCurrent(fig, sprintf('spearman_cluster%d', cluster), outDir);

    fprintf('\n');
end

%% 分工况 Spearman 汇总大图（2×2）
%  第一排: 工况1(低温高尘)  工况4(高温高尘)
%  第二排: 工况3(低温低尘)  工况2(高温低尘)
gridMap = [1, 4; 3, 2];
clusterLabel = {
    sprintf('工况 1: 低温高尘 (n=%d)', sum(idx==1 & all(isfinite(X_all),2)));
    sprintf('工况 2: 高温低尘 (n=%d)', sum(idx==2 & all(isfinite(X_all),2)));
    sprintf('工况 3: 低温低尘 (n=%d)', sum(idx==3 & all(isfinite(X_all),2)));
    sprintf('工况 4: 高温高尘 (n=%d)', sum(idx==4 & all(isfinite(X_all),2)));
};

figGrid = figure('Name', '分工况 Spearman 秩相关系数矩阵', ...
    'Position', [50, 50, 1200, 900]);
cmap = jet(256);
for row = 1:2
    for col = 1:2
        c = gridMap(row, col);
        subplot(2, 2, (row-1)*2 + col);
        if isempty(R_clusters{c})
            axis off; continue;
        end
        R = R_clusters{c};
        P = P_clusters{c};
        nV = nVars;
        img = zeros(nV, nV, 3);
        for ii = 1:nV
            for jj = 1:nV
                if P(ii,jj) > 0.05
                    img(ii, jj, :) = [0.65, 0.65, 0.65];
                else
                    cidx = round((R(ii,jj) + 1) / 2 * 255) + 1;
                    cidx = max(1, min(256, cidx));
                    img(ii, jj, :) = cmap(cidx, :);
                end
            end
        end
        image(img);
        set(gca, 'XTick', 1:nVars, 'XTickLabel', allSymbols, ...
                 'YTick', 1:nVars, 'YTickLabel', allSymbols);
        xtickangle(45);
        axis equal tight;
        title(clusterLabel{c}, 'FontSize', 10);
        for ii = 1:nVars
            for jj = 1:nVars
                if ii ~= jj
                    t = sprintf('%.2f', R(ii,jj));
                    if P(ii,jj) <= 0.05
                        t = [t, newline, significanceStars(P(ii,jj))];
                    else
                        t = [t, newline, 'n.s.'];
                    end
                    text(jj, ii, t, 'HorizontalAlignment', 'center', 'FontSize', 6);
                end
            end
        end
    end
end
saveCurrent(figGrid, 'spearman_grid', outDir);

fprintf('显著性: *** p<0.001  ** p<0.01  * p<0.05  n.s. p>=0.05\n');
diary off;
