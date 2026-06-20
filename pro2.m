%% 第二问：分工况 Spearman 相关性分析
clear;clc;close all;
if isfile('pro2.txt'), delete('pro2.txt'); end
diary('pro2.txt');

load('dc.mat', 'data_clean');
load('dkmeans.mat', 'idx', 'kOpt', 'centers_H', 'centers_C');

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

    % 导出 CSV
    csvName = sprintf('pro2_spearman_cluster%d.csv', cluster);
    writeCorrCSV(R_spearman, csvName, allSymbols);
    fprintf('  已导出 %s\n', csvName);

    fprintf('\n');
end

fprintf('显著性: *** p<0.001  ** p<0.01  * p<0.05  n.s. p>=0.05\n');
diary off;
