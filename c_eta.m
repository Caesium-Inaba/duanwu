%% 自变量与 η 散点图矩阵（控 C 变量标注）
%  将 C 按 20%~25%、50%~55%、70%~75% 分位数分三组，
%  在图上用不同颜色标出，控制变量法观察其他自变量与 η 的关系。
clear;clc;close all;

load('dc.mat', 'data_clean');

%% 变量准备
H   = data_clean.Temp_C;
C   = data_clean.C_in_gNm3;
Q   = data_clean.Q_Nm3h;
U1  = data_clean.U1_kV;  U2 = data_clean.U2_kV;
U3  = data_clean.U3_kV;  U4 = data_clean.U4_kV;
T1  = data_clean.T1_s;   T2 = data_clean.T2_s;
T3  = data_clean.T3_s;   T4 = data_clean.T4_s;
P   = data_clean.P_total_kW;
eta = data_clean.eff;

% 过滤 NaN
ok = isfinite(eta);
for v = {H, C, Q, U1, U2, U3, U4, T1, T2, T3, T4, P}
    ok = ok & isfinite(v{1});
end
fprintf('完整行: %d / %d\n', sum(ok), length(eta));

H = H(ok); C = C(ok); Q = Q(ok);
U1 = U1(ok); U2 = U2(ok); U3 = U3(ok); U4 = U4(ok);
T1 = T1(ok); T2 = T2(ok); T3 = T3(ok); T4 = T4(ok);
P = P(ok); eta = eta(ok);

%% C 分位数分组
pLo = prctile(C, [20, 25]);   % 低 C 组：20%~25%
pMi = prctile(C, [50, 55]);   % 中 C 组：50%~55%
pHi = prctile(C, [70, 75]);   % 高 C 组：70%~75%

inLo = C >= pLo(1) & C <= pLo(2);
inMi = C >= pMi(1) & C <= pMi(2);
inHi = C >= pHi(1) & C <= pHi(2);
inBand = inLo | inMi | inHi;

fprintf('C 分组:  低 [%.4f, %.4f] (%d 点)\n', pLo(1), pLo(2), sum(inLo));
fprintf('         中 [%.4f, %.4f] (%d 点)\n', pMi(1), pMi(2), sum(inMi));
fprintf('         高 [%.4f, %.4f] (%d 点)\n', pHi(1), pHi(2), sum(inHi));

outDir = fullfile('img', 'c_eta');
if ~exist(outDir, 'dir'), mkdir(outDir); end

%% 变量列表
varList = {H, C, Q, U1, U2, U3, U4, T1, T2, T3, T4, P};
varNames = {'H (°C)', 'C (g/Nm^3)', 'Q (Nm^3/h)', ...
    'U_1 (kV)', 'U_2 (kV)', 'U_3 (kV)', 'U_4 (kV)', ...
    'T_1 (s)', 'T_2 (s)', 'T_3 (s)', 'T_4 (s)', ...
    'P (kW)'};
varFiles = {'H', 'C', 'Q', 'U1', 'U2', 'U3', 'U4', 'T1', 'T2', 'T3', 'T4', 'P'};

% 颜色定义
cLo  = [0.85 0.20 0.25];  % 深红 — 低 C
cMi  = [0.20 0.55 0.35];  % 深绿 — 中 C
cHi  = [0.20 0.35 0.75];  % 深蓝 — 高 C
cBg  = [0.75 0.75 0.75];  % 浅灰 — 非三组

%% 逐一画散点图（控 C 标注）
for i = 1:length(varList)
    x = varList{i};
    [rho, pval] = corr(x, eta, 'type', 'Spearman');

    fig = figure('Name', sprintf('%s vs \\eta (控 C)', varNames{i}), ...
        'Position', [50, 50, 700, 540]);

    % 灰底：所有非三组点
    scatter(x(~inBand), eta(~inBand), 4, cBg, 'filled', 'MarkerFaceAlpha', 0.08);
    hold on;
    % 低 C 组（红）
    scatter(x(inLo), eta(inLo), 12, cLo, 'filled', 'MarkerFaceAlpha', 0.5);
    % 中 C 组（绿）
    scatter(x(inMi), eta(inMi), 12, cMi, 'filled', 'MarkerFaceAlpha', 0.5);
    % 高 C 组（蓝）
    scatter(x(inHi), eta(inHi), 12, cHi, 'filled', 'MarkerFaceAlpha', 0.5);
    hold off;

    xlabel(varNames{i});
    ylabel('\eta (%)');
    grid on;

    % 图例 + 统计信息
    leg = legend({sprintf('其他 (%.0f%%)', 100*(1-sum(inBand)/length(C))), ...
        sprintf('低C [P_{20},P_{25}] (%.0f%%)', 100*sum(inLo)/length(C)), ...
        sprintf('中C [P_{50},P_{55}] (%.0f%%)', 100*sum(inMi)/length(C)), ...
        sprintf('高C [P_{70},P_{75}] (%.0f%%)', 100*sum(inHi)/length(C))}, ...
        'Location', 'best', 'FontSize', 7);
    leg.ItemTokenSize = [10, 10];

    text(0.02, 0.06, sprintf('\\rho_{全} = %.4f  (p = %.2e)', rho, pval), ...
        'Units', 'normalized', 'VerticalAlignment', 'bottom', 'FontSize', 9);

    saveCurrent(fig, sprintf('%s_vs_eta_ctrlC', varFiles{i}), outDir);
end

%% 汇总大图 (4×3，控 C 标注)
fig = figure('Name', '各自变量 vs η 汇总 (控 C)', 'Position', [50, 50, 1600, 1100]);
for i = 1:12
    subplot(3, 4, i);
    x = varList{i};

    scatter(x(~inBand), eta(~inBand), 2, cBg, 'filled', 'MarkerFaceAlpha', 0.06);
    hold on;
    scatter(x(inLo), eta(inLo), 6, cLo, 'filled', 'MarkerFaceAlpha', 0.4);
    scatter(x(inMi), eta(inMi), 6, cMi, 'filled', 'MarkerFaceAlpha', 0.4);
    scatter(x(inHi), eta(inHi), 6, cHi, 'filled', 'MarkerFaceAlpha', 0.4);
    hold off;

    [rho, ~] = corr(x, eta, 'type', 'Spearman');
    xlabel(varNames{i}, 'FontSize', 7);
    ylabel('\eta (%)', 'FontSize', 7);
    title(sprintf('\\rho = %.4f', rho), 'FontSize', 9);
    grid on;
end

% 在汇总图加一个总图例
lg = legend({'其他', '低C [20,25]', '中C [50,55]', '高C [70,75]'}, ...
    'Position', [0.93, 0.93, 0.06, 0.06], 'FontSize', 6);
lg.ItemTokenSize = [8, 8];

saveCurrent(fig, 'all_vs_eta_ctrlC_grid', outDir);

%% 仅 C vs η（重点图，大图）
fig = figure('Name', 'C vs η 控 C 分组', 'Position', [50, 50, 750, 580]);
scatter(C(~inBand), eta(~inBand), 4, cBg, 'filled', 'MarkerFaceAlpha', 0.08);
hold on;
scatter(C(inLo), eta(inLo), 16, cLo, 'filled', 'MarkerFaceAlpha', 0.55);
scatter(C(inMi), eta(inMi), 16, cMi, 'filled', 'MarkerFaceAlpha', 0.55);
scatter(C(inHi), eta(inHi), 16, cHi, 'filled', 'MarkerFaceAlpha', 0.55);
hold off;
xlabel('C (g/Nm^3)');
ylabel('\eta (%)');
[rho, pval] = corr(C, eta, 'type', 'Spearman');
text(0.02, 0.06, sprintf('\\rho = %.4f  (p = %.2e)', rho, pval), ...
    'Units', 'normalized', 'VerticalAlignment', 'bottom', 'FontSize', 11);
leg = legend({sprintf('其他 (%.0f%%)', 100*(1-sum(inBand)/length(C))), ...
    sprintf('低C [%.4f, %.4f]', pLo(1), pLo(2)), ...
    sprintf('中C [%.4f, %.4f]', pMi(1), pMi(2)), ...
    sprintf('高C [%.4f, %.4f]', pHi(1), pHi(2))}, ...
    'Location', 'best', 'FontSize', 8);
leg.ItemTokenSize = [10, 10];
grid on;
saveCurrent(fig, 'C_vs_eta_ctrlC', outDir);

fprintf('\n完成。输出目录: %s\n', outDir);
