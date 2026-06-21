%% dream.m — "梦想情景"：用历史最优 η 和最低 C 能否达标？
%  C_out = C × 1000 × (1 − η/100)   （C: g/Nm³, C_out: mg/Nm³）
%  国标: C_out ≤ 10 mg/Nm³
clear;clc;close all;

load('dc.mat', 'data_clean');
load('dkmeans.mat', 'idx', 'kOpt', 'centers_H', 'centers_C');

%% 变量准备
C_in  = data_clean.C_in_gNm3;
eta   = data_clean.eff;
ok    = isfinite(C_in) & isfinite(eta) & isfinite(data_clean.P_total_kW);
C_in  = C_in(ok);
eta   = eta(ok);
idx   = idx(ok);
fprintf('有效样本: %d\n', sum(ok));

%% 历史全局极值
eta_max_global = max(eta);
C_min_global   = min(C_in);
fprintf('\n历史全局最大 η = %.4f%%\n', eta_max_global);
fprintf('历史全局最小 C = %.4f g/Nm³\n', C_min_global);

%% 梦想情景计算
% 情景 A: 全局 min C + 全局 max η（最强组合，虽未必同一点）
C_out_dream_global = C_min_global * 1000 * (1 - eta_max_global / 100);
fprintf('\n═══════════════════════════════════════\n');
fprintf('  「梦想情景」：min C + max η\n');
fprintf('  C_out = %.4f × 1000 × (1 − %.4f/100)\n', C_min_global, eta_max_global);
fprintf('       = %.4f mg/Nm³\n', C_out_dream_global);
if C_out_dream_global <= 10
    fprintf('  ✅ 达标！\n');
else
    fprintf('  ❌ 仍超标 %.2f mg/Nm³\n', C_out_dream_global - 10);
end
fprintf('═══════════════════════════════════════\n');

%% 要达到 C_out=10，需要多大的 η？
fprintf('\n——— 达标所需 η（各工况）———\n');
fprintf('  C_out = C × 1000 × (1 − η/100) ≤ 10\n');
fprintf('  → η ≥ (1 − 10/(C×1000)) × 100\n\n');
for k = 1:kOpt
    C_k = centers_C(k);
    eta_need = (1 - 10 / (C_k * 1000)) * 100;
    mask_k = idx == k;
    eta_max_k = max(eta(mask_k));
    fprintf('  工况 %d (C=%.2f g/Nm³): 需要 η ≥ %.4f%%,  历史最大 η = %.4f%%', ...
        k, C_k, eta_need, eta_max_k);
    if eta_max_k >= eta_need
        fprintf('  ✅');
    else
        fprintf('  ❌ 差 %.4f 百分点', eta_need - eta_max_k);
    end
    fprintf('\n');
end

%% 全局反算：要达到 C_out=10，min C 需要配对多大的 η
eta_need_global = (1 - 10 / (C_min_global * 1000)) * 100;
fprintf('\n——— 全局反算 ———\n');
fprintf('  若 C = %.4f (全局最小), 需 η ≥ %.4f%% 才能达标\n', C_min_global, eta_need_global);
fprintf('  历史最大 η = %.4f%%, 差距 = %.4f 百分点\n', eta_max_global, eta_need_global - eta_max_global);

%% 分工况梦想情景
fprintf('\n——— 分工况梦想情景 (min C_k + max η_global) ———\n');
for k = 1:kOpt
    mask_k = idx == k;
    C_min_k = min(C_in(mask_k));
    C_out_dream_k = C_min_k * 1000 * (1 - eta_max_global / 100);
    fprintf('  工况 %d: min C = %.4f, dream C_out = %.4f mg/Nm³', ...
        k, C_min_k, C_out_dream_k);
    if C_out_dream_k <= 10
        fprintf('  ✅');
    else
        fprintf('  ❌ +%.2f', C_out_dream_k - 10);
    end
    fprintf('\n');
end

%% 实际点中 C_out 的分布
C_out_all = C_in * 1000 .* (1 - eta / 100);
fprintf('\n——— 历史 C_out 分布 ———\n');
fprintf('  min  = %.4f mg/Nm³\n', min(C_out_all));
fprintf('  max  = %.4f mg/Nm³\n', max(C_out_all));
fprintf('  mean = %.4f mg/Nm³\n', mean(C_out_all));
fprintf('  std  = %.4f mg/Nm³\n', std(C_out_all));
nUnder10 = sum(C_out_all <= 10);
fprintf('  ≤10 mg/Nm³ 的点: %d / %d (%.2f%%)\n', nUnder10, length(C_out_all), 100*nUnder10/length(C_out_all));

%% 散点图：C vs C_out，标注国标线与梦想点
outDir = fullfile('img', 'dream');
if ~exist(outDir, 'dir'), mkdir(outDir); end

fig = figure('Name', 'C vs C_out — 梦想情景', 'Position', [50, 50, 800, 600]);

% 颜色按 η 高低
scatter(C_in, C_out_all, 5, eta, 'filled', 'MarkerFaceAlpha', 0.15);
colormap jet; cb = colorbar; cb.Label.String = '\eta (%)';
hold on;
yline(10, 'r--', 'LineWidth', 1.5);  % 国标线

% 标注梦想点
plot(C_min_global, C_out_dream_global, 'r*', 'MarkerSize', 14, 'LineWidth', 1.5);
text(C_min_global, C_out_dream_global, ...
    sprintf('  梦想点\n  C=%.2f, η=%.2f%%\n  C_{out}=%.2f', ...
    C_min_global, eta_max_global, C_out_dream_global), ...
    'VerticalAlignment', 'bottom', 'FontSize', 9, 'Color', 'r');

xlabel('C_{in} (g/Nm³)');
ylabel('C_{out} (mg/Nm³)');
title('即使梦想组合也难达标');
grid on;
saveCurrent(fig, 'dream_C_vs_Cout', outDir);

fprintf('\n完成。\n');
