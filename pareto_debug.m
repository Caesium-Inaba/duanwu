%% pareto_debug.m — 加载 opt_lease.mat 快速出图（调试用）
%  subplot 顺序 (1,4;3,2)：上行高C，下行低C
clear;clc;close all;

load('opt_lease.mat', 'results');
load('dc.mat', 'data_clean');
load('dkmeans.mat', 'idx');

% 历史数据
eta = data_clean.eff;  C_in = data_clean.C_in_gNm3;  P_hist = data_clean.P_total_kW;
ok = isfinite(eta) & isfinite(C_in) & isfinite(P_hist);
eta = eta(ok); C_in = C_in(ok); P_hist = P_hist(ok); idx = idx(ok);

outDir = fullfile('img', 'pro2opt_lease');
if ~exist(outDir, 'dir'), mkdir(outDir); end

subOrder = [1, 4, 3, 2];  % 位置1→工况1, 位置2→工况4, 位置3→工况3, 位置4→工况2
colors = lines(4);

fig = figure('Name', 'Pareto 前沿汇总 (调试)', 'Position', [50, 50, 1050, 850]);
for pos = 1:4
    k = subOrder(pos);
    r = results{k};
    if isempty(r), continue; end

    subplot(2, 2, pos);

    % 灰底：全部网格候选点
    scatter(r.Cout_all, r.P_all, 1, [0.75 0.75 0.75], 'filled', ...
        'MarkerFaceAlpha', 0.03);
    hold on;

    % 蓝点：历史实测
    mask_k = idx == k;
    C_out_hist = C_in(mask_k) * 1000 .* (1 - eta(mask_k) / 100);
    scatter(C_out_hist, P_hist(mask_k), 3, [0.2 0.5 0.8], 'filled', ...
        'MarkerFaceAlpha', 0.10);

    % 红线：Pareto 前沿
    [c_sort, ord] = sort(r.Cout_pareto);
    plot(c_sort, r.P_pareto(ord), '-o', 'Color', colors(k,:), ...
        'LineWidth', 1.8, 'MarkerSize', 3);
    xlabel('C_{out} (mg/Nm³)'); ylabel('P (kW)');
    title(sprintf('工况 %d (H≈%.1f, C≈%.2f)', k, r.H_center, r.C_center));
    grid on;
end

saveas(fig, fullfile(outDir, 'pareto_combined.svg'));
saveas(fig, fullfile(outDir, 'pareto_combined.fig'));
fprintf('已保存: %s\n', fullfile(outDir, 'pareto_combined.*'));
