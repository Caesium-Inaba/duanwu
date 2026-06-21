%% pro2opt_lease.m — 双目标 Pareto 优化（无约束）
%  min [P, C_out]  网格搜索 + 非支配排序
%  f: η = RF (f_model) → C_out = C_k·1000·(1-η/100)
%  g: P = Poly (g_mdls)
%  缓存：opt_lease.mat + img/pro2opt_lease/*.fig
clear;clc;close all;

%% 缓存
outDir = fullfile('img', 'pro2opt_lease');
if ~exist(outDir, 'dir'), mkdir(outDir); end
cacheFile = 'opt_lease.mat';

figNames = {};
for c = 1:4
    figNames{end+1} = sprintf('pareto_cluster%d', c);
end
figNames{end+1} = 'pareto_combined';
allFigsExist = all(cellfun(@(n) isfile(fullfile(outDir, [n '.fig'])), figNames));

%% 加载模型与数据
load('rf_models.mat', 'f_model');
load('poly_models.mat', 'g_mdls');
load('dkmeans.mat', 'idx', 'kOpt', 'centers_H', 'centers_C');
load('dc.mat', 'data_clean');

%% 变量
Ubar1 = (data_clean.U1_kV + data_clean.U2_kV) / 2;
Ubar2 = (data_clean.U3_kV + data_clean.U4_kV) / 2;
Tbar1 = (data_clean.T1_s  + data_clean.T2_s)  / 2;
Tbar2 = (data_clean.T3_s  + data_clean.T4_s)  / 2;
Q_all = data_clean.Q_Nm3h;
H_all = data_clean.Temp_C;
C_all = data_clean.C_in_gNm3;
P_all = data_clean.P_total_kW;
eta_all = data_clean.eff;

nGrid = 25;   % 每维 25 格 → 25^4 ≈ 39 万点/工况

%% ============================================================
%%  主流程
%% ============================================================
if allFigsExist
    load(cacheFile, 'results');
    fprintf('从缓存加载 (%s)。\n', cacheFile);
    openCachedFigs();

elseif isfile(cacheFile)
    load(cacheFile, 'results');
    fprintf('加载 %s，重新出图...\n', cacheFile);
    plotPareto(results, Ubar1, Ubar2, Tbar1, Tbar2, P_all, ...
        C_all, eta_all, idx, outDir);

else
    fprintf('从头网格搜索 (%d^4 = %d 候选/工况)...\n', nGrid, nGrid^4);
    results = cell(kOpt, 1);
    rng(42);  % 固定种子（RF predict 是确定的，但留作习惯）

    for cluster = 1:kOpt
        mask = idx == cluster;
        H_k = centers_H(cluster);
        C_k = centers_C(cluster);
        Q_k = mean(Q_all(mask), 'omitnan');

        u1_lim = [min(Ubar1(mask)), max(Ubar1(mask))];
        u2_lim = [min(Ubar2(mask)), max(Ubar2(mask))];
        t1_lim = [min(Tbar1(mask)), max(Tbar1(mask))];
        t2_lim = [min(Tbar2(mask)), max(Tbar2(mask))];

        u1_v = linspace(u1_lim(1), u1_lim(2), nGrid);
        u2_v = linspace(u2_lim(1), u2_lim(2), nGrid);
        t1_v = linspace(t1_lim(1), t1_lim(2), nGrid);
        t2_v = linspace(t2_lim(1), t2_lim(2), nGrid);
        [G1, G2, G3, G4] = ndgrid(u1_v, u2_v, t1_v, t2_v);
        nTotal = numel(G1);

        % 预测 C_out（RF f model）
        X_f = [repmat([H_k, C_k, Q_k], nTotal, 1), G1(:), G2(:), G3(:), G4(:)];
        eta_pred = predict(f_model, X_f);
        C_out_pred = C_k * 1000 * (1 - eta_pred / 100);

        % 预测 P（Poly g model）
        X_g = [G1(:), G2(:), G3(:), G4(:)];
        g_mdl = g_mdls{cluster};
        if isempty(g_mdl)
            fprintf('  工况 %d: g 模型缺失，跳过\n', cluster);
            results{cluster} = [];
            continue;
        end
        P_pred = predict(g_mdl, X_g);

        % 非支配排序
        pareto_mask = nonDominated(P_pred, C_out_pred);
        nPareto = sum(pareto_mask);

        results{cluster} = struct(...
            'cluster', cluster, ...
            'H_center', H_k, 'C_center', C_k, 'Q_center', Q_k, ...
            'U1_all', G1(:), 'U2_all', G2(:), 'T1_all', G3(:), 'T2_all', G4(:), ...
            'P_all', P_pred, 'Cout_all', C_out_pred, ...
            'P_pareto', P_pred(pareto_mask), ...
            'Cout_pareto', C_out_pred(pareto_mask), ...
            'U1_pareto', G1(pareto_mask), 'U2_pareto', G2(pareto_mask), ...
            'T1_pareto', G3(pareto_mask), 'T2_pareto', G4(pareto_mask), ...
            'u1_range', u1_lim, 'u2_range', u2_lim, ...
            't1_range', t1_lim, 't2_range', t2_lim);

        fprintf('  工况 %d: %d 候选 → %d Pareto (%d%%)\n', ...
            cluster, nTotal, nPareto, round(100*nPareto/nTotal));
    end

    save(cacheFile, 'results');
    fprintf('已保存 %s\n', cacheFile);
    plotPareto(results, Ubar1, Ubar2, Tbar1, Tbar2, P_all, ...
        C_all, eta_all, idx, outDir);
end

%% ============================================================
%%  终端输出
%% ============================================================
fprintf('\n========== Pareto 前沿摘要 ==========\n');
fprintf('(C_out = C×1000×(1−η/100), 理论min=0, 历史≈49–50)\n\n');
fprintf('工况  H(°C)  C(g/Nm³)   C_out范围(mg/Nm³)   P范围(kW)     Pareto点数\n');
fprintf('----  ------  ---------  ------------------  ------------  ------------\n');
for k = 1:kOpt
    r = results{k};
    if isempty(r), continue; end
    cout_min = min(r.Cout_pareto);
    cout_max = max(r.Cout_pareto);
    p_min = min(r.P_pareto);
    p_max = max(r.P_pareto);
    fprintf('  %d   %5.1f   %6.2f     [%6.4f, %6.4f]    [%6.0f, %6.0f]    %4d\n', ...
        k, r.H_center, r.C_center, cout_min, cout_max, p_min, p_max, ...
        length(r.P_pareto));
end

% 各工况 Pareto 两端点
fprintf('\n========== 各工况 Pareto 两端（最佳 C_out vs 最佳 P）==========\n');
for k = 1:kOpt
    r = results{k};
    if isempty(r), continue; end
    % 最小 C_out 端
    [cout_min, idx_c] = min(r.Cout_pareto);
    % 最小 P 端
    [p_min, idx_p] = min(r.P_pareto);
    fprintf(['\n工况 %d (H≈%.1f, C≈%.2f):\n', ...
        '  最小 C_out 端: Ū₁=%.2f, Ū₂=%.2f, T̄₁=%.1f, T̄₂=%.1f → ', ...
        'C_out=%.4f, P=%.2f kW\n', ...
        '  最小 P 端:    Ū₁=%.2f, Ū₂=%.2f, T̄₁=%.1f, T̄₂=%.1f → ', ...
        'C_out=%.4f, P=%.2f kW\n'], ...
        k, r.H_center, r.C_center, ...
        r.U1_pareto(idx_c), r.U2_pareto(idx_c), ...
        r.T1_pareto(idx_c), r.T2_pareto(idx_c), ...
        cout_min, r.P_pareto(idx_c), ...
        r.U1_pareto(idx_p), r.U2_pareto(idx_p), ...
        r.T1_pareto(idx_p), r.T2_pareto(idx_p), ...
        r.Cout_pareto(idx_p), p_min);
end

fprintf('\n完成。\n');

%% ============================================================
%%  局部函数
%% ============================================================

function pareto_mask = nonDominated(P_vals, C_vals)
    % 双目标最小化非支配排序，O(n log n)
    %  点 A 支配 B ⇔ P_A ≤ P_B 且 C_A ≤ C_B 且至少一个严格 <
    n = length(P_vals);
    [~, order] = sortrows([P_vals(:), C_vals(:)]);
    P_sorted = P_vals(order);
    C_sorted = C_vals(order);

    pareto_sorted = false(n, 1);
    pareto_sorted(1) = true;
    best_C = C_sorted(1);

    for i = 2:n
        if C_sorted(i) < best_C
            pareto_sorted(i) = true;
            best_C = C_sorted(i);
        end
    end

    pareto_mask = false(n, 1);
    pareto_mask(order) = pareto_sorted;
end

function plotPareto(results, Ubar1, Ubar2, Tbar1, Tbar2, ...
        P_all, C_all, eta_all, idx, outDir)
    kOpt = length(results);

    for k = 1:kOpt
        r = results{k};
        if isempty(r), continue; end

        % 该工况历史实测点
        mask = idx == k;
        ok = isfinite(eta_all) & isfinite(C_all);
        mask = mask & ok;
        C_out_hist = C_all(mask) * 1000 .* (1 - eta_all(mask) / 100);
        P_hist     = P_all(mask);

        fig = figure('Name', sprintf('Pareto 前沿 — 工况 %d', k), ...
            'Position', [50, 50, 750, 560]);

        % 灰底：全部网格候选点
        scatter(r.Cout_all, r.P_all, 1, [0.75 0.75 0.75], 'filled', ...
            'MarkerFaceAlpha', 0.04);
        hold on;

        % 蓝点：历史实测数据
        scatter(C_out_hist, P_hist, 5, [0.2 0.5 0.8], 'filled', ...
            'MarkerFaceAlpha', 0.12, 'DisplayName', '历史实测');

        % 红线：Pareto 前沿
        [c_sort, ord] = sort(r.Cout_pareto);
        plot(c_sort, r.P_pareto(ord), 'r-o', 'LineWidth', 2.5, ...
            'MarkerSize', 6, 'MarkerFaceColor', 'r', ...
            'DisplayName', 'Pareto 前沿');

        % 标注两端
        [~, idx_c] = min(r.Cout_pareto);
        [~, idx_p] = min(r.P_pareto);
        plot(r.Cout_pareto(idx_c), r.P_pareto(idx_c), 'rv', ...
            'MarkerSize', 10, 'LineWidth', 1.5);
        plot(r.Cout_pareto(idx_p), r.P_pareto(idx_p), 'rs', ...
            'MarkerSize', 10, 'LineWidth', 1.5);

        text(r.Cout_pareto(idx_c), r.P_pareto(idx_c), ...
            sprintf('  最佳排放\n  C_{out}=%.4f\n  P=%.0f kW', ...
            r.Cout_pareto(idx_c), r.P_pareto(idx_c)), ...
            'FontSize', 8, 'VerticalAlignment', 'middle');
        text(r.Cout_pareto(idx_p), r.P_pareto(idx_p), ...
            sprintf('  最省电\n  C_{out}=%.4f\n  P=%.0f kW', ...
            r.Cout_pareto(idx_p), r.P_pareto(idx_p)), ...
            'FontSize', 8, 'VerticalAlignment', 'middle');

        xlabel('C_{out} (mg/Nm³)');
        ylabel('P (kW)');
        title(sprintf('工况 %d: H≈%.1f°C, C≈%.2f g/Nm³, Q≈%.0f Nm³/h', ...
            k, r.H_center, r.C_center, r.Q_center));
        legend('Location', 'best');
        grid on;
        saveCurrent(fig, sprintf('pareto_cluster%d', k), outDir);
    end

    % 汇总图（subplot 顺序：(1,4;3,2)，高C在上行、低C在下行）
    fig = figure('Name', 'Pareto 前沿汇总', 'Position', [50, 50, 1050, 850]);
    colors = lines(kOpt);
    subOrder = [1, 4, 3, 2];  % 位置1=工况1, 位置2=工况4, 位置3=工况3, 位置4=工况2
    for pos = 1:kOpt
        k = subOrder(pos);
        r = results{k};
        if isempty(r), continue; end
        subplot(2, 2, pos);

        % 该工况历史实测
        mask = idx == k;
        ok = isfinite(eta_all) & isfinite(C_all);
        mask = mask & ok;
        C_out_hist = C_all(mask) * 1000 .* (1 - eta_all(mask) / 100);
        P_hist = P_all(mask);

        scatter(r.Cout_all, r.P_all, 1, [0.75 0.75 0.75], 'filled', ...
            'MarkerFaceAlpha', 0.03);
        hold on;
        scatter(C_out_hist, P_hist, 3, [0.2 0.5 0.8], 'filled', ...
            'MarkerFaceAlpha', 0.08);
        [c_sort, ord] = sort(r.Cout_pareto);
        plot(c_sort, r.P_pareto(ord), '-o', 'Color', colors(k,:), ...
            'LineWidth', 1.8, 'MarkerSize', 3);
        xlabel('C_{out} (mg/Nm³)'); ylabel('P (kW)');
        title(sprintf('工况 %d (H≈%.1f, C≈%.2f)', k, r.H_center, r.C_center));
        grid on;
    end
    saveCurrent(fig, 'pareto_combined', outDir);
end

function saveCurrent(fig, name, outDir)
    saveas(fig, fullfile(outDir, [name '.svg']));
    saveas(fig, fullfile(outDir, [name '.fig']));
end

function openCachedFigs()
    outDir = fullfile('img', 'pro2opt_lease');
    for c = 1:4
        openfig(fullfile(outDir, sprintf('pareto_cluster%d.fig', c)), 'visible');
    end
    openfig(fullfile(outDir, 'pareto_combined.fig'), 'visible');
end
