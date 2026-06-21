%% 第二问：网格搜索优化
%  对各工况，在数据范围内搜索 [Ū₁, Ū₂, T̄₁, T̄₂]
%  使 P 最小，同时满足 η ≥ 100 - 1/Cₖ
%  方法：粗搜 (20⁴) → 局域加密 (20⁴)
clear;clc;close all;

if isfile('pro2opt.txt'), delete('pro2opt.txt'); end
diary('pro2opt.txt');

%% 加载模型与数据
if ~isfile('rf_models.mat')
    error('rf_models.mat 不存在，请先运行 pro2RF.m');
end
load('rf_models.mat', 'f_model', 'g_models', 'varNames_f', 'varNames_g', 'nTrees', ...
    'centers_H', 'centers_C', 'kOpt');
load('dc.mat', 'data_clean');
load('dkmeans.mat', 'idx');

fprintf('===== 第二问：网格搜索优化 =====\n\n');

%% 变量合并
Ubar1 = (data_clean.U1_kV + data_clean.U2_kV) / 2;
Ubar2 = (data_clean.U3_kV + data_clean.U4_kV) / 2;
Tbar1 = (data_clean.T1_s  + data_clean.T2_s)  / 2;
Tbar2 = (data_clean.T3_s  + data_clean.T4_s)  / 2;
H_all = data_clean.Temp_C;
C_all = data_clean.C_in_gNm3;
Q_all = data_clean.Q_Nm3h;

%% 优化参数
nGridCoarse = 20;   % 粗搜每维点数
nGridFine   = 20;   % 加密每维点数
fineRange   = 0.10; % 加密范围 ±10%

%% ============================================================
%%  对每个工况进行网格搜索
%% ============================================================
results = cell(kOpt, 1);

for cluster = 1:kOpt
    fprintf('\n========== 工况 %d ==========\n', cluster);
    fprintf('中心: H = %.1f °C, C = %.2f g/Nm³\n', centers_H(cluster), centers_C(cluster));

    % 该工况数据
    mask = idx == cluster;
    H_k = centers_H(cluster);
    C_k = centers_C(cluster);
    Q_k = mean(Q_all(mask), 'omitnan');
    fprintf('Q 中心 = %.1f Nm³/h\n', Q_k);

    % 该工况内各变量的 min / max
    Ubar1_min = min(Ubar1(mask)); Ubar1_max = max(Ubar1(mask));
    Ubar2_min = min(Ubar2(mask)); Ubar2_max = max(Ubar2(mask));
    Tbar1_min = min(Tbar1(mask)); Tbar1_max = max(Tbar1(mask));
    Tbar2_min = min(Tbar2(mask)); Tbar2_max = max(Tbar2(mask));

    fprintf('变量范围:\n');
    fprintf('  Ū₁: [%.1f, %.1f] kV\n', Ubar1_min, Ubar1_max);
    fprintf('  Ū₂: [%.1f, %.1f] kV\n', Ubar2_min, Ubar2_max);
    fprintf('  T̄₁: [%.0f, %.0f] s\n', Tbar1_min, Tbar1_max);
    fprintf('  T̄₂: [%.0f, %.0f] s\n', Tbar2_min, Tbar2_max);

    % 约束：η ≥ 100 - 1/C
    eta_min = 100 - 1 / C_k;
    fprintf('约束: η ≥ %.4f%%  (C_out ≤ 10 mg/Nm³)\n', eta_min);

    % ---- 粗搜 ----
    fprintf('粗搜 (%d^4 = %d 候选点)...\n', nGridCoarse, nGridCoarse^4);
    tic;
    [X_coarse, P_coarse, eta_coarse, nFeas_c] = gridSearch(...
        [Ubar1_min, Ubar1_max], [Ubar2_min, Ubar2_max], ...
        [Tbar1_min, Tbar1_max], [Tbar2_min, Tbar2_max], nGridCoarse, ...
        f_model, g_models{cluster}, H_k, C_k, Q_k, eta_min);
    t_coarse = toc;

    fprintf('  耗时 %.1f s, 可行点: %d / %d (%.1f%%)\n', ...
        t_coarse, nFeas_c, nGridCoarse^4, 100*nFeas_c/nGridCoarse^4);
    fprintf('  粗搜最优: Ū₁=%.2f, Ū₂=%.2f, T̄₁=%.1f, T̄₂=%.1f\n', X_coarse);
    fprintf('            P=%.2f kW, η=%.4f%%\n', P_coarse, eta_coarse);

    % ---- 局域加密 ----
    u1_fine = [max(Ubar1_min, X_coarse(1)*(1-fineRange)), min(Ubar1_max, X_coarse(1)*(1+fineRange))];
    u2_fine = [max(Ubar2_min, X_coarse(2)*(1-fineRange)), min(Ubar2_max, X_coarse(2)*(1+fineRange))];
    t1_fine = [max(Tbar1_min, X_coarse(3)*(1-fineRange)), min(Tbar1_max, X_coarse(3)*(1+fineRange))];
    t2_fine = [max(Tbar2_min, X_coarse(4)*(1-fineRange)), min(Tbar2_max, X_coarse(4)*(1+fineRange))];

    fprintf('加密搜索 (%d^4)...\n', nGridFine);
    tic;
    [X_fine, P_fine, eta_fine, nFeas_f] = gridSearch(...
        u1_fine, u2_fine, t1_fine, t2_fine, nGridFine, ...
        f_model, g_models{cluster}, H_k, C_k, Q_k, eta_min);
    t_fine = toc;

    fprintf('  耗时 %.1f s, 可行点: %d / %d (%.1f%%)\n', ...
        t_fine, nFeas_f, nGridFine^4, 100*nFeas_f/nGridFine^4);
    fprintf('  加密最优: Ū₁=%.2f, Ū₂=%.2f, T̄₁=%.1f, T̄₂=%.1f\n', X_fine);
    fprintf('            P=%.2f kW, η=%.4f%%\n', P_fine, eta_fine);

    % 保存结果
    results{cluster} = struct(...
        'cluster', cluster, ...
        'H_center', H_k, 'C_center', C_k, 'Q_center', Q_k, ...
        'eta_min', eta_min, ...
        'Ubar1_opt', X_fine(1), 'Ubar2_opt', X_fine(2), ...
        'Tbar1_opt', X_fine(3), 'Tbar2_opt', X_fine(4), ...
        'P_min', P_fine, 'eta_opt', eta_fine, ...
        'Ubar1_range', [Ubar1_min, Ubar1_max], ...
        'Ubar2_range', [Ubar2_min, Ubar2_max], ...
        'Tbar1_range', [Tbar1_min, Tbar1_max], ...
        'Tbar2_range', [Tbar2_min, Tbar2_max]);
end

%% ============================================================
%%  结果汇总表
%% ============================================================
fprintf('\n\n');
fprintf('╔══════════════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                        第二问优化结果汇总表                                  ║\n');
fprintf('╠══════╤════════╤════════╤════════╤════════╤════════╤════════╤════════╤══════╣\n');
fprintf('║ 工况 │  H(°C) │ C(g/m³)│Ū₁(kV) │Ū₂(kV) │T̄₁(s)  │T̄₂(s)  │P_min(kW)│ η(%) ║\n');
fprintf('╟──────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼──────╢\n');

for k = 1:kOpt
    r = results{k};
    fprintf('║  %d   │ %6.1f │ %6.2f │ %6.2f │ %6.2f │ %6.0f │ %6.0f │ %7.2f │%6.2f ║\n', ...
        r.cluster, r.H_center, r.C_center, ...
        r.Ubar1_opt, r.Ubar2_opt, r.Tbar1_opt, r.Tbar2_opt, ...
        r.P_min, r.eta_opt);
end

fprintf('╚══════╧════════╧════════╧════════╧════════╧════════╧════════╧════════╧══════╝\n');

%% 各工况详细说明
fprintf('\n各工况最优解详细说明:\n');
for k = 1:kOpt
    r = results{k};
    fprintf('\n工况 %d: H=%.1f°C, C=%.2f g/Nm³, Q=%.1f Nm³/h\n', ...
        k, r.H_center, r.C_center, r.Q_center);
    fprintf('  搜索范围:\n');
    fprintf('    Ū₁ ∈ [%.1f, %.1f] kV\n', r.Ubar1_range);
    fprintf('    Ū₂ ∈ [%.1f, %.1f] kV\n', r.Ubar2_range);
    fprintf('    T̄₁ ∈ [%.0f, %.0f] s\n', r.Tbar1_range);
    fprintf('    T̄₂ ∈ [%.0f, %.0f] s\n', r.Tbar2_range);
    fprintf('  最优解:\n');
    fprintf('    Ū₁ = %.2f kV, Ū₂ = %.2f kV\n', r.Ubar1_opt, r.Ubar2_opt);
    fprintf('    T̄₁ = %.0f s, T̄₂ = %.0f s\n', r.Tbar1_opt, r.Tbar2_opt);
    fprintf('    P_min = %.2f kW, η = %.4f%%\n', r.P_min, r.eta_opt);
    fprintf('    约束 η ≥ %.4f%% → 边距 %.4f 百分点\n', r.eta_min, r.eta_opt - r.eta_min);
end

diary off;
fprintf('\n完成。\n');

%% ============================================================
%%  局部函数：向量化网格搜索
%% ============================================================
function [X_opt, P_opt, eta_opt, nFeas] = gridSearch(...
        u1_range, u2_range, t1_range, t2_range, nPts, ...
        f_model, g_model, H_k, C_k, Q_k, eta_min)

    u1_v = linspace(u1_range(1), u1_range(2), nPts);
    u2_v = linspace(u2_range(1), u2_range(2), nPts);
    t1_v = linspace(t1_range(1), t1_range(2), nPts);
    t2_v = linspace(t2_range(1), t2_range(2), nPts);

    [G1, G2, G3, G4] = ndgrid(u1_v, u2_v, t1_v, t2_v);
    nTotal = numel(G1);

    % 构造 f 模型输入矩阵：(H, C, Q, Ū₁, Ū₂, T̄₁, T̄₂)
    X_f_all = [repmat([H_k, C_k, Q_k], nTotal, 1), G1(:), G2(:), G3(:), G4(:)];

    % 向量化预测 η
    eta_all = predict(f_model, X_f_all);

    % 过滤可行点
    feasible = eta_all >= eta_min;
    nFeas = sum(feasible);

    if nFeas == 0
        X_opt = [NaN, NaN, NaN, NaN];
        P_opt = NaN;
        eta_opt = NaN;
        return;
    end

    % 对可行点预测 P
    X_g_feas = [G1(feasible), G2(feasible), G3(feasible), G4(feasible)];
    P_feas = predict(g_model, X_g_feas);

    % 取 P 最小者
    [P_opt, idxMin] = min(P_feas);
    X_opt = X_g_feas(idxMin, :);
    eta_opt = eta_all(feasible);
    eta_opt = eta_opt(idxMin);
end
