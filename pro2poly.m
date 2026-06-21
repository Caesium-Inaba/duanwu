%% 第二问：二次多项式回归建模（替代 RF，支持外推）
%  f: η = f(H, C, Q, Ū₁, Ū₂, T̄₁, T̄₂)  全局二次多项式
%  gₖ: P = gₖ(Ū₁, Ū₂, T̄₁, T̄₂)         分工况二次多项式
clear;clc;close all;

if isfile('pro2poly.txt'), delete('pro2poly.txt'); end
diary('pro2poly.txt');

%% 目录与缓存策略
outDir = fullfile('img', 'pro2poly');
if ~exist(outDir, 'dir'), mkdir(outDir); end

cacheFile = 'poly_models.mat';

% 期望输出的所有 fig 文件
figNames = {'f_pred_vs_actual', 'f_oob_residuals', 'f_importance'};
for c = 1:4
    figNames{end+1} = sprintf('g%d_pred_vs_actual', c);
    figNames{end+1} = sprintf('g%d_importance', c);
    figNames{end+1} = sprintf('g%d_oob_residuals', c);
end

allFigsExist = all(cellfun(@(n) isfile(fullfile(outDir, [n '.fig'])), figNames));

%% 加载数据
load('dc.mat', 'data_clean');
load('dkmeans.mat', 'idx', 'kOpt', 'centers_H', 'centers_C');

fprintf('===== 第二问：二次多项式回归建模 =====\n\n');
fprintf('数据总量: %d 行\n', height(data_clean));
fprintf('模型: 完整二次型（含交互项）\n');

%% 变量合并
Ubar1 = (data_clean.U1_kV + data_clean.U2_kV) / 2;
Ubar2 = (data_clean.U3_kV + data_clean.U4_kV) / 2;
Tbar1 = (data_clean.T1_s  + data_clean.T2_s)  / 2;
Tbar2 = (data_clean.T3_s  + data_clean.T4_s)  / 2;

H = data_clean.Temp_C;
C_in = data_clean.C_in_gNm3;
Q = data_clean.Q_Nm3h;
P_total = data_clean.P_total_kW;
eta = data_clean.eff;

varNames_f = {'$H$','$C$','$Q$','$\bar{U}_1$','$\bar{U}_2$','$\bar{T}_1$','$\bar{T}_2$'};
varNames_g = {'$\bar{U}_1$','$\bar{U}_2$','$\bar{T}_1$','$\bar{T}_2$'};

%% ============================================================
%%  缓存判断
%% ============================================================
if allFigsExist
    load(cacheFile, 'f_mdl', 'g_mdls');
    fprintf('检测到所有 fig 已存在，跳过训练。\n');

elseif isfile(cacheFile)
    load(cacheFile, 'f_mdl', 'g_mdls');
    fprintf('检测到 poly_models.mat，加载模型并重新出图...\n');
    X_f_all = [H, C_in, Q, Ubar1, Ubar2, Tbar1, Tbar2];
    ok_f = all(isfinite(X_f_all), 2) & isfinite(eta);
    X_f_all = X_f_all(ok_f, :); y_f_all = eta(ok_f);
    rng(42); n_f = length(y_f_all);
    nTrain_f = round(0.8 * n_f); perm_f = randperm(n_f);
    X_test_f = X_f_all(perm_f(nTrain_f+1:end), :);
    y_test_f = y_f_all(perm_f(nTrain_f+1:end));
    X_train_f = X_f_all(perm_f(1:nTrain_f), :);
    plotF_poly_figs(f_mdl, X_test_f, y_test_f, X_train_f, varNames_f, outDir);
    X_g_all = [Ubar1, Ubar2, Tbar1, Tbar2];
    plotG_poly_figs(g_mdls, idx, kOpt, X_g_all, P_total, centers_H, centers_C, varNames_g, outDir);

else
    fprintf('从头训练二次多项式模型...\n');

    %% f 模型：η = f(H, C, Q, Ū₁, Ū₂, T̄₁, T̄₂)  全局二次
    fprintf('\n========== f 模型：η 全局二次多项式 ==========\n');
    X_f = [H, C_in, Q, Ubar1, Ubar2, Tbar1, Tbar2];
    y_f = eta;
    ok_f = all(isfinite(X_f), 2) & isfinite(y_f);
    X_f = X_f(ok_f, :); y_f = y_f(ok_f);
    n_f = length(y_f);
    fprintf('有效样本: %d\n', n_f);

    rng(42);
    nTrain_f = round(0.8 * n_f);
    perm_f = randperm(n_f);
    X_train_f = X_f(perm_f(1:nTrain_f), :); y_train_f = y_f(perm_f(1:nTrain_f));
    X_test_f  = X_f(perm_f(nTrain_f+1:end), :); y_test_f  = y_f(perm_f(nTrain_f+1:end));
    fprintf('训练: %d, 测试: %d, 参数: 36\n', nTrain_f, n_f - nTrain_f);

    f_mdl = fitlm(X_train_f, y_train_f, 'quadratic');
    y_pred_f = predict(f_mdl, X_test_f);
    y_pred_train_f = predict(f_mdl, X_train_f);
    R2_f = 1 - sum((y_test_f - y_pred_f).^2) / sum((y_test_f - mean(y_test_f)).^2);
    R2_train_f = 1 - sum((y_train_f - y_pred_train_f).^2) / sum((y_train_f - mean(y_train_f)).^2);
    RMSE_f = sqrt(mean((y_test_f - y_pred_f).^2));
    deltaR2_f = R2_train_f - R2_f;

    fprintf('训练 R² = %.4f, 测试 R² = %.4f, ΔR² = %.4f', R2_train_f, R2_f, deltaR2_f);
    if deltaR2_f > 0.02, fprintf(' ⚠ 过拟合风险'); end
    fprintf('\nRMSE = %.4f\n', RMSE_f);

    plotF_poly_figs(f_mdl, X_test_f, y_test_f, X_train_f, varNames_f, outDir);

    %% gₖ 模型：P = gₖ(Ū₁, Ū₂, T̄₁, T̄₂)  分工况二次
    fprintf('\n========== g 模型：P 分工况二次多项式 ==========\n');
    X_g_all = [Ubar1, Ubar2, Tbar1, Tbar2]; y_g_all = P_total;
    g_mdls = cell(kOpt, 1);

    for cluster = 1:kOpt
        fprintf('\n-- 工况 %d (H=%.1f, C=%.2f) --\n', cluster, ...
            centers_H(cluster), centers_C(cluster));
        mask = idx == cluster;
        X_k = X_g_all(mask, :); y_k = y_g_all(mask);
        ok = all(isfinite(X_k), 2) & isfinite(y_k);
        X_k = X_k(ok, :); y_k = y_k(ok);
        n_g = length(y_k);
        fprintf('  有效样本: %d\n', n_g);
        if n_g < 50, fprintf('  样本过少，跳过\n'); continue; end

        rng(42 + cluster);
        nTrain_g = round(0.8 * n_g);
        perm_g = randperm(n_g);
        X_train_g = X_k(perm_g(1:nTrain_g), :); y_train_g = y_k(perm_g(1:nTrain_g));
        X_test_g  = X_k(perm_g(nTrain_g+1:end), :); y_test_g  = y_k(perm_g(nTrain_g+1:end));
        fprintf('  训练: %d, 测试: %d, 参数: 15\n', nTrain_g, n_g - nTrain_g);

        g_mdl = fitlm(X_train_g, y_train_g, 'quadratic');
        g_mdls{cluster} = g_mdl;

        y_pred_g = predict(g_mdl, X_test_g);
        y_pred_train_g = predict(g_mdl, X_train_g);
        R2_g = 1 - sum((y_test_g - y_pred_g).^2) / sum((y_test_g - mean(y_test_g)).^2);
        R2_train_g = 1 - sum((y_train_g - y_pred_train_g).^2) / sum((y_train_g - mean(y_train_g)).^2);
        deltaR2_g = R2_train_g - R2_g;
        RMSE_g = sqrt(mean((y_test_g - y_pred_g).^2));

        fprintf('  训练 R² = %.4f, 测试 R² = %.4f, ΔR² = %.4f', R2_train_g, R2_g, deltaR2_g);
        if deltaR2_g > 0.03, fprintf(' ⚠ 过拟合风险'); end
        fprintf('\n  RMSE = %.4f kW\n', RMSE_g);
    end

    plotG_poly_figs(g_mdls, idx, kOpt, X_g_all, P_total, centers_H, centers_C, varNames_g, outDir);

    %% 保存模型
    fprintf('\n========== 保存模型 ==========\n');
    save(cacheFile, 'f_mdl', 'g_mdls', 'varNames_f', 'varNames_g', 'centers_H', 'centers_C', 'kOpt');
    fprintf('已保存 %s\n', cacheFile);
end

%% R² 汇总
fprintf('\n========== R² 汇总 ==========\n');
if ~allFigsExist
    fprintf('f (η, poly):  训练R²=%.4f  测试R²=%.4f  Δ=%.4f  RMSE=%.4f\n', ...
        R2_train_f, R2_f, deltaR2_f, RMSE_f);
end
fprintf('(多项式系数及 p 值详见 diary)\n');

diary off;
fprintf('\n完成。\n');

%% ============================================================
%%  局部函数
%% ============================================================

function plotF_poly_figs(mdl, X_test, y_test, X_train, varNames_f, outDir)
    y_pred = predict(mdl, X_test);
    res = y_test - y_pred;
    R2 = 1 - sum(res.^2) / sum((y_test - mean(y_test)).^2);
    RMSE = sqrt(mean(res.^2));

    % 图 1：预测 vs 实测
    fig = figure('Name', 'f 模型 (Poly)：η 预测 vs 实测', 'Position', [50, 50, 600, 550]);
    scatter(y_test, y_pred, 8, [0.2 0.4 0.7], 'filled', 'MarkerFaceAlpha', 0.3);
    hold on;
    lims = [min([y_test; y_pred]), max([y_test; y_pred])];
    plot(lims, lims, 'k--', 'LineWidth', 1); hold off;
    xlabel('实测 $\eta$ (%)'); ylabel('预测 $\eta$ (%)');
    grid on; axis equal tight;
    text(0.05, 0.95, sprintf('R^2 = %.4f\nRMSE = %.4f', R2, RMSE), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 10);
    saveCurrent(fig, 'f_pred_vs_actual', outDir);

    % 图 2：残差直方图
    fig = figure('Name', 'f 模型 (Poly)：残差分布', 'Position', [100, 100, 500, 400]);
    histogram(res, 40, 'FaceColor', [0.2 0.4 0.7], 'EdgeColor', 'none');
    hold on; xline(0, 'k--', 'LineWidth', 1); hold off;
    xlabel('残差 $\eta$ (%)'); ylabel('频数');
    text(0.05, 0.95, sprintf('\\mu = %.4f\n\\sigma = %.4f', mean(res), std(res)), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 9);
    grid on;
    saveCurrent(fig, 'f_oob_residuals', outDir);

    % 图 3：系数 t 统计量（替代 RF 的重要性）
    fig = figure('Name', 'f 模型 (Poly)：系数 |t| 统计量', 'Position', [150, 150, 700, 420]);
    coefNames = mdl.CoefficientNames;
    tStats = abs(mdl.Coefficients.tStat);
    % 去掉截距项
    coefNames = coefNames(2:end); tStats = tStats(2:end);
    [~, ord] = sort(tStats, 'descend');
    bar(tStats(ord), 'FaceColor', [0.2 0.4 0.7]);
    % 只标注前几个重要项
    nShow = min(15, length(coefNames));
    set(gca, 'XTick', 1:nShow, 'XTickLabel', coefNames(ord(1:nShow)));
    xtickangle(45);
    ylabel('|t| 统计量'); grid on;
    saveCurrent(fig, 'f_importance', outDir);
end

function plotG_poly_figs(g_mdls, idx, kOpt, X_g_all, y_g_all, centers_H, centers_C, varNames_g, outDir)
    for cluster = 1:kOpt
        g_mdl = g_mdls{cluster};
        if isempty(g_mdl), continue; end
        mask = idx == cluster;
        X_k = X_g_all(mask, :); y_k = y_g_all(mask);
        ok = all(isfinite(X_k), 2) & isfinite(y_k);
        X_k = X_k(ok, :); y_k = y_k(ok);

        rng(42 + cluster);
        n_g = length(y_k);
        nTrain_g = round(0.8 * n_g);
        perm_g = randperm(n_g);
        X_test_g = X_k(perm_g(nTrain_g+1:end), :); y_test_g = y_k(perm_g(nTrain_g+1:end));

        y_pred = predict(g_mdl, X_test_g);
        res = y_test_g - y_pred;
        R2 = 1 - sum(res.^2) / sum((y_test_g - mean(y_test_g)).^2);
        RMSE = sqrt(mean(res.^2));

        % 图 1：预测 vs 实测
        fig = figure('Name', sprintf('g%d (Poly)：P 预测 vs 实测', cluster), ...
            'Position', [50, 50, 600, 550]);
        scatter(y_test_g, y_pred, 8, [0.8 0.3 0.2], 'filled', 'MarkerFaceAlpha', 0.3);
        hold on;
        lims_g = [min([y_test_g; y_pred]), max([y_test_g; y_pred])];
        plot(lims_g, lims_g, 'k--', 'LineWidth', 1); hold off;
        xlabel('实测 $P$ (kW)'); ylabel('预测 $P$ (kW)');
        text(0.05, 0.95, sprintf('R^2 = %.4f\nRMSE = %.4f', R2, RMSE), ...
            'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 10);
        grid on; axis equal tight;
        saveCurrent(fig, sprintf('g%d_pred_vs_actual', cluster), outDir);

        % 图 2：系数 |t|
        fig = figure('Name', sprintf('g%d (Poly)：系数 |t| 统计量', cluster), ...
            'Position', [150, 150, 500, 380]);
        coefNames = g_mdl.CoefficientNames;
        tStats = abs(g_mdl.Coefficients.tStat);
        coefNames = coefNames(2:end); tStats = tStats(2:end);
        [~, ord] = sort(tStats, 'descend');
        bar(tStats(ord), 'FaceColor', [0.8 0.3 0.2]);
        set(gca, 'XTick', 1:length(coefNames), 'XTickLabel', coefNames(ord));
        xtickangle(45);
        ylabel('|t| 统计量'); grid on;
        saveCurrent(fig, sprintf('g%d_importance', cluster), outDir);

        % 图 3：残差直方图
        fig = figure('Name', sprintf('g%d (Poly)：残差分布', cluster), ...
            'Position', [100, 100, 500, 400]);
        histogram(res, 30, 'FaceColor', [0.8 0.3 0.2], 'EdgeColor', 'none');
        hold on; xline(0, 'k--', 'LineWidth', 1); hold off;
        xlabel('残差 $P$ (kW)'); ylabel('频数');
        text(0.05, 0.95, sprintf('\\mu = %.4f\n\\sigma = %.4f', mean(res), std(res)), ...
            'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 9);
        grid on;
        saveCurrent(fig, sprintf('g%d_oob_residuals', cluster), outDir);
    end
end
