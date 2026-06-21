%% 第二问：随机森林建模
%  f: η = f(H, C, Q, Ū₁, Ū₂, T̄₁, T̄₂)  全局 RF
clear;clc;close all;

if isfile('pro2RF.txt'), delete('pro2RF.txt'); end
diary('pro2RF.txt');

%% 目录与缓存策略
outDir = fullfile('img', 'pro2RF');
if ~exist(outDir, 'dir'), mkdir(outDir); end

cacheFile = 'rf_models.mat';

% 期望输出的所有 fig 文件
figNames = {'f_pred_vs_actual', 'f_oob_residuals', 'f_importance'};

allFigsExist = all(cellfun(@(n) isfile(fullfile(outDir, [n '.fig'])), figNames));
cacheExists = isfile(cacheFile);

%% 加载数据（无论如何都需要，变量合并和元数据来自这里）
load('dc.mat', 'data_clean');

fprintf('===== 第二问：随机森林建模 =====\n\n');
fprintf('数据总量: %d 行\n', height(data_clean));

%% 变量合并（减少共线性，降维）
Ubar1 = (data_clean.U1_kV + data_clean.U2_kV) / 2;
Ubar2 = (data_clean.U3_kV + data_clean.U4_kV) / 2;
Tbar1 = (data_clean.T1_s  + data_clean.T2_s)  / 2;
Tbar2 = (data_clean.T3_s  + data_clean.T4_s)  / 2;

H = data_clean.Temp_C;
C_in = data_clean.C_in_gNm3;
Q = data_clean.Q_Nm3h;
eta = data_clean.eff;

varNames_f = {'$H$','$C$','$Q$','$\bar{U}_1$','$\bar{U}_2$','$\bar{T}_1$','$\bar{T}_2$'};
nTrees = 200;


%%  缓存判断

if allFigsExist
    % 情况 1：fig 全部存在，直接跳过
    load(cacheFile, 'f_model');
    R2_train_f = NaN; R2_f = NaN; deltaR2_f = NaN; RMSE_f = NaN;
    fprintf('检测到所有 fig 已存在，跳过训练。\n');
    fprintf('从 rf_models.mat 加载模型。\n');

elseif cacheExists
    % 情况 2：rf_models.mat 存在但 fig 不完整 → 加载模型，重出图
    load(cacheFile, 'f_model');
    fprintf('检测到 rf_models.mat，加载模型并重新出图...\n');

    % 重算评估指标用于出图
    X_f = [H, C_in, Q, Ubar1, Ubar2, Tbar1, Tbar2];
    y_f = eta;
    ok_f = all(isfinite(X_f), 2) & isfinite(y_f);
    X_f = X_f(ok_f, :); y_f = y_f(ok_f);
    rng(42);
    n_f = length(y_f);
    nTrain_f = round(0.8 * n_f);
    perm_f = randperm(n_f);
    idxTest_f = perm_f(nTrain_f+1:end);
    X_test_f = X_f(idxTest_f, :); y_test_f = y_f(idxTest_f);
    X_train_f = X_f(perm_f(1:nTrain_f), :); y_train_f = y_f(perm_f(1:nTrain_f));
    y_pred_f = predict(f_model, X_test_f);
    R2_f = 1 - sum((y_test_f - y_pred_f).^2) / sum((y_test_f - mean(y_test_f)).^2);
    RMSE_f = sqrt(mean((y_test_f - y_pred_f).^2));
    res_f = y_test_f - y_pred_f;
    y_pred_train_f = predict(f_model, X_train_f);
    R2_train_f = 1 - sum((y_train_f - y_pred_train_f).^2) / sum((y_train_f - mean(y_train_f)).^2);
    deltaR2_f = R2_train_f - R2_f;

    % 出 f 图
    plotF_figs(f_model, y_test_f, y_pred_f, res_f, X_train_f, varNames_f, R2_f, RMSE_f, outDir, nTrees);

else
    % 情况 3：从头训练
    fprintf('从头训练随机森林模型...\n');

    %% f 模型：η = f(H, C, Q, Ū₁, Ū₂, T̄₁, T̄₂)  全局 RF
    fprintf('\n========== f 模型：η 全局 RF ==========\n');
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
    fprintf('训练: %d, 测试: %d\n', nTrain_f, n_f - nTrain_f);

    f_model = TreeBagger(nTrees, X_train_f, y_train_f, ...
        'Method', 'regression', 'OOBPrediction', 'on', ...
        'OOBPredictorImportance', 'on', 'MinLeafSize', 5);

    y_pred_f = predict(f_model, X_test_f);
    y_pred_train_f = predict(f_model, X_train_f);
    R2_f = 1 - sum((y_test_f - y_pred_f).^2) / sum((y_test_f - mean(y_test_f)).^2);
    R2_train_f = 1 - sum((y_train_f - y_pred_train_f).^2) / sum((y_train_f - mean(y_train_f)).^2);
    RMSE_f = sqrt(mean((y_test_f - y_pred_f).^2));
    MAE_f = mean(abs(y_test_f - y_pred_f));
    res_f = y_test_f - y_pred_f;
    deltaR2_f = R2_train_f - R2_f;

    fprintf('训练 R² = %.4f, 测试 R² = %.4f, ΔR² = %.4f', R2_train_f, R2_f, deltaR2_f);
    if deltaR2_f > 0.02, fprintf(' ⚠ 过拟合风险'); end
    fprintf('\nRMSE = %.4f, MAE = %.4f\n', RMSE_f, MAE_f);

    plotF_figs(f_model, y_test_f, y_pred_f, res_f, X_train_f, varNames_f, R2_f, RMSE_f, outDir, nTrees);

    %% 保存模型
    fprintf('\n========== 保存模型 ==========\n');
    save(cacheFile, 'f_model', 'varNames_f');
    fprintf('已保存 %s\n', cacheFile);
end

%% R² 汇总
fprintf('\n========== R² 汇总 ==========\n');
if allFigsExist
    fprintf('(从缓存加载，跳过训练指标)\n');
else
    fprintf('f (η):  训练R²=%.4f  测试R²=%.4f  Δ=%.4f  RMSE=%.4f\n', ...
        R2_train_f, R2_f, deltaR2_f, RMSE_f);
end

fprintf('\n完成。\n');
diary off;


%%  辅助函数


function plotF_figs(model, y_test, y_pred, res, X_train, varNames_f, R2, RMSE, outDir, nTrees)
    MAE = mean(abs(res));

    % f 检验图 1：预测 vs 实测
    fig = figure('Name', 'f 模型：η 预测 vs 实测', 'Position', [50, 50, 600, 550]);
    scatter(y_test, y_pred, 8, [0.2 0.4 0.7], 'filled', 'MarkerFaceAlpha', 0.3);
    hold on;
    lims = [min([y_test; y_pred]), max([y_test; y_pred])];
    plot(lims, lims, 'k--', 'LineWidth', 1);
    hold off;
    xlabel('实测 $\eta$ (%)'); ylabel('预测 $\eta$ (%)');
    grid on; axis equal tight;
    text(0.05, 0.95, sprintf('R^2 = %.4f\nRMSE = %.4f\nMAE = %.4f', R2, RMSE, MAE), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 10);
    saveCurrent(fig, 'f_pred_vs_actual', outDir);

    % f 检验图 2：OOB + 残差
    fig = figure('Name', 'f 模型：OOB Error 与残差分布', 'Position', [100, 100, 900, 400]);
    subplot(1,2,1);
    oobErr = oobError(model);
    plot(1:nTrees, oobErr, 'b-', 'LineWidth', 1.2);
    xlabel('树棵数'); ylabel('OOB MSE'); grid on;
    subplot(1,2,2);
    histogram(res, 40, 'FaceColor', [0.2 0.4 0.7], 'EdgeColor', 'none');
    hold on; xline(0, 'k--', 'LineWidth', 1); hold off;
    xlabel('残差 $\eta$ (%)'); ylabel('频数');
    mu_res = mean(res); sigma_res = std(res);
    text(0.05, 0.95, sprintf('\\mu = %.4f\n\\sigma = %.4f', mu_res, sigma_res), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 9);
    grid on;
    saveCurrent(fig, 'f_oob_residuals', outDir);

    % f 检验图 3：变量重要性
    fig = figure('Name', 'f 模型：变量重要性', 'Position', [150, 150, 600, 420]);
    imp = model.OOBPermutedVarDeltaError;
    [~, ord] = sort(imp, 'descend');
    bar(imp(ord), 'FaceColor', [0.2 0.4 0.7]);
    set(gca, 'XTickLabel', varNames_f(ord));
    xtickangle(30);
    ylabel('OOB 变量重要性 (\Delta MSE)'); grid on;
    saveCurrent(fig, 'f_importance', outDir);
end

