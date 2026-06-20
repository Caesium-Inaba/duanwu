%% 第二问：随机森林建模
%  f: η = f(H, C, Q, Ū₁, Ū₂, T̄₁, T̄₂)  全局 RF
%  gₖ: P = gₖ(Ū₁, Ū₂, T̄₁, T̄₂)         分工况 RF, k=1..4
clear;clc;close all;

if isfile('pro2RF.txt'), delete('pro2RF.txt'); end
diary('pro2RF.txt');

%% 目录
outDir = fullfile('img', 'pro2RF');
if ~exist(outDir, 'dir'), mkdir(outDir); end

%% 加载数据
load('dc.mat', 'data_clean');
load('dkmeans.mat', 'idx', 'kOpt', 'centers_H', 'centers_C');

fprintf('===== 第二问：随机森林建模 =====\n\n');
fprintf('数据总量: %d 行\n', height(data_clean));

%% 变量合并（减少共线性，降维）
%  Ū₁ = (U1+U2)/2   Ū₂ = (U3+U4)/2
%  T̄₁ = (T1+T2)/2   T̄₂ = (T3+T4)/2
Ubar1 = (data_clean.U1_kV + data_clean.U2_kV) / 2;
Ubar2 = (data_clean.U3_kV + data_clean.U4_kV) / 2;
Tbar1 = (data_clean.T1_s  + data_clean.T2_s)  / 2;
Tbar2 = (data_clean.T3_s  + data_clean.T4_s)  / 2;

H = data_clean.Temp_C;
C_in = data_clean.C_in_gNm3;
Q = data_clean.Q_Nm3h;
P_total = data_clean.P_total_kW;
eta = data_clean.eff;

%% ============================================================
%%  f 模型：η = f(H, C, Q, Ū₁, Ū₂, T̄₁, T̄₂)  全局 RF
%% ============================================================
fprintf('\n========== f 模型：η 全局 RF ==========\n');

X_f = [H, C_in, Q, Ubar1, Ubar2, Tbar1, Tbar2];
y_f = eta;
varNames_f = {'$H$','$C$','$Q$','$\bar{U}_1$','$\bar{U}_2$','$\bar{T}_1$','$\bar{T}_2$'};

% 剔除含 NaN 的行
ok_f = all(isfinite(X_f), 2) & isfinite(y_f);
X_f = X_f(ok_f, :);
y_f = y_f(ok_f);
n_f = length(y_f);
fprintf('有效样本: %d\n', n_f);

% 80/20 划分
rng(42);
nTrain_f = round(0.8 * n_f);
perm_f = randperm(n_f);
idxTrain_f = perm_f(1:nTrain_f);
idxTest_f  = perm_f(nTrain_f+1:end);

X_train_f = X_f(idxTrain_f, :);
y_train_f = y_f(idxTrain_f);
X_test_f  = X_f(idxTest_f, :);
y_test_f  = y_f(idxTest_f);
fprintf('训练: %d, 测试: %d\n', nTrain_f, n_f - nTrain_f);

% 训练随机森林
nTrees = 200;
fprintf('训练中 (nTrees=%d)...\n', nTrees);
f_model = TreeBagger(nTrees, X_train_f, y_train_f, ...
    'Method', 'regression', ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance', 'on', ...
    'MinLeafSize', 5);

% 预测与评估
y_pred_f = predict(f_model, X_test_f);
R2_f = 1 - sum((y_test_f - y_pred_f).^2) / sum((y_test_f - mean(y_test_f)).^2);
RMSE_f = sqrt(mean((y_test_f - y_pred_f).^2));
MAE_f = mean(abs(y_test_f - y_pred_f));
res_f = y_test_f - y_pred_f;

fprintf('R² = %.4f, RMSE = %.4f, MAE = %.4f\n', R2_f, RMSE_f, MAE_f);

%% f 模型：检验图 1 — 预测 vs 实测散点图
fig = figure('Name', 'f 模型：η 预测 vs 实测', 'Position', [50, 50, 600, 550]);
scatter(y_test_f, y_pred_f, 8, [0.2 0.4 0.7], 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
lims = [min([y_test_f; y_pred_f]), max([y_test_f; y_pred_f])];
plot(lims, lims, 'k--', 'LineWidth', 1);
hold off;
xlabel('实测 $\eta$ (%)'); ylabel('预测 $\eta$ (%)');
grid on; axis equal tight;
text(0.05, 0.95, sprintf('R^2 = %.4f\nRMSE = %.4f\nMAE = %.4f', R2_f, RMSE_f, MAE_f), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 10);
saveCurrent(fig, 'f_pred_vs_actual', outDir);

%% f 模型：检验图 2 — OOB error 曲线 + 残差直方图
fig = figure('Name', 'f 模型：OOB Error 与残差分布', 'Position', [100, 100, 900, 400]);

% 左：OOB error
subplot(1,2,1);
oobErr_f = oobError(f_model);
plot(1:nTrees, oobErr_f, 'b-', 'LineWidth', 1.2);
xlabel('树棵数'); ylabel('OOB MSE');
grid on;

% 右：残差直方图
subplot(1,2,2);
histogram(res_f, 40, 'FaceColor', [0.2 0.4 0.7], 'EdgeColor', 'none');
hold on; xline(0, 'k--', 'LineWidth', 1); hold off;
xlabel('残差 $\eta$ (%)'); ylabel('频数');
mu_res = mean(res_f); sigma_res = std(res_f);
text(0.05, 0.95, sprintf('\\mu = %.4f\n\\sigma = %.4f', mu_res, sigma_res), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 9);
grid on;

saveCurrent(fig, 'f_oob_residuals', outDir);

%% f 模型：检验图 3 — 变量重要性
fig = figure('Name', 'f 模型：变量重要性', 'Position', [150, 150, 600, 420]);
imp_f = f_model.OOBPermutedVarDeltaError;
[~, impOrder_f] = sort(imp_f, 'descend');
bar(imp_f(impOrder_f), 'FaceColor', [0.2 0.4 0.7]);
set(gca, 'XTickLabel', varNames_f(impOrder_f));
xtickangle(30);
ylabel('OOB 变量重要性 (\Delta MSE)');
grid on;
saveCurrent(fig, 'f_importance', outDir);

fprintf('f 模型完成。\n');

%% ============================================================
%%  gₖ 模型：P = gₖ(Ū₁, Ū₂, T̄₁, T̄₂)  分工况 RF
%% ============================================================
fprintf('\n========== g 模型：P 分工况 RF ==========\n');

X_g_all = [Ubar1, Ubar2, Tbar1, Tbar2];
y_g_all = P_total;
varNames_g = {'$\bar{U}_1$','$\bar{U}_2$','$\bar{T}_1$','$\bar{T}_2$'};

g_models = cell(kOpt, 1);

for cluster = 1:kOpt
    clusterLabel = sprintf('工况 %d', cluster);
    fprintf('\n-- %s (H=%.1f, C=%.2f) --\n', clusterLabel, ...
        centers_H(cluster), centers_C(cluster));

    % 过滤该工况且有完整数据的行
    mask = idx == cluster;
    X_cluster = X_g_all(mask, :);
    y_cluster = y_g_all(mask);
    ok_g = all(isfinite(X_cluster), 2) & isfinite(y_cluster);
    X_cluster = X_cluster(ok_g, :);
    y_cluster = y_cluster(ok_g);
    n_g = length(y_cluster);
    fprintf('  有效样本: %d\n', n_g);

    if n_g < 50
        fprintf('  样本过少，跳过\n');
        continue;
    end

    % 80/20 划分
    rng(42 + cluster);
    nTrain_g = round(0.8 * n_g);
    perm_g = randperm(n_g);
    idxTrain_g = perm_g(1:nTrain_g);
    idxTest_g  = perm_g(nTrain_g+1:end);

    X_train_g = X_cluster(idxTrain_g, :);
    y_train_g = y_cluster(idxTrain_g);
    X_test_g  = X_cluster(idxTest_g, :);
    y_test_g  = y_cluster(idxTest_g);
    fprintf('  训练: %d, 测试: %d\n', nTrain_g, n_g - nTrain_g);

    % 训练
    fprintf('  训练中 (nTrees=%d)...\n', nTrees);
    g_model = TreeBagger(nTrees, X_train_g, y_train_g, ...
        'Method', 'regression', ...
        'OOBPrediction', 'on', ...
        'OOBPredictorImportance', 'on', ...
        'MinLeafSize', 5);
    g_models{cluster} = g_model;

    % 预测与评估
    y_pred_g = predict(g_model, X_test_g);
    R2_g = 1 - sum((y_test_g - y_pred_g).^2) / sum((y_test_g - mean(y_test_g)).^2);
    RMSE_g = sqrt(mean((y_test_g - y_pred_g).^2));
    MAE_g = mean(abs(y_test_g - y_pred_g));
    res_g = y_test_g - y_pred_g;
    fprintf('  R² = %.4f, RMSE = %.4f kW, MAE = %.4f kW\n', R2_g, RMSE_g, MAE_g);

    % gₖ 检验图 1：预测 vs 实测
    fig = figure('Name', sprintf('g%d 模型：P 预测 vs 实测', cluster), ...
        'Position', [50, 50, 600, 550]);
    scatter(y_test_g, y_pred_g, 8, [0.8 0.3 0.2], 'filled', 'MarkerFaceAlpha', 0.3);
    hold on;
    lims_g = [min([y_test_g; y_pred_g]), max([y_test_g; y_pred_g])];
    plot(lims_g, lims_g, 'k--', 'LineWidth', 1);
    hold off;
    xlabel('实测 $P$ (kW)'); ylabel('预测 $P$ (kW)');
    text(0.05, 0.95, sprintf('R^2 = %.4f\nRMSE = %.4f\nMAE = %.4f', R2_g, RMSE_g, MAE_g), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 10);
    grid on; axis equal tight;
    saveCurrent(fig, sprintf('g%d_pred_vs_actual', cluster), outDir);

    % gₖ 检验图 2：变量重要性
    fig = figure('Name', sprintf('g%d 模型：变量重要性', cluster), ...
        'Position', [150, 150, 500, 380]);
    imp_g = g_model.OOBPermutedVarDeltaError;
    [~, impOrder_g] = sort(imp_g, 'descend');
    bar(imp_g(impOrder_g), 'FaceColor', [0.8 0.3 0.2]);
    set(gca, 'XTickLabel', varNames_g(impOrder_g));
    xtickangle(30);
    ylabel('OOB 变量重要性 (\Delta MSE)');
    grid on;
    saveCurrent(fig, sprintf('g%d_importance', cluster), outDir);

    % gₖ 检验图 3：OOB error 曲线 + 残差直方图
    fig = figure('Name', sprintf('g%d 模型：OOB Error 与残差分布', cluster), ...
        'Position', [100, 100, 900, 400]);

    subplot(1,2,1);
    oobErr_g = oobError(g_model);
    plot(1:nTrees, oobErr_g, 'r-', 'LineWidth', 1.2);
    xlabel('树棵数'); ylabel('OOB MSE');
    grid on;

    subplot(1,2,2);
    histogram(res_g, 30, 'FaceColor', [0.8 0.3 0.2], 'EdgeColor', 'none');
    hold on; xline(0, 'k--', 'LineWidth', 1); hold off;
    xlabel('残差 $P$ (kW)'); ylabel('频数');
    mu_res_g = mean(res_g); sigma_res_g = std(res_g);
    text(0.05, 0.95, sprintf('\\mu = %.4f\n\\sigma = %.4f', mu_res_g, sigma_res_g), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 9);
    grid on;

    saveCurrent(fig, sprintf('g%d_oob_residuals', cluster), outDir);

    fprintf('  g%d 模型完成。\n', cluster);
end

%% ============================================================
%%  保存模型
%% ============================================================
fprintf('\n========== 保存模型 ==========\n');
save('rf_models.mat', 'f_model', 'g_models', ...
    'varNames_f', 'varNames_g', 'nTrees', ...
    'centers_H', 'centers_C', 'kOpt');
fprintf('已保存 rf_models.mat\n');

%% R² 汇总
fprintf('\n========== R² 汇总 ==========\n');
fprintf('f (η 全局):  R² = %.4f, RMSE = %.4f\n', R2_f, RMSE_f);
for cluster = 1:kOpt
    if isempty(g_models{cluster}), continue; end
    g_model = g_models{cluster};
    mask = idx == cluster;
    X_g_k = X_g_all(mask, :); y_g_k = y_g_all(mask);
    ok = all(isfinite(X_g_k), 2) & isfinite(y_g_k);
    y_all = y_g_k(ok);
    y_pred_all = predict(g_model, X_g_k(ok, :));
    res_all = y_all - y_pred_all;
    R2_all = 1 - sum(res_all.^2) / sum((y_all - mean(y_all)).^2);
    RMSE_all = sqrt(mean(res_all.^2));
    fprintf('g%d (P 工况%d):  R² = %.4f, RMSE = %.4f kW\n', ...
        cluster, cluster, R2_all, RMSE_all);
end

diary off;
fprintf('\n完成。\n');
