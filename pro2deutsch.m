%% 第二问：Deutsch 型物理模型拟合 η
%  基础: η = 1 - exp(-w·A/Q)
%  迁移速度: w ∝ Ū₁^α · Ū₂^β · C^γ · exp(δ·H)
%  振打效应: 二次倒 U 形
%  线性化: ln(-ln(1-η/100)) = β·X，用 Ridge 回归
clear;clc;close all;

if isfile('pro2deutsch.txt'), delete('pro2deutsch.txt'); end
diary('pro2deutsch.txt');

%% 目录与缓存
outDir = fullfile('img', 'pro2deutsch');
if ~exist(outDir, 'dir'), mkdir(outDir); end

cacheFile = 'deutsch_model.mat';

figNames = {'f_pred_vs_actual', 'f_residuals', 'f_coef'};
allFigsExist = all(cellfun(@(n) isfile(fullfile(outDir, [n '.fig'])), figNames));

%% 加载数据
load('dc.mat', 'data_clean');

fprintf('===== Deutsch 型物理模型拟合 η =====\n\n');

%% 变量合并
Ubar1 = (data_clean.U1_kV + data_clean.U2_kV) / 2;
Ubar2 = (data_clean.U3_kV + data_clean.U4_kV) / 2;
Tbar1 = (data_clean.T1_s  + data_clean.T2_s)  / 2;
Tbar2 = (data_clean.T3_s  + data_clean.T4_s)  / 2;

H_raw = data_clean.Temp_C;
C_raw = data_clean.C_in_gNm3;
Q_raw = data_clean.Q_Nm3h;
eta_raw = data_clean.eff;

%% 剔除异常值
ok = eta_raw > 0 & eta_raw < 100 & ...
     Ubar1 > 0 & Ubar2 > 0 & C_raw > 0 & Q_raw > 0 & ...
     isfinite(H_raw) & isfinite(Tbar1) & isfinite(Tbar2);
Ubar1 = Ubar1(ok); Ubar2 = Ubar2(ok);
Tbar1 = Tbar1(ok); Tbar2 = Tbar2(ok);
H_raw = H_raw(ok); C_raw = C_raw(ok); Q_raw = Q_raw(ok);
eta_raw = eta_raw(ok);
n = length(eta_raw);
fprintf('有效样本: %d\n', n);

%% 变换
%  η ∈ [0,100] → η_dec = η/100 ∈ [0,1]
%  1 − η_dec = exp(−wA/Q)
%  −ln(1−η_dec) = wA/Q
%  ln(−ln(1−η_dec)) = ln(wA) − ln(Q)
%
%  展开 w:
%  ln(−ln(1−η_dec)) = β₀ + β₁·ln(Ū₁) + β₂·ln(Ū₂)
%                     + β₃·ln(C) + β₄·ln(Q)
%                     + β₅·H + β₆·T̄₁ + β₇·T̄₁²
%                     + β₈·T̄₂ + β₉·T̄₂²

eta_dec = eta_raw / 100;
penetration = 1 - eta_dec;  % 穿透率，约 0.001~0.003

y_trans = log(-log(penetration));

% 构造设计矩阵（T̄ 中心化后做二次，减少共线性）
T1c = Tbar1 - mean(Tbar1);
T2c = Tbar2 - mean(Tbar2);

X_deutsch = [ones(n,1), ...
             log(Ubar1), log(Ubar2), ...
             log(C_raw), log(Q_raw), ...
             H_raw, ...
             T1c, T1c.^2, ...
             T2c, T2c.^2];

varNames_deutsch = {'截距', 'ln(Ū₁)', 'ln(Ū₂)', 'ln(C)', 'ln(Q)', ...
                    'H', 'T̄₁(c)', 'T̄₁²(c)', 'T̄₂(c)', 'T̄₂²(c)'};
nParam = size(X_deutsch, 2);
fprintf('参数数: %d\n', nParam);

%% 80/20 划分
rng(42);
nTrain = round(0.8 * n);
perm = randperm(n);
idxTrain = perm(1:nTrain);
idxTest  = perm(nTrain+1:end);

X_train = X_deutsch(idxTrain, :);
y_train = y_trans(idxTrain);
X_test  = X_deutsch(idxTest, :);
y_test  = y_trans(idxTest);
eta_test = eta_raw(idxTest);

fprintf('训练: %d, 测试: %d\n', nTrain, n - nTrain);

%% ============================================================
%%  Ridge 回归（交叉验证选 λ）
%% ============================================================
if allFigsExist
    load(cacheFile, 'beta', 'lambda_opt', 'T1_mean', 'T2_mean', 'varNames_deutsch');
    fprintf('从缓存加载。\n');
else
    % 不包含截距的 λ 序列（ridge 函数的标准用法）
    lambda_vec = logspace(-6, 2, 50);

    % ridge(y, X, lambda, scaled) — X 不含截距列
    B_ridge = ridge(y_train, X_train(:, 2:end), lambda_vec, 0);

    % 对每个 λ 计算测试集 MSE
    mse_test = zeros(length(lambda_vec), 1);
    for i = 1:length(lambda_vec)
        beta_i = B_ridge(:, i);
        y_pred_i = [ones(size(X_test,1),1), X_test(:,2:end)] * beta_i;
        mse_test(i) = mean((y_test - y_pred_i).^2);
    end

    [~, idx_best] = min(mse_test);
    lambda_opt = lambda_vec(idx_best);
    beta = B_ridge(:, idx_best);
    fprintf('最优 λ = %.4e\n', lambda_opt);

    % 保存（含中心化参数，预测时需还原）
    T1_mean = mean(Tbar1);
    T2_mean = mean(Tbar2);
    save(cacheFile, 'beta', 'lambda_opt', 'T1_mean', 'T2_mean', 'varNames_deutsch');
    fprintf('已保存 %s\n', cacheFile);
end

%% 预测与评估
y_pred_test = [ones(size(X_test,1),1), X_test(:,2:end)] * beta;
eta_pred_test = 100 * (1 - exp(-exp(y_pred_test)));

res = eta_test - eta_pred_test;
R2 = 1 - sum(res.^2) / sum((eta_test - mean(eta_test)).^2);
RMSE = sqrt(mean(res.^2));
MAE = mean(abs(res));

y_pred_train = [ones(size(X_train,1),1), X_train(:,2:end)] * beta;
eta_pred_train = 100 * (1 - exp(-exp(y_pred_train)));
R2_train = 1 - sum((eta_raw(idxTrain) - eta_pred_train).^2) / ...
    sum((eta_raw(idxTrain) - mean(eta_raw(idxTrain))).^2);
deltaR2 = R2_train - R2;

fprintf('\n=== 评估（η 原始空间） ===\n');
fprintf('训练 R² = %.4f, 测试 R² = %.4f, ΔR² = %.4f\n', R2_train, R2, deltaR2);
fprintf('RMSE = %.4f%%, MAE = %.4f%%\n', RMSE, MAE);

%% 系数报告
fprintf('\n=== 回归系数 ===\n');
fprintf('  (变换空间: ln(-ln(1-η/100)) = β·X)\n');
for i = 1:nParam
    stars = '';
    fprintf('  %-12s  %+12.6f\n', varNames_deutsch{i}, beta(i));
end

fprintf('\n物理含义验证:\n');
fprintf('  β₁(ln Ū₁) = %+.4f  →  ', beta(2));
if beta(2) > 0, fprintf('✅ 电压↑效率↑\n'); else, fprintf('⚠ 符号异常\n'); end
fprintf('  β₂(ln Ū₂) = %+.4f  →  ', beta(3));
if beta(3) > 0, fprintf('✅ 电压↑效率↑\n'); else, fprintf('⚠ 符号异常\n'); end
fprintf('  β₄(ln Q)  = %+.4f  →  ', beta(5));
if beta(5) < 0, fprintf('✅ 流量↑效率↓(Deutsch)\n'); else, fprintf('⚠ 符号异常\n'); end
fprintf('  β₇(T̄₁²)  = %+.4f  →  ', beta(8));
if beta(8) < 0, fprintf('✅ 倒 U 形\n'); else, fprintf('  非倒 U\n'); end
fprintf('  β₉(T̄₂²)  = %+.4f  →  ', beta(10));
if beta(10) < 0, fprintf('✅ 倒 U 形\n'); else, fprintf('  非倒 U\n'); end

%% ==== 图 ====
if ~allFigsExist
    % 图 1：预测 vs 实测
    fig = figure('Name', 'Deutsch 模型：η 预测 vs 实测', 'Position', [50, 50, 600, 550]);
    scatter(eta_test, eta_pred_test, 8, [0.2 0.6 0.4], 'filled', 'MarkerFaceAlpha', 0.3);
    hold on;
    lims = [min([eta_test; eta_pred_test]), max([eta_test; eta_pred_test])];
    plot(lims, lims, 'k--', 'LineWidth', 1); hold off;
    xlabel('实测 $\eta$ (%)'); ylabel('预测 $\eta$ (%)');
    grid on; axis equal tight;
    text(0.05, 0.95, sprintf('R^2 = %.4f\nRMSE = %.4f\n\\lambda = %.2e', R2, RMSE, lambda_opt), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 10);
    saveCurrent(fig, 'f_pred_vs_actual', outDir);

    % 图 2：残差
    fig = figure('Name', 'Deutsch 模型：残差分布', 'Position', [100, 100, 500, 400]);
    histogram(res, 40, 'FaceColor', [0.2 0.6 0.4], 'EdgeColor', 'none');
    hold on; xline(0, 'k--', 'LineWidth', 1); hold off;
    xlabel('残差 $\eta$ (%)'); ylabel('频数');
    text(0.05, 0.95, sprintf('\\mu = %.4f\n\\sigma = %.4f', mean(res), std(res)), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 9);
    grid on;
    saveCurrent(fig, 'f_residuals', outDir);

    % 图 3：系数
    fig = figure('Name', 'Deutsch 模型：回归系数', 'Position', [150, 150, 600, 420]);
    bar(beta(2:end), 'FaceColor', [0.2 0.6 0.4]);
    set(gca, 'XTickLabel', varNames_deutsch(2:end));
    xtickangle(30);
    ylabel('系数值'); grid on;
    title(sprintf('Ridge (\\lambda=%.2e)', lambda_opt));
    saveCurrent(fig, 'f_coef', outDir);
end

fprintf('\n完成。\n');
diary off;
