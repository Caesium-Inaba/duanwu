%% 自动检测并清除各变量对时间的局部异常（如 May 6 高温类噪声）
clc; clear; close all;

data = readtable('原题\Cement_ESP_Data.csv');
t = datetime(data.timestamp);
t_hours = hours(t - t(1));  % 以小时为单位的相对时间
n = height(data);

%% 傅里叶级数拟合（周为基频，谐波覆盖至 ~6h 周期）
T = 7 * 24;        % 一周 = 168 h
nHarm = 28;        % 最高谐波：168/28 = 6 h 周期
X = ones(n, 1);
for k = 1:nHarm
    X = [X, sin(2*pi*k*t_hours/T), cos(2*pi*k*t_hours/T)]; %#ok<AGROW>
end

% 逐变量拟合，计算标准化残差
varNames = data.Properties.VariableNames(2:end);
nVar = numel(varNames);
z_all = zeros(n, nVar);       % 点级 z-score
win = 60;                     % 60 分钟滑动窗

fprintf('=== 各变量异常检测报告 ===\n');
for i = 1:nVar
    y = data.(varNames{i});
    ok = isfinite(y);
    beta = X(ok,:) \ y(ok);
    y_fit = X * beta;
    res = y - y_fit;
    sigma = std(res(ok));
    if sigma > 0
        z = res / sigma;
    else
        z = zeros(size(res));
    end
    z_all(:,i) = z;

    % 滑动窗内平均 |z|，标记连续异常
    mov_abs_z = movmean(abs(z), win);
    bad_i = mov_abs_z > 2.5;
    bad_i = imdilate(bad_i, ones(win, 1));  % 膨胀确保整段覆盖
    nBad = sum(bad_i);
    if nBad > 0
        fprintf('  %-18s  异常点: %6d / %d  (%.1f%%)\n', varNames{i}, nBad, n, 100*nBad/n);
    end
    data.([varNames{i} '_bad']) = bad_i;
end

%% 合并：任一点被任一变量标记即剔除
bad_any = any(table2array(data(:, contains(data.Properties.VariableNames, '_bad'))), 2);
data_clean = data(~bad_any, :);

fprintf('\n合并后剔除: %d / %d  (%.1f%%)\n', sum(bad_any), n, 100*sum(bad_any)/n);

% 补上派生变量
data_clean.dC  = data_clean.C_in_gNm3 - data_clean.C_out_mgNm3 / 1000;
data_clean.eff = (data_clean.C_in_gNm3 - data_clean.C_out_mgNm3 / 1000) ...
                 ./ data_clean.C_in_gNm3 * 100;

% 删掉临时标记列
data_clean = removevars(data_clean, contains(data_clean.Properties.VariableNames, '_bad'));

save('dc.mat', 'data_clean');
writetable(data_clean, 'dc.csv');
fprintf('已保存 dc.mat 和 dc.csv，清洗后 %d 行\n', height(data_clean));

%% ===== 重新绘图，保存到 img\clean\ =====
outDir = fullfile('img', 'clean');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

t_clean = datetime(data_clean.timestamp);
varNames_plot = data_clean.Properties.VariableNames(2:end);
labels = {'Temp_C (℃)', 'C_{in} (g/Nm³)', 'Q (Nm³/h)', ...
          'U_1 (kV)', 'U_2 (kV)', 'U_3 (kV)', 'U_4 (kV)', ...
          'T_1 (s)', 'T_2 (s)', 'T_3 (s)', 'T_4 (s)', ...
          'C_{out} (mg/Nm³)', 'P_{total} (kW)', ...
          '净化量 (g/Nm³)', '除尘效率 (%)'};

% 清洗后各变量独立图
for i = 1:numel(varNames_plot)
    fig = figure('Name', ['清洗后 - ', labels{i}]);
    scatter(t_clean, data_clean.(varNames_plot{i}), 2, 'filled');
    xlabel('时间');
    ylabel(labels{i});
    grid on;
    xtickformat('MM-dd HH:mm');

    saveas(fig, fullfile(outDir, [varNames_plot{i}, '.svg']));
    saveas(fig, fullfile(outDir, [varNames_plot{i}, '.png']));
    savefig(fig, fullfile(outDir, [varNames_plot{i}, '.fig']));
end

fprintf('图片已保存到 %s\n', outDir);

%% ===== 诊断图：标出被删除的点及其删除原因 =====
diagDir = fullfile('img', 'clean');
t_orig = datetime(data.timestamp);

for i = 1:nVar
    bad_i = data.([varNames{i} '_bad']);        % 此变量导致的异常
    bad_other = bad_any & ~bad_i;                % 其他变量导致异常、此变量正常
    kept = ~bad_any;                             % 保留的点

    fig = figure('Name', sprintf('清洗诊断 - %s', varNames{i}));
    hold on;
    h1 = scatter(t_orig(kept), data.(varNames{i})(kept), 2, 'filled', ...
        'MarkerFaceColor', [0.7 0.7 0.7], 'DisplayName', '保留');
    h2 = scatter(t_orig(bad_i), data.(varNames{i})(bad_i), 6, 'filled', ...
        'MarkerFaceColor', [0.85 0.2 0.2], 'DisplayName', '此变量异常删除');
    h3 = scatter(t_orig(bad_other), data.(varNames{i})(bad_other), 4, 'filled', ...
        'MarkerFaceColor', [1.0 0.6 0.2], 'DisplayName', '他变量异常连带删除');
    hold off;
    xlabel('时间');
    ylabel(varNames{i});
    legend([h2, h3, h1], 'Location', 'best');
    grid on;
    xtickformat('MM-dd HH:mm');

    saveas(fig, fullfile(diagDir, [varNames{i} '_diag.svg']));
    saveas(fig, fullfile(diagDir, [varNames{i} '_diag.png']));
    savefig(fig, fullfile(diagDir, [varNames{i} '_diag.fig']));
end

fprintf('诊断图已保存到 %s\n', diagDir);
