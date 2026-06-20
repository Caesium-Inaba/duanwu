%% 对各变量（除 C_out）作单正弦函数拟合，周期由 FFT 自动确定
clear;clc;close all;
if isfile('pro1.txt'), delete('pro1.txt'); end
diary('pro1.txt');

load('dc.mat', 'data_clean');
t = datetime(data_clean.timestamp);
t_hours = hours(t - t(1));
n = length(t_hours);
dt = 1/60;  % 采样间隔 = 1 min = 1/60 h

varNames = setdiff(data_clean.Properties.VariableNames(2:end), {'C_out_mgNm3'});

labelMap = containers.Map(...
    {'Temp_C','C_in_gNm3','Q_Nm3h', ...
     'U1_kV','U2_kV','U3_kV','U4_kV', ...
     'T1_s','T2_s','T3_s','T4_s', ...
     'P_total_kW','dC','eff'}, ...
    {'Temp_C (℃)','C_{in} (g/Nm³)','Q (Nm³/h)', ...
     'U_1 (kV)','U_2 (kV)','U_3 (kV)','U_4 (kV)', ...
     'T_1 (s)','T_2 (s)','T_3 (s)','T_4 (s)', ...
     'P_{total} (kW)','净化量 (g/Nm³)','除尘效率 (%)'});

outDir = fullfile('img', 'pro1');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

fprintf('==== 单正弦拟合报告（周期由 FFT 自动检测）====\n\n');

for i = 1:numel(varNames)
    vn = varNames{i};
    y = data_clean.(vn);
    lbl = labelMap(vn);

    ok = isfinite(y);
    y_ok = y(ok);
    yc = y_ok - mean(y_ok);
    n_ok = length(yc);

    % FFT 找主导周期
    Y = abs(fft(yc) / n_ok);
    Y = Y(1:floor(n_ok/2)+1);
    Y(1) = 0;
    f = (0:floor(n_ok/2))' / n_ok * 60;  % cycles/hour
    [~, idxPeak] = max(Y);
    T_detected = 1 / f(idxPeak);

    % 限制周期在合理范围 [4h, 200h]
    T_use = max(4, min(200, T_detected));

    % 单正弦拟合
    omega = 2 * pi / T_use;
    X = [ones(n,1), sin(omega * t_hours), cos(omega * t_hours)];
    beta = X(ok,:) \ y(ok);
    yFit = X * beta;
    res = y - yFit;

    RSS = sum(res(ok).^2);
    TSS = sum((y_ok - mean(y_ok)).^2);
    R2 = 1 - RSS / TSS;
    RMSE = sqrt(RSS / sum(ok));

    a0 = beta(1);
    A = sqrt(beta(2)^2 + beta(3)^2);
    phi = atan2(beta(3), beta(2));

    % CLI
    fprintf('─ %s ─────────────────────────────\n', lbl);
    fprintf('  检测周期: %.1f h   振幅: %.4f   相位: %.4f rad\n', T_use, A, phi);
    fprintf('  a0 = %.4f    R² = %.4f    RMSE = %.4f\n', a0, R2, RMSE);
    fprintf('  y = %.4f + %.4f · sin(2πt/%.1f + %.4f rad)\n\n', a0, A, T_use, phi);

    % 图1：拟合叠加
    fig = figure('Name', sprintf('拟合 - %s', lbl));
    hold on;
    scatter(t, y, 2, [0.7 0.7 0.7], 'filled');
    plot(t, yFit, 'r-', 'LineWidth', 1);
    xlabel('时间'); ylabel(lbl);
    grid on; xtickformat('MM-dd HH:mm');
    legend({'数据', '拟合'}, 'Location', 'best');
    saveas(fig, fullfile(outDir, [vn, '.svg']));
    saveas(fig, fullfile(outDir, [vn, '.png']));
    savefig(fig, fullfile(outDir, [vn, '.fig']));

    % 图2：诊断
    fig2 = figure('Name', sprintf('诊断 - %s', lbl));
    tlo = tiledlayout(3, 1);

    nexttile;
    plot(t, res, '.', 'MarkerSize', 3);
    yline(0, '--');
    ylabel('残差'); grid on;
    xtickformat('MM-dd HH:mm');

    nexttile;
    histogram(res, 40, 'Normalization', 'pdf', 'EdgeAlpha', 0.3);
    hold on;
    res_ok = res(isfinite(res));
    xg = linspace(min(res_ok), max(res_ok), 200);
    smu = mean(res_ok); ssigma = std(res_ok);
    plot(xg, exp(-(xg-smu).^2/(2*ssigma^2))/(ssigma*sqrt(2*pi)), 'r-', 'LineWidth', 1.5);
    xlabel('残差'); ylabel('概率密度'); grid on;

    nexttile;
    maxLag = min(120, length(res_ok)-1);
    acf = zeros(maxLag+1, 1);
    for lag = 0:maxLag
        x1 = res_ok(1:end-lag); x2 = res_ok(1+lag:end);
        acf(lag+1) = mean((x1-mean(x1)).*(x2-mean(x2))) / (std(x1)*std(x2));
    end
    stem(0:maxLag, acf, 'Marker', 'none');
    hold on;
    conf = 1.96 / sqrt(length(res_ok));
    yline( conf, 'r--'); yline(-conf, 'r--'); yline(0, '--');
    xlabel('滞后 (min)'); ylabel('ACF'); grid on;

    tlo.Title.String = sprintf('诊断 - %s  (R²=%.3f, T=%.1fh)', lbl, R2, T_use);
    saveas(fig2, fullfile(outDir, [vn, '_diag.svg']));
    saveas(fig2, fullfile(outDir, [vn, '_diag.png']));
    savefig(fig2, fullfile(outDir, [vn, '_diag.fig']));
end

fprintf('图片已保存到 %s\n', outDir);
diary off;