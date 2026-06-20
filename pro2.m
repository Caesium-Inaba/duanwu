%% 工况点聚类分析
clear;clc;close all;

outDir = fullfile('img', 'pro2');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

cacheFile = 'dkmeans.mat';
figNames = {'working_point_raw', 'elbow', 'silhouette', 'working_point'};

%% 加载数据并确定缓存策略
% 优先：fig 已有 → 直接显示
% 次之：dkmeans.mat 有 → load 数据后作图并存 fig
% 末：从头计算

allFigsExist = all(cellfun(@(n) isfile(fullfile(outDir, [n '.fig'])), figNames));

if allFigsExist
    % 情况1：fig 已存在，直接打开
    openCachedFigs();
    fprintf('从 img\\pro2\\ 加载已有图窗\n');
    load(cacheFile, 'kOpt', 'centers_H', 'centers_C');

elseif isfile(cacheFile)
    % 情况2：dkmeans.mat 有但 fig 不完整，load 数据后作图并存
    load(cacheFile, 'H', 'C', 'kOpt', 'idx', 'centers_H', 'centers_C', 'wcss', 'silAvg');
    plotRawScatter(H, C);
    plotElbow(wcss);
    plotSilhouette(silAvg, kOpt);
    plotClustered(H, C, idx, centers_H, centers_C, kOpt);
    fprintf('从 dkmeans.mat 加载数据，重新出图\n');

else
    % 情况3：从头计算
    load('dc.mat', 'data_clean');
    H = data_clean.Temp_C;
    C = data_clean.C_in_gNm3;

    Hz = (H - mean(H)) / std(H);
    Cz = (C - mean(C)) / std(C);
    Xz = [Hz, Cz];

    kMax = 10;
    wcss = zeros(kMax, 1);
    for k = 1:kMax
        [~, ~, sumd] = kmeans(Xz, k, 'Replicates', 5);
        wcss(k) = sum(sumd);
    end

    silAvg = zeros(kMax-1, 1);
    for k = 2:kMax
        idx_tmp = kmeans(Xz, k, 'Replicates', 5);
        s = silhouette(Xz, idx_tmp);
        silAvg(k-1) = mean(s);
    end
    [~, kSil] = max(silAvg);
    kOpt = kSil + 1;

    [idx, centers_z] = kmeans(Xz, kOpt, 'Replicates', 10);
    centers_H = centers_z(:,1) * std(H) + mean(H);
    centers_C = centers_z(:,2) * std(C) + mean(C);

    plotRawScatter(H, C);
    plotElbow(wcss);
    plotSilhouette(silAvg, kOpt);
    plotClustered(H, C, idx, centers_H, centers_C, kOpt);
    save(cacheFile, 'H', 'C', 'kOpt', 'idx', 'centers_H', 'centers_C', 'wcss', 'silAvg');
    fprintf('完成计算，已保存 dkmeans.mat\n');
end

%% 终端输出
fprintf('\n肘部法 WCSS:\n');
for k = 1:length(wcss)
    fprintf('  k=%d: %.2f', k, wcss(k));
    if k > 1
        fprintf('  (Δ=-%.1f%%)', 100*(wcss(k-1)-wcss(k))/wcss(k-1));
    end
    fprintf('\n');
end
fprintf('\n轮廓系数:\n');
for k = 2:length(silAvg)+1
    mark = '';
    if k == kOpt, mark = ' <-- 最优'; end
    fprintf('  k=%d: %.3f%s\n', k, silAvg(k-1), mark);
end
fprintf('\n选用 k=%d\n', kOpt);
fprintf('聚类中心（原始坐标）:\n');
for j = 1:kOpt
    fprintf('  工况 %d: H=%.1f℃, C=%.2f g/Nm³\n', j, centers_H(j), centers_C(j));
end

%% ── 局部函数 ──

function plotRawScatter(H, C)
    figure('Name', '工况点图（原始）');
    scatter(H, C, 4, 'filled');
    xlabel('温度 (℃)'); ylabel('入口粉尘浓度 (g/Nm³)');
    grid on;
    saveCurrent('working_point_raw');
end

function plotElbow(wcss)
    figure('Name', '肘部法');
    plot(1:length(wcss), wcss, 'o-', 'LineWidth', 1.5);
    xlabel('聚类数 k'); ylabel('WCSS');
    grid on;
    saveCurrent('elbow');
end

function plotSilhouette(silAvg, kOpt)
    figure('Name', '轮廓系数');
    plot(2:length(silAvg)+1, silAvg, 'o-', 'LineWidth', 1.5);
    hold on;
    plot(kOpt, silAvg(kOpt-1), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    xlabel('聚类数 k'); ylabel('平均轮廓系数');
    grid on;
    saveCurrent('silhouette');
end

function plotClustered(H, C, idx, centers_H, centers_C, kOpt)
    figure('Name', '工况点图');
    hold on;
    colors = lines(kOpt);
    for j = 1:kOpt
        mask = idx == j;
        scatter(H(mask), C(mask), 4, colors(j,:), 'filled', ...
            'DisplayName', sprintf('工况 %d', j));
    end
    scatter(centers_H, centers_C, 120, 'k', 'd', 'filled', ...
        'DisplayName', '聚类中心');
    xlabel('温度 (℃)'); ylabel('入口粉尘浓度 (g/Nm³)');
    grid on;
    legend('Location', 'best');
    saveCurrent('working_point');
end

function saveCurrent(name)
    outDir = fullfile('img', 'pro2');
    fig = gcf;
    saveas(fig, fullfile(outDir, [name '.svg']));
    saveas(fig, fullfile(outDir, [name '.png']));
    savefig(fig, fullfile(outDir, [name '.fig']));
end

function openCachedFigs()
    outDir = fullfile('img', 'pro2');
    openfig(fullfile(outDir, 'working_point_raw.fig'), 'visible');
    openfig(fullfile(outDir, 'elbow.fig'), 'visible');
    openfig(fullfile(outDir, 'silhouette.fig'), 'visible');
    openfig(fullfile(outDir, 'working_point.fig'), 'visible');
end
