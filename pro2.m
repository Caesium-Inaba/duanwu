%% 工况点聚类分析
clear;clc;close all;

outDir = fullfile('img', 'pro2');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

cacheFile = 'dkmeans.mat';

if isfile(cacheFile)
    %% 已有缓存：直接出图
    load(cacheFile, 'H', 'C', 'kOpt', 'idx', 'centers_H', 'centers_C', 'wcss', 'silAvg');

    figRaw = openfig(fullfile(outDir, 'working_point_raw.fig'), 'visible');
    figElbow = openfig(fullfile(outDir, 'elbow.fig'), 'visible');
    figSil = openfig(fullfile(outDir, 'silhouette.fig'), 'visible');
    figClust = openfig(fullfile(outDir, 'working_point.fig'), 'visible');

    fprintf('已从 dkmeans.mat 加载缓存，k=%d\n', kOpt);
else
    %% 首次运行：完整计算并缓存
    load('dc.mat', 'data_clean');
    H = data_clean.Temp_C;
    C = data_clean.C_in_gNm3;

    % 原始散点图（无聚类）
    figRaw = figure('Name', '工况点图（原始）');
    scatter(H, C, 4, 'filled');
    xlabel('温度 (℃)');
    ylabel('入口粉尘浓度 (g/Nm³)');
    grid on;
    saveas(figRaw, fullfile(outDir, 'working_point_raw.svg'));
    saveas(figRaw, fullfile(outDir, 'working_point_raw.png'));
    savefig(figRaw, fullfile(outDir, 'working_point_raw.fig'));

    % z-score 标准化
    Hz = (H - mean(H)) / std(H);
    Cz = (C - mean(C)) / std(C);
    Xz = [Hz, Cz];

    % 肘部法
    kMax = 10;
    wcss = zeros(kMax, 1);
    for k = 1:kMax
        [~, ~, sumd] = kmeans(Xz, k, 'Replicates', 5);
        wcss(k) = sum(sumd);
    end

    figElbow = figure('Name', '肘部法');
    plot(1:kMax, wcss, 'o-', 'LineWidth', 1.5);
    xlabel('聚类数 k'); ylabel('WCSS');
    grid on;
    saveas(figElbow, fullfile(outDir, 'elbow.svg'));
    saveas(figElbow, fullfile(outDir, 'elbow.png'));
    savefig(figElbow, fullfile(outDir, 'elbow.fig'));

    % 轮廓系数法确定 k
    silAvg = zeros(kMax-1, 1);
    for k = 2:kMax
        idx_tmp = kmeans(Xz, k, 'Replicates', 5);
        s = silhouette(Xz, idx_tmp);
        silAvg(k-1) = mean(s);
    end
    [~, kSil] = max(silAvg);
    kOpt = kSil + 1;

    figSil = figure('Name', '轮廓系数');
    plot(2:kMax, silAvg, 'o-', 'LineWidth', 1.5);
    hold on;
    plot(kOpt, silAvg(kSil), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    xlabel('聚类数 k'); ylabel('平均轮廓系数');
    grid on;
    saveas(figSil, fullfile(outDir, 'silhouette.svg'));
    saveas(figSil, fullfile(outDir, 'silhouette.png'));
    savefig(figSil, fullfile(outDir, 'silhouette.fig'));

    % k-means 聚类
    [idx, centers_z] = kmeans(Xz, kOpt, 'Replicates', 10);
    centers_H = centers_z(:,1) * std(H) + mean(H);
    centers_C = centers_z(:,2) * std(C) + mean(C);

    % 聚类着色散点图
    figClust = figure('Name', '工况点图');
    hold on;
    colors = lines(kOpt);
    for j = 1:kOpt
        mask = idx == j;
        scatter(H(mask), C(mask), 4, colors(j,:), 'filled', ...
            'DisplayName', sprintf('工况 %d', j));
    end
    scatter(centers_H, centers_C, 120, 'k', 'd', 'filled', ...
        'DisplayName', '聚类中心');
    xlabel('温度 (℃)');
    ylabel('入口粉尘浓度 (g/Nm³)');
    grid on;
    legend('Location', 'best');
    saveas(figClust, fullfile(outDir, 'working_point.svg'));
    saveas(figClust, fullfile(outDir, 'working_point.png'));
    savefig(figClust, fullfile(outDir, 'working_point.fig'));

    % 缓存
    save(cacheFile, 'H', 'C', 'kOpt', 'idx', 'centers_H', 'centers_C', 'wcss', 'silAvg');
    fprintf('已保存 dkmeans.mat，k=%d\n', kOpt);
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
fprintf('\n聚类中心（原始坐标）:\n');
for j = 1:kOpt
    fprintf('  工况 %d: H=%.1f℃, C=%.2f g/Nm³\n', j, centers_H(j), centers_C(j));
end
