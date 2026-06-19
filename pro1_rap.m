%% 读取数据并计算除尘效率
clear;clc;close all;

load('data.mat', 'data');
t = datetime(data.timestamp);

eff = (data.C_in_gNm3 - data.C_out_mgNm3 / 1000) ./ data.C_in_gNm3 * 100;

%% 差分
d_eff = diff(eff);
t_diff = t(1:end-1);

%% 绘图
outDir = fullfile('img', 'pro1_rap');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

fig = figure('Name', 'eff 差分');
plot(t_diff, d_eff, '-', 'LineWidth', 0.5);
xlabel('时间');
ylabel('\Delta 除尘效率 (%)');
grid on;
xtickformat('MM-dd HH:mm');

saveas(fig, fullfile(outDir, 'd_eff.svg'));
saveas(fig, fullfile(outDir, 'd_eff.png'));
savefig(fig, fullfile(outDir, 'd_eff.fig'));
