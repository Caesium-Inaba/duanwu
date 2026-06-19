clc; clear; close all;
%% 读取数据并绘制各变量与时间戳的散点图
data = readtable('原题\Cement_ESP_Data.csv');
save("data.mat", "data")

% 解析时间戳
t = datetime(data.timestamp);

% 计算净化量（入口浓度 g/Nm³ - 出口浓度 mg/Nm³ 转 g/Nm³）
dC = data.C_in_gNm3 - data.C_out_mgNm3 / 1000;
data.dC = dC;

% 计算除尘效率（注意单位：出口浓度 mg/Nm³ → g/Nm³ 除以 1000）
eff = (data.C_in_gNm3 - data.C_out_mgNm3 / 1000) ./ data.C_in_gNm3 * 100;
data.eff = eff;

% 需要绘制的变量（排除 timestamp）
varNames = data.Properties.VariableNames(2:end);

% 标签映射
labels = {'Temp_C (℃)', 'C_{in} (g/Nm³)', 'Q (Nm³/h)', ...
          'U_1 (kV)', 'U_2 (kV)', 'U_3 (kV)', 'U_4 (kV)', ...
          'T_1 (s)', 'T_2 (s)', 'T_3 (s)', 'T_4 (s)', ...
          'C_{out} (mg/Nm³)', 'P_{total} (kW)', ...
          '净化量 (g/Nm³)', '除尘效率 (%)'};

% 创建保存目录
if ~exist('img\orig', 'dir')
    mkdir('img\orig');
end

% 每个变量单独画一张图
for i = 1:numel(varNames)
    fig = figure('Name', labels{i});
    scatter(t, data.(varNames{i}), 2, 'filled');
    xlabel('时间');
    ylabel(labels{i});
    grid on;
    xtickformat('MM-dd HH:mm');

    % 保存 svg、png、fig
    saveas(fig, fullfile('img\orig', [varNames{i} '.svg']));
    saveas(fig, fullfile('img\orig', [varNames{i} '.png']));
    savefig(fig, fullfile('img\orig', [varNames{i} '.fig']));
end
