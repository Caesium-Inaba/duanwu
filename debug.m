%% debug.m — 从 opt_lease.mat 提取精简 Pareto 前沿
%  去掉 *_all 网格候选点，仅保留 Pareto 前沿，存入 pareto_frontier.mat
clear;clc;

load('opt_lease.mat', 'results');

results_frontier = cell(length(results), 1);
dropFields = {'U1_all', 'U2_all', 'T1_all', 'T2_all', 'P_all', 'Cout_all'};
for k = 1:length(results)
    r = results{k};
    if isempty(r), continue; end
    r = rmfield(r, intersect(fieldnames(r), dropFields));
    results_frontier{k} = r;
end

save('pareto_frontier.mat', 'results_frontier');
fprintf('已保存 pareto_frontier.mat（仅 Pareto 前沿点，不含网格候选）。\n');
