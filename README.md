## 文件介绍

这里是某大学生的数学建模练习，自用。

重度 AI 使用，qwq

## 支撑材料文件结构

```
.
├── draw.m                           # 读取 CSV 并作各变量时序图
├── clean.m                          # 数据清洗
├── dc.csv                           # 清洗后的数据表
├── pro1_quali.m                     # 正态性检验与 Spearman 相关矩阵
├── pro1_quali_spearman.csv          # Spearman 秩相关系数矩阵
├── pro1_quali_spearman_p.csv        # Spearman p 值矩阵
├── pro1.m                           # 各变量时序变化规律拟合
├── writeCorrCSV.m                   # m函数文件：相关系数矩阵写入 CSV 表
├── saveCurrent.m                    # m函数文件：保存图窗为 svg + png + fig
├── significanceStars.m              # m函数文件：p 值 → 显著性星号
├── pro2_cluster.m                   # K-means 聚类
├── dkmeans.mat                      # K-means 聚类结果
├── pro2.m                           # 四个工况各自的 Spearman 相关矩阵
├── pro2_spearman_cluster{k}_rho.csv   # 工况 k 的 Spearman ρ (k=1..4)
├── pro2_spearman_cluster{k}_pval.csv  # 工况 k 的 p 值 (k=1..4)
├── pro2RF.m                         # 构建随机森林模型
├── pro2poly.m                       # 构建二次多项式回归模型
├── rf_models.mat                    # pro2RF.m 输出：f_model (TreeBagger), g_models (4×1 cell)
├── poly_models.mat                  # pro2poly.m 输出：g_mdls (4×1 cell, LinearModel)
├── pro2opt.m                        # 单目标网格搜索
├── dream.m                          # 证明现有装备不可能达标
├── pro2opt_lease.m                  # 双目标 Pareto
└── opt_lease.mat                    # 双目标 Pareto 的结果
```
