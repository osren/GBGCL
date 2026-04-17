# GBGCL — Granular Ball Graph Contrastive Learning

> 基于粒球理论的图对比学习框架，用于图表示学习研究。

## 项目结构

```
GBGCL/
├── src/                    # 核心代码
│   ├── data.py             # 数据加载
│   ├── models.py           # 模型定义
│   ├── train.py            # 训练主程序
│   ├── granular.py         # 粒球构建模块
│   └── gb_utils.py         # 粒球工具函数
│
├── topo/                   # 拓扑感知模块
│
├── tools/                  # 辅助工具脚本
│   ├── analyze_results.py  # 结果分析
│   ├── sweep.py            # 超参搜索
│   └── visualize_granules.py
│
├── scripts/                # 实验运行脚本
│   ├── experiments_status.py
│   └── generate_report.py
│
├── figures/                # 框架图
│   ├── final/              # 最终版（v8）PNG & PDF
│   ├── archive/            # 历史迭代版本
│   └── pptx/               # 可编辑 PPTX
│
├── docs/                   # 文档
│   ├── weekly_reports/     # 周报（docx/pdf）
│   └── defense/            # 答辩材料、修改计划
│
├── slides/                 # 工作进展 PPT
│
├── analysis/               # 实验结果分析
├── results/                # 实验输出
├── logs/                   # 运行日志
├── backup/                 # 旧版代码备份
├── sgrl_images/            # 参考文献图示
│
├── env.yaml                # Conda 环境配置
├── environment.yml         # 备用环境配置
└── README.md
```

## 环境配置

```bash
conda env create -f env.yaml
conda activate gbgcl
```

## 快速开始

```bash
# 训练
python src/train.py --dataset CS --epochs 200

# 实验状态查看
python scripts/experiments_status.py
```

## 数据集

支持 Coauthor-CS、Coauthor-Physics、Amazon-Photo、Amazon-Computers 四个基准数据集。

## 框架图

最新框架图见 `figures/final/GBGCL_Framework_v8.png`：

- **输入层**：原始图数据集（CS/Physics/Photo/Computers）
- **Granular Ball Module**：Build Granules → Ball Graph → Ball Diffusion
- **对比学习**：Online / Target 双分支 + EMA
- **损失函数**：L_BYOL + L_Ball + L_Scatter

## 相关论文

- SGRL（参考基线）见 `docs/defense/sgrl.pdf`
