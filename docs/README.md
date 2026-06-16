# GBGCL 文档中心

**Granular Ball Graph Contrastive Learning** — 在 SGRL (NeurIPS 2024) 上引入粒球扩散的图对比学习研究。

## 从这里开始

| 文档 | 说明 |
|------|------|
| **[ROADMAP.md](ROADMAP.md)** | 愿景、障碍、G-SGRL 路线、里程碑 |
| **[BASELINES.md](BASELINES.md)** | SGRL 论文基线 vs 本项目结果（可引用） |
| **[EXPERIMENTS.md](EXPERIMENTS.md)** | 已做实验与失败假设 |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | 代码架构与数据流 |
| **[COMMANDS.md](COMMANDS.md)** | 训练与扫参命令 |

## 其他

| 路径 | 说明 |
|------|------|
| [服务器实验指南.md](服务器实验指南.md) | 远程实验 |
| [knowledge/](knowledge/) | API / 模块参考（开发用） |
| [weekly_reports/](weekly_reports/) | 周报存档 |
| [defense/](defense/) | 答辩材料 |
| [archive/](archive/) | **历史文档，勿作现状依据** |

## 代码入口

- 训练：`src/train.py`（在 `src/` 目录运行）
- 扫参：`tools/sweepX.py`
- 结果：`results/<dataset>_summary.csv`

## 论文与 Skill

- SGRL 论文本地解析：`output/sgrl_d15c3ab0/sgrl/hybrid_auto/sgrl.md`
- AI 专家背景：`.cursor/skills/gnn-gbgcl-expert/`

## 引用 SGRL

```bibtex
@inproceedings{he2024sgrl,
  title={Exploitation of a Latent Mechanism in Graph Contrastive Learning: Representation Scattering},
  author={He, Dongxiao and others},
  booktitle={NeurIPS},
  year={2024}
}
```
