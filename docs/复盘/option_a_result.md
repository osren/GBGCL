# Photo 数据集实验结果汇总

**日期：2026年5月**

---

## 实验结果

| 配置 | 均值 | 最大值 | 变化 |
|------|------|-------|------|
| BYOL (无粒球) | 0.9379 | 0.9389 | baseline |
| Baseline (有粒球) | 0.9344 | 0.9353 | -0.35% |
| Option A (残差) | 0.9286 | 0.9301 | -0.93% |

---

## 结论

1. **无粒球的 BYOL 效果最好** - 说明当前粒球模块存在问题
2. **加粒球反而下降** - Baseline 比 BYOL 低 0.35%
3. **Option A 残差让情况更糟** - 比 Baseline 再降 0.58%

---

## 问题分析

当前粒球模块存在以下问题：
1. 粒球只在 epoch 0 构建，不跟随嵌入更新
2. 粒球扩散结果不累积
3. 残差方式（Option A）反而引入噪声
4. 架构设计需要重新审视

---

## 后续建议

1. **放弃当前粒球方案** - 效果不如纯 BYOL
2. **重新设计粒球模块** - 需要更深入地融入训练流程
3. **或直接使用 BYOL baseline** - 当前最强配置

---

## 实验命令参考

```bash
# BYOL baseline
python src/train.py --dataset_name Photo --num_epochs 700 --trials 5 --device cuda

# Baseline (有粒球)
python src/train.py --dataset_name Photo --use_gb --gb_quity homo --num_epochs 700 --trials 5 --device cuda

# Option A (有粒球+残差)
python src/train.py --dataset_name Photo --use_gb --gb_quity homo --gb_residual_online --gb_residual_weight 0.1 --num_epochs 700 --trials 5 --device cuda
```