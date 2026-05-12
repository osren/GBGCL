# BYOL Baseline 实验命令

## 当前最佳配置：Photo 数据集

```bash
cd /Users/didi/Desktop/GBGCL

# 700 epochs, 5 trials - 标准 BYOL baseline
nohup python src/train.py --dataset_name Photo --num_epochs 700 --trials 5 --device cuda > logs/byol_700_5.log 2>&1 &
```

## 查看结果

```bash
# 查看日志
tail -f logs/byol_700_5.log

# 提取 ACC
grep "TRIAL" logs/byol_700_5.log | awk '{print $5}'
```

## 其他数据集

```bash
# CS
nohup python src/train.py --dataset_name CS --num_epochs 700 --trials 5 --device cuda > logs/byol_cs.log 2>&1 &

# Computers
nohup python src/train.py --dataset_name Computers --num_epochs 700 --trials 5 --device cuda > logs/byol_computers.log 2>&1 &

# Physics
nohup python src/train.py --dataset_name Physics --num_epochs 700 --trials 5 --device cuda > logs/byol_physics.log 2>&1 &
```

## 汇总脚本

```bash
# 运行汇总
bash scripts/summary.sh
```