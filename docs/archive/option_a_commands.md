# Option A 测试命令

## 700 epochs, 5 trials

```bash
cd /Users/didi/Desktop/GBGCL

# Option A: with residual
nohup python src/train.py --dataset_name Photo --use_gb --gb_quity homo --gb_residual_online --gb_residual_weight 0.1 --num_epochs 700 --trials 5 --gb_rebuild_every 10 --device cuda > logs/option_a_700_5.log 2>&1 &

# Baseline: without residual
nohup python src/train.py --dataset_name Photo --use_gb --gb_quity homo --num_epochs 700 --trials 5 --gb_rebuild_every 10 --device cuda > logs/baseline_700_5.log 2>&1 &
```

## 查看结果

```bash
tail -f logs/option_a_700_5.log
tail -f logs/baseline_700_5.log
```

## 或者无粒球 baseline（纯 BYOL）

```bash
nohup python src/train.py --dataset_name Photo --num_epochs 700 --trials 5 --device cuda > logs/byol_baseline.log 2>&1 &
```