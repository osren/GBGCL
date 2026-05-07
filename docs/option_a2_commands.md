# Option A2 集成投票方案 - 服务器验证命令

> 在服务器上依次执行以下命令进行验证

---

## 1. 验证语法正确性

```bash
cd /Users/didi/Desktop/GBGCL
python3 -m py_compile src/gb_utils.py
python3 -m py_compile src/train.py
echo "Syntax OK"
```

---

## 2. Ensemble 模式 - 四个数据集

```bash
cd /Users/didi/Desktop/GBGCL

# Photo 数据集
python src/train.py --dataset_name Photo --use_gb --gb_ensemble --gb_ensemble_quities homo,detach,edges --num_epochs 50 --trials 1 --gb_rebuild_every 10 --device cuda

# Computers 数据集
python src/train.py --dataset_name Computers --use_gb --gb_ensemble --gb_ensemble_quities homo,detach,edges --num_epochs 50 --trials 1 --gb_rebuild_every 10 --device cuda

# CS 数据集
python src/train.py --dataset_name CS --use_gb --gb_ensemble --gb_ensemble_quities homo,detach,edges --num_epochs 50 --trials 1 --gb_rebuild_every 10 --device cuda

# Physics 数据集
python src/train.py --dataset_name Physics --use_gb --gb_ensemble --gb_ensemble_quities homo,detach,edges --num_epochs 50 --trials 1 --gb_rebuild_every 10 --device cuda
```

预期输出日志：
```
[Ensemble] quity=homo, weight=0.2xxx
[Ensemble] quity=detach, weight=0.6xxx
[Ensemble] quity=edges, weight=0.2xxx
[Ensemble] Selected: detach
```

---

## 3. 完整训练（可选）

```bash
python src/train.py --dataset_name Photo --use_gb --gb_ensemble --num_epochs 700 --trials 5 --device cuda
```