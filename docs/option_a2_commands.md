# Option A2 集成投票方案 - 服务器验证命令

> 在服务器上依次执行以下命令进行验证

---

## 1. 验证语法正确性

```bash
cd /Users/didi/Desktop/GBGCL/src
python3 -m py_compile gb_utils.py
python3 -m py_compile train.py
echo "Syntax OK"
```

---

## 2. 测试模式运行（短训练）- 对比基线

```bash
cd src

# baseline: homo
python train.py --dataset_name Photo --use_gb --gb_quity homo --num_epochs 50 --trials 1 --device cpu

# baseline: detach
python train.py --dataset_name Photo --use_gb --gb_quity detach --num_epochs 50 --trials 1 --device cpu

# baseline: edges
python train.py --dataset_name Photo --use_gb --gb_quity edges --num_epochs 50 --trials 1 --device cpu
```

---

## 3. Ensemble 模式运行

```bash
python train.py --dataset_name Photo --use_gb --gb_ensemble --gb_ensemble_quities homo,detach,edges --num_epochs 50 --trials 1 --device cpu
```

预期输出日志：
```
[Ensemble] quity=homo, weight=0.2xxx
[Ensemble] quity=detach, weight=0.6xxx
[Ensemble] quity=edges, weight=0.2xxx
[Ensemble] Selected: detach
```

---

## 4. 完整训练（700 epochs, 5 trials）

```bash
python train.py --dataset_name Photo --use_gb --gb_ensemble --num_epochs 700 --trials 5 --device cuda
```