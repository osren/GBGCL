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

## 2. 测试模式运行（短训练）- 对比基线

```bash
cd /Users/didi/Desktop/GBGCL

# baseline: homo
python src/train.py --dataset_name Photo --use_gb --gb_quity homo --num_epochs 50 --trials 1 --gb_rebuild_every 10 --device cuda

# baseline: detach
python src/train.py --dataset_name Photo --use_gb --gb_quity detach --num_epochs 50 --trials 1 --gb_rebuild_every 10 --device cuda

# baseline: edges
python src/train.py --dataset_name Photo --use_gb --gb_quity edges --num_epochs 50 --trials 1 --gb_rebuild_every 10 --device cuda
```

---

## 3. Ensemble 模式运行

```bash
python src/train.py --dataset_name Photo --use_gb --gb_ensemble --gb_ensemble_quities homo,detach,edges --num_epochs 50 --trials 1 --gb_rebuild_every 10 --device cuda
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
python src/train.py --dataset_name Photo --use_gb --gb_ensemble --num_epochs 700 --trials 5 --device cuda
```