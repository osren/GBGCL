## 1. sweepX.py - Expand Search Space

- [x] 1.1 Update DATASETS to include ["Physics", "Computers", "Photo", "CS"]
- [x] 1.2 Add alpha=0.5 to ALPHAS list (now [0.7, 0.5, 0.3])
- [x] 1.3 Add K=3 to KS list (now [3, 5, 10, 20])
- [x] 1.4 Expand FILTERS with more combinations for Computers, Photo, Physics as specified in PHASE1_修改计划.md

## 2. sweepX.py - Windows Compatibility

- [x] 2.1 Fix build_cmd to detect Windows platform
- [x] 2.2 Use "set CUDA_VISIBLE_DEVICES=0 &&" prefix for Windows
- [x] 2.3 Add src/ prefix to train.py path (python src/train.py)

## 3. train.py - Results Directory

- [x] 3.1 Add --results_dir argument to argparse (default: 'results')
- [x] 3.2 Update csv_path to use os.path.join(args.results_dir, ...)
- [x] 3.3 Add os.makedirs(args.results_dir, exist_ok=True)

## 4. sweepX.py - Results Directory & Working Directory

- [x] 4.1 Change RESULTS_DIR from "results_CUDA" to "results"
- [x] 4.2 Add "--results_dir", RESULTS_DIR to build_cmd arguments
- [x] 4.3 Update run_one to set cwd=src_dir (parent of tools/)
- [x] 4.4 Update result_csv_paths to use RESULTS_DIR

## 5. analyze_results.py - Directory Alignment

- [x] 5.1 Change RESULTS_DIR to "results" to match train.py output