## Context

The GBGCL project uses `sweepX.py` to run hyperparameter sweeps across datasets. Current state:
- `sweepX.py` only runs Physics dataset, narrow alpha search [0.7, 0.3]
- Windows commands use Linux-style env vars, causing failures
- `train.py` writes to `results/` but `sweepX.py` reads from `results_CUDA/`
- Working directory defaults to `tools/` causing path resolution failures

## Goals / Non-Goals

**Goals:**
1. Expand DATASETS to include Physics, Computers, Photo, CS
2. Add alpha=0.5 and K=3 to search space
3. Expand FILTERS with more dataset-specific combinations
4. Fix Windows compatibility in command generation
5. Unify results directory across all scripts
6. Fix working directory for subprocess calls

**Non-Goals:**
- No new features or algorithm changes
- No spec requirement changes - this is implementation-only

## Decisions

1. **Results directory**: Use `results/` as the single source of truth (not `results_CUDA/`)
   - Rationale: `train.py` already writes to `results/`, aligning others avoids confusion
   - Alternative: Use `results_CUDA/` - rejected as it requires modifying train.py output path

2. **Windows command format**: Use `set CUDA_VISIBLE_DEVICES=0 &&` prefix
   - Rationale: PowerShell-compatible, avoids shell syntax errors
   - Alternative: Use subprocess.run with env dict - more Pythonic but more code changes

3. **Working directory**: Set `cwd` to `src/` in subprocess.call
   - Rationale: Resolves dataset paths and relative imports correctly

## Risks / Trade-offs

- [Risk] K=3 may cause under-smoothing on Physics (34K nodes) → Mitigation: Keep K=5,10,20 as primary options
- [Risk] Alpha=0.5 may not improve over 0.3/0.7 → Mitigation: Still useful as data point for analysis
- [Risk] Windows subprocess shell=True may fail with special chars → Mitigation: Use shlex.quote for all user-provided strings