## Why

The current `sweepX.py` has critical limitations that block Phase 1 experiment completion: (1) narrow hyperparameter search space missing key alpha values and dataset-specific filters, (2) Windows incompatibility in command generation, (3) inconsistent result directories causing checkpoint/resume failures, and (4) wrong working directory causing path resolution errors. These issues prevent proper hyperparameter sweeps for Physics/Computers/Photo datasets.

## What Changes

1. **Expand sweepX.py DATASETS**: Run all four datasets (Physics, Computers, Photo, CS) instead of just Physics
2. **Add alpha=0.5 to ALPHAS**: Include middle-ground value between 0.7 and 0.3 for better local/global balance
3. **Expand FILTERS**: Add more candidate combinations for Computers, Photo, Physics with different (quality, similarity, alpha) tuples
4. **Add K=3 to KS**: Add lightweight diffusion option for medium-sized graphs
5. **Fix Windows compatibility**: Use `set CUDA_VISIBLE_DEVICES=0 &&` instead of Linux-style env prefix
6. **Fix result directory**: Add `--results_dir` parameter to train.py, unify to "results" directory
7. **Fix working directory**: Explicitly set CWD to src/ when running train.py from sweepX.py
8. **Update analyze_results.py**: Align RESULTS_DIR to match train.py output

## Capabilities

### New Capabilities
None - all changes are implementation-level fixes.

### Modified Capabilities
None - no existing spec requirements are being changed.

## Impact

- **Files modified**: `tools/sweepX.py`, `src/train.py`, `tools/analyze_results.py`
- **Breaking changes**: None - all changes are backward-compatible improvements
- **Dependencies**: No new external dependencies required