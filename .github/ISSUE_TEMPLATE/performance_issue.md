---
name: Performance issue
about: Report a performance regression or a need for optimization (speed/memory)
title: "[Perf]: "
labels: [performance]
assignees: ''
---

## Summary
Describe the performance problem and why it matters.

## Minimal benchmark / profiling snippet
Provide a minimal code sample that demonstrates the performance issue.

```python
# minimal benchmark
import time
start = time.perf_counter()
# your code
print("elapsed:", time.perf_counter() - start)
```

## Data characteristics
- Dataset size: [rows, columns, points]
- Dtypes and ranges: [brief]
- Sparsity / missing values: [yes/no]

## Expected vs Actual
- Expected time/memory: [e.g., ~100ms, <200MB]
- Actual time/memory: [e.g., ~2s, ~1.2GB]

## Environment
- OS:
- Python:
- statista:
- Dependencies (numpy/scipy/pandas/matplotlib/scikit-learn):

## Profiling results (if available)
Paste cProfile, line_profiler, or memory profiler output.

## Additional context
Related issues/PRs, references, or suggested optimization ideas.
