---
name: Bug report
about: Report a reproducible problem with the statista library (distributions, EVA, sensitivity, time-series, plotting)
title: "[Bug]: "
labels: [bug]
assignees: ''
---

## Description
A clear and concise description of the bug. What is wrong and what did you expect instead?

## Minimal Reproducible Example (MRE)
Provide the smallest code snippet that reproduces the issue. Include imports and any necessary setup. If randomness is involved, set a fixed seed.

```python
# Please adjust to the minimal code that triggers the issue
import numpy as np
import pandas as pd
from statista import distributions  # or the relevant module

np.random.seed(42)  # if applicable

# code here
```

### Data sample (if applicable)
- If the bug depends on specific data, include a very small CSV/JSON snippet or describe its structure (columns, dtypes, units, missing values). If data cannot be shared, describe how to synthesize a similar dataset.

```
# Small CSV-like example or a few rows/records
```

## Steps to Reproduce
1. ...
2. ...
3. ...

## Expected behavior
Describe what you expected to happen.

## Actual behavior / Error traceback
Paste the full error/traceback if there is one.

```
<full traceback here>
```

### Plots / Figures
- If the issue is visual (e.g., plotting, distribution fit diagnostics), attach images or paste code that generates the figure.
- Mention the Matplotlib backend if relevant (e.g., Agg, TkAgg).

## Environment
Please complete the following information:
- OS: [e.g., Windows 11, macOS 14, Ubuntu 24.04]
- Python: [e.g., 3.11.7]
- statista version: [e.g., 0.6.3]
- Installation method: [pip, from source, editable install]
- Key dependencies (if relevant):
  - numpy: [e.g., 2.0.1]
  - scipy: [e.g., 1.14.1]
  - pandas: [e.g., 2.2.2]
  - matplotlib: [e.g., 3.9.1]
  - scikit-learn: [e.g., 1.5.1]
- Locale/encoding (if data-related): [e.g., en_US.UTF-8]

## Randomness and Determinism
- Does the bug disappear or change with a different random seed?
- If applicable, specify any seeds used (e.g., np.random.seed(42)).

## Additional context
Add any other context that might help us diagnose the problem (links to docs pages, related issues, configuration, system constraints).
