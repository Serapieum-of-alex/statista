---
name: Feature request
about: Propose a new capability or enhancement for statista (API, algorithm, performance, UX)
title: "[Feature]: "
labels: [enhancement]
assignees: ''
---

## Problem statement
What problem are you trying to solve? Why is it important in the context of statistics, distributions, EVA, sensitivity analysis, time-series, or plotting?

## Proposed solution
Describe the feature in detail. If this is an API addition/change, specify the interface:

```python
# Example
import numpy as np
from statista import distributions

def new_function(x: np.ndarray, *, param: float = 1.0) -> float:
    ...
```

- Module(s) affected: [e.g., statista.distributions]
- New classes/functions/methods: [list]
- Input/Output shapes and types: [describe]
- Parameter names/defaults: [describe]

## Example usage
Provide a minimal code snippet demonstrating how the feature would be used.

```python
# sample usage
```

## Alternatives considered
List any alternative approaches or prior art (including SciPy, scikit-learn, statsmodels, etc.). Explain trade-offs.

## Backward compatibility
- Does this change break existing APIs? If yes, describe migration path.

## Performance and numerical stability
- Any expected performance impact (time/memory)?
- Any numerical stability considerations (e.g., underflow/overflow, precision)?

## Documentation
- What docs/tutorials/examples would need to be added/updated?

## Additional context
Links to related issues, papers, or references.
