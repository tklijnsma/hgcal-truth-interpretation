# Setup

A conda environment with python 3 and at least `pip install` `uproot`, `matplotlib` and `numpy-indexed`. For the decay tree plots, also `networkx` and `pygraphviz`.

# Usage

```
import ntuputils
event, fig = ntuputils.plot_events(
    'event1001_pdgid6_1GeV_Nov10_finecalo_numEvent10.root'
    )
```