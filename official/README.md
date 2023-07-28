# Code for "Fixation Time on Directed Graphs"



## Generating small graphs

We use [`nauty`](https://pallini.di.uniroma1.it/) to generate all graphs of certain properties.

Generate all directed graphs:
```
geng -c -q 5 | directg -q > data/directed/direct5.d6
```

Generate all directed oriented graphs:
```
geng -c -q 5 | directg -o -q > data/directed-oriented/direct5.d6
```

## Generating figures

The folder `figure_generators` contains scripts that generate the figures in the main text.
There are also some special files:

* `pepa.py` contains functions to compute fixation times and related quantities; this file is an edited version of
  a file from the paper ...
* `utils.py` contains functions to parse `nauty` output, style the plots, and run simulations