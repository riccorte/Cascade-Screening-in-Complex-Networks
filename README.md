# Cascade Screening in Complex Networks

Code for my internship at CoMuNe Lab (University of Padova), which leads into my master's thesis.

## Contents

| Notebook | What it does |
|---|---|
| `notebooks/01_site_percolation_mp_vs_mc.ipynb` | Site percolation: message passing vs Monte Carlo on K4, a random regular graph, Erdős-Rényi and Barabási-Albert. Also covers node dependent φ, where the most connected nodes are removed |
| `notebooks/02_mp_accuracy_clustering.ipynb` | Accuracy of message passing when the network has short loops: random graphs with triangles (Newman 2009), error on the curve and on the threshold, and how it changes with the size of the network |
| `notebooks/03_hybrid_phi_method.ipynb` | Screening with message passing (hybrid φ: screened nodes at φ = 1): the matrix M = diag(φ) G, check of φ_eff, critical threshold by bisection |
| `notebooks/04_hybrid_phi_synthetic.ipynb` | Hybrid φ on three test networks (communities, Erdős-Rényi, Barabási-Albert) with random, degree and betweenness screening: threshold, critical budget, total robustness, curves S(φ), check against Monte Carlo |
| `notebooks/05_giant_component_mp_vs_mc.ipynb` | What message passing and Monte Carlo call the giant component on a finite network: checks at φ = 0, toy networks, trees and cycles, check of the messages by hand |
| `notebooks/06_protein_networks.ipynb` | The analysis of notebooks 03 and 04 on four protein interaction networks |
| `notebooks/07_threshold_fit.ipynb` | An operational threshold from a power-law fit of S(φ), the same rule for message passing and Monte Carlo |
| `notebooks/08_node_impact.ipynb` | Impact of a single node: D_i = S_on − S_off, indirect damage, check against paired Monte Carlo, shift of the threshold, removal of many nodes |

`notebooks/mp.py` collects the functions shared by notebooks 02 and 04-08 (message passing, Monte Carlo, the matrices G and
M, screening protocols, compiled solvers, the threshold fit). Notebooks 01 and 03 define their functions in their own cells.

The notes that go with the code (derivations and results) are on Overleaf.

## Data and saved results

- `data/raw/zitnik2019_treeoflife/treeoflife.interactomes/`: the four protein interaction networks of the tree-of-life
  dataset of Zitnik et al. (2019) used in notebooks 06-08, one edge list per species, named by its taxonomy id
  (1148 Synechocystis, 85962 H. pylori, 5833 P. falciparum, 511145 E. coli).
- `notebooks/risultati/NN/`: the csv of the long runs of notebook NN. The cells that produce them read the csv when it
  is there (notebooks 02-04: flags `ESEGUI_RUN... = False`; notebooks 06-08: networks already in the csv are skipped),
  so the notebooks run in minutes instead of hours. To run again from scratch, set the flag to `True` or remove the csv.

## Running it

Python 3.12. Install the packages:

```bash
pip install -r requirements.txt
```

Then open a notebook from the `notebooks/` folder (it imports `mp.py` and reads `risultati/` from there) and run it top to
bottom. Notebook 01 takes about 7 minutes.
