# DSKG
A Deep Sequential Model for Knowledge Graph Completion

### Requirements
* python 3.x
* tensorflow 1.x
* numpy, pandas
* jupyter

### Running

1. unpack the data.tar.gz, which includes all of three datasets.

2. run jupyter:
<code> jupyter notebook </code>

3. open runDSKG.ipynb & run all cells (Kernel -> Restart & Run All)

You can also directly click runDSKG.ipynb in this page to preview the results we have run.


### Results

| Models                |  Hits@1  |  Hits@10 |    MRR   |    MR   |
|:----------------------|:--------:|:--------:|:--------:|:-------:|
| TransE\\(^\dagger\\)  |   13.3   |   40.9   |   22.3   |   315   |
| TransR\\(^\dagger\\)  |   10.9   |   38.2   |   19.9   |   417   |
| PTransE\\(^\dagger\\) |   21.0   |   50.1   |   31.4   |   299   |
| DISTMULT              |   15.5   |   41.9   |   24.1   |   254   |
| NLFeat                |    \-    |   41.4   |   27.2   |    \-   |
| ComplEx               |   15.2   |   41.9   |   24.0   |   248   |
| NeuralLP              |    \-    |   36.2   |   24.0   |    \-   |
| ConvE                 |   23.9   |   49.1   |   31.6   |   246   |
| InverseModel          |    0.4   |    1.2   |    0.7   |  7,124  |
| DSKG (cascade)        |   20.5   |   50.1   |   30.3   |   842   |
| DSKG                  | **24.9** | **52.1** | **33.9** | **175** |