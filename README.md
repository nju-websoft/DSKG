# DSKG

DSKG: Deep Sequential models for Knowledge Graphs

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

### Entity Prediction Results

#### FB15K-237

| Models         |  Hits@1  |  Hits@10 |    MRR   |    MR   |
|:---------------|:--------:|:--------:|:--------:|:-------:|
| TransE (our)   |   13.3   |   40.9   |   22.3   |   315   |
| TransR (our)   |   10.9   |   38.2   |   19.9   |   417   |
| PTransE (our)  |   21.0   |   50.1   |   31.4   |   299   |
| DISTMULT       |   15.5   |   41.9   |   24.1   |   254   |
| NLFeat         |    \-    |   41.4   |   27.2   |    \-   |
| ComplEx        |   15.2   |   41.9   |   24.0   |   248   |
| NeuralLP       |    \-    |   36.2   |   24.0   |    \-   |
| ConvE          |   23.9   |   49.1   |   31.6   |   246   |
| InverseModel   |    0.4   |    1.2   |    0.7   |  7,124  |
| DSKG (cascade) |   20.5   |   50.1   |   30.3   |   842   |
| DSKG           | **24.9** | **52.1** | **33.9** | **175** |

#### FB15K

| Models         |  Hits@1  |  Hits@10 |    MRR   |   MR   |
|----------------|:--------:|:--------:|:--------:|:------:|
| TransE (our)   |   30.5   |   73.7   |   45.8   |   71   |
| TransR (our)   |   37.7   |   76.7   |   51.9   |   84   |
| PTransE (our)  |   63.8   |   87.2   |   73.1   |   59   |
| DISTMULT       |   54.6   |   82.4   |   65.4   |   97   |
| NLFeat         |    \-    |   87.0   | **82.1** |   \-   |
| ComplEx        |   59.9   |   84.0   |   69.2   |   \-   |
| NeuralLP       |    \-    |   83.7   |   76.0   |   \-   |
| ConvE          |   67.0   |   87.3   |   74.5   |   64   |
| InverseModel   |   74.3   |   78.6   |   75.9   |  1,563 |
| DSKG (cascade) |   64.9   |   87.7   |   73.0   |   151  |
| DSKG           | **75.3** | **90.2** |   80.9   | **30** |

#### WN18

| Models         |  Hits@1  |  Hits@10 |    MRR   |    MR   |
|----------------|:--------:|:--------:|:--------:|:-------:|
| TransE (our)   |   27.4   |   94.4   |   57.8   |   431   |
| TransR (our)   |   54.8   |   94.7   |   72.6   |   415   |
| PTransE (our)  |   87.3   |   94.2   |   90.5   |   516   |
| DISTMULT       |   72.8   |   93.6   |   82.2   |   902   |
| NLFeat         |    \-    |   94.3   |   94.0   |    \-   |
| ComplEx        |   93.6   |   94.7   |   94.1   |    \-   |
| NeuralLP       |    \-    |   94.5   |   94.0   |    \-   |
| ConvE          |   93.5   |   95.5   |   94.2   |   504   |
| InverseModel   |   75.7   | **96.9** |   85.7   |   602   |
| DSKG (cascade) |   93.9   |   95.0   |   94.3   |   959   |
| DSKG           | **94.2** |   95.2   | **94.6** | **337** |

### Citation

> Lingbing Guo, Qingheng Zhang, Weiyi Ge, Wei Hu*, Yuzhong Qu. DSKG: A Deep Sequential Model for Knowledge Graph Completion. In: CCKS 2018. https://arxiv.org/pdf/1810.12582.pdf
> Lingbing Guo, Qingheng Zhang, Wei Hu∗, Zequn Sun, Yuzhong Qu. Learning to Complete Knowledge Graphs with Deep Sequential Models. Data Intelligence, 1(3):224–243, 2019. http://www.data-intelligence.org/p/25/
