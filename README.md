# TabTransformer: Tabular Data Modeling Using Contextual Embeddings

This project is inteded to review and analyze following article:

https://arxiv.org/pdf/2012.06678.pdf

where novel TabTransformer architecture is presented in order to 
deal with learning task on tabular data.

# Run Instructions

First directory 'EasyTabt' contains a minimal code
to try out TabTransformer (running time < 5 min),
while 'MovieLens' contains a whole pipeline to test
TabTransformer on MovieLens dataset, following papers
steps only for supervised learning.

To run 'EasyTabt' demo, first run easy_tabt.py
to generate both TabTransfromer and GBDT roc curves
and also TabTransformer learned embedding. In 'runs' folder
you can observe tabt transformer training loss
decreasing correctly and in 'plots' roc curves are saved.
Optionally run embedding_visualization.py to generate data
exploited by Tensorboard embedding projector, in order to 
visualize embeddings before and after passing through transformer
layers.

To run 'MovieLens' demo, run tab_transformer_demo.py and
the whole pipeline will start.

# Third Party Code

- Movielens dataset:

[https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)

- IMDB dataset:

[https://www.imdb.com/interfaces/](https://www.imdb.com/interfaces/)

- Phil Wang repository for the model: 

[https://github.com/lucidrains/tab-transformer-pytorch](https://github.com/lucidrains/tab-transformer-pytorch)

- Tensorboard embedding projector:

[https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin)

- ROC curves visualizationfrom scikit-learn:

https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

# Credits

Review author: Gaspari Michele
