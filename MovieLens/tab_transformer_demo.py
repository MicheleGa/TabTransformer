import os
from os import makedirs, environ
from os.path import join, dirname
from shutil import rmtree
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from dataset import MovieLensDataset
from model import model_train_and_test
from data_visualization import plot_final_demo_results

environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = False

torch.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)


def main(args):

    """Starts pipeline workflow"""

    root = dirname(__file__)
    data_path = join(root, 'data')
    models_path = join(root, 'models')
    plots_path = join(root, 'plots')
    recording_file = join(root, 'pipeline_output.txt')

    # clean all (previous) outputs in recording file
    with open(recording_file, 'r+') as rec_file:
        rec_file.truncate(0)

    # make/clean plots folder
    plots_dir = Path(plots_path)
    if plots_dir.exists():
        rmtree(plots_dir)
    makedirs(plots_dir)

    # make/clean models folder
    models_dir = Path(models_path)
    if models_dir.exists():
        rmtree(models_dir)
    makedirs(models_dir)

    dataset = MovieLensDataset(data_path,
                               recording_file,
                               plots_path)

    # download datasets using requests library
    dataset.download_and_extract(args.movie_lens_zip_url, args.imdb_url)

    # build input dataframe for models, performing data preprocessing
    dataset.create_dataframe()
    # final plots before training and TSNE visualization
    dataset.visualization()

    print('\n**** Model ****\n')
    with open(recording_file, 'a') as rec_file:
        rec_file.write('\n**** Model ****\n\n')

    results = {'train_roc_auc_mean': [], 'train_roc_auc_std': [], 'mean_test_auc_score': [], 'std_test_auc_score': []}

    # supervised training and testing
    for model in args.models:
        makedirs(join(models_dir, model))
        makedirs(join(plots_path, model))

        if model == 'TABT':
            # tab transformer needs to separate categorical features from continuous ones
            dataset.load_tab_transformer_data()
        else:
            # set dataset X, y, num_classes fields and encode categorical data
            dataset.load()

        result = model_train_and_test(dataset,
                                      args.k_fold_cv,
                                      join(models_dir, model),
                                      recording_file,
                                      join(plots_path, model))
        # unpack results
        train_roc_auc_mean, train_roc_auc_std, mean_test_auc_score, std_test_auc_score = result
        results['train_roc_auc_mean'].append(train_roc_auc_mean)
        results['train_roc_auc_std'].append(train_roc_auc_std)
        results['mean_test_auc_score'].append(mean_test_auc_score)
        results['std_test_auc_score'].append(std_test_auc_score)

    results = pd.DataFrame(results)
    results['model'] = args.models

    print(results)
    with open(recording_file, 'a') as rec_file:
        rec_file.write(str(results) + '\n\n')

    plot_final_demo_results(results, plots_path, recording_file)


def result_analysis(plots_folder):

    """Plots from article results"""

    datasets_dataframe = pd.DataFrame()
    datasets_dataframe['name'] = ['albert', '1995_income', 'dota2games', 'hcdr_main', 'adult', 'bank_marketing',
                                  'blastchar', 'insurance_co', 'jasmine', 'online_shoppers', 'philippine', 'qsar_bio',
                                  'seismicbumps', 'shrutime', 'spambase']

    datasets_dataframe['datapoints'] = [425240, 32561, 92650, 307511, 34190, 45211,
                                        7043, 5822, 2984, 12330, 5832, 1055,
                                        2583, 10000, 4601]

    datasets_dataframe['n_features'] = [79, 14, 117, 120, 25, 16,
                                        20, 85, 145, 17, 309,
                                        41,
                                        18, 11, 57]

    datasets_dataframe['positive_class'] = [50, 24.1, 52.7, 8.1, 85.4, 11.7,
                                            26.5, 6.0, 50.0, 15.5, 50.0,
                                            33.7,
                                            6.6, 20.4, 39.4]
    ax = sns.barplot(x='name', y='positive_class', data=datasets_dataframe, palette='deep')
    ax.set_title('Balance in target class')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(plots_folder + os.sep + 'article_supervised_balance_in_target_class.jpg')
    plt.show()

    mlp_vs_tabt_vs_gbdt = pd.DataFrame()

    # dataset name column
    mlp_vs_tabt_vs_gbdt['dataset'] = pd.concat([datasets_dataframe.name,
                                                datasets_dataframe.name,
                                                datasets_dataframe.name])
    # add MLP performance
    mlp_performance = [74.0, 90.5, 63.1, 74.3, 72.5, 92.9,
                       83.9, 69.7, 85.1, 91.9, 82.1, 91.0,
                       73.5, 84.6, 98.4]

    # add TABT performance
    tabt_performance = [75.7, 90.6, 63.3, 75.1, 73.7, 93.4,
                        83.5, 74.4, 85.3, 92.7, 83.4, 91.8,
                        75.1, 85.6, 98.5]

    # add GBDT performance
    gbdt_performance = [76.3, 90.6, 62.1, 75.6, 75.6, 93.3,
                        84.7, 73.2, 86.2, 93.0, 81.2, 91.3,
                        75.6, 85.9, 98.7]

    performances = mlp_performance + tabt_performance + gbdt_performance
    mlp_vs_tabt_vs_gbdt['ROC AUC'] = performances

    # add model name column
    mlp_model_name = ['MLP'] * len(mlp_performance)
    tabt_model_name = ['TABT'] * len(tabt_performance)
    gbdt_model_name = ['GBDT'] * len(gbdt_performance)

    models_name = mlp_model_name + tabt_model_name + gbdt_model_name
    mlp_vs_tabt_vs_gbdt['model'] = models_name

    sns.catplot(data=mlp_vs_tabt_vs_gbdt, kind="bar",
                x="dataset", y="ROC AUC", hue="model",
                palette="deep")
    plt.title('MLP vs TABT vs GBDT')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plots_folder + os.sep + 'article_supervised_mlp_vs_tabt_vs_gbdt.jpg')
    plt.show()

    mlp_vs_tabt_vs_gbdt = mlp_vs_tabt_vs_gbdt.rename(columns={'dataset': 'name'})
    df = pd.merge(datasets_dataframe, mlp_vs_tabt_vs_gbdt, on='name', how='left')

    for col in df.columns:
        if col == 'datapoints' or col == 'n_features' or col == 'positive_class':
            sorted_df = df.sort_values(by=[col])
            sns.lineplot(data=sorted_df, x=col, y='ROC AUC', hue='model')
            plt.savefig(plots_folder + os.sep + f'article_supervised_{col}.jpg')
            plt.show()
            if col == 'datapoints':
                sorted_df = sorted_df[sorted_df[col] < 50000]
                sns.lineplot(data=sorted_df, x=col, y='ROC AUC', hue='model')
                plt.savefig(plots_folder + os.sep + f'article_supervised_{col}_zoom.jpg')
                plt.show()
            if col == 'n_features':
                sorted_df = sorted_df[sorted_df[col] < 50]
                sns.lineplot(data=sorted_df, x=col, y='ROC AUC', hue='model')
                plt.savefig(plots_folder + os.sep + f'article_supervised_{col}_zoom.jpg')
                plt.show()


class DotDict(dict):
    # dot.notation access to dictionary attributes
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_args_dict():
    args_dictionary = {
        'movie_lens_zip_url': 'https://files.grouplens.org/datasets/movielens/ml-25m.zip',
        'imdb_url': 'https://datasets.imdbws.com/',
        'data_path': 'data',
        'recording_file': 'pipeline_output.txt',
        'plots_folder': 'plots',
        'models_path': 'models',
        'models': ['TABT', 'GBDT', 'SVM', 'LR', 'MLP'],
        'k_fold_cv': 5
    }

    return DotDict(args_dictionary)


if __name__ == '__main__':
    args = get_args_dict()

    # run pipeline on MovieLens
    main(args)

    # plots results from paper
    result_analysis(args.plots_folder)

