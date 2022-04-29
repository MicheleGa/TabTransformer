import os
from os import listdir, makedirs, sep
from os.path import isfile, join
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from data_acquisition import data_acquisition
from data_visualization import print_dataframe_infos, combine_dataframes, high_dimensional_visualization, \
    plots_final_df_infos
from data_preprocessing import merge_dataframes


class MovieLensDataset(Dataset):
    """
    Pytorch Dataset class containing movies infos from
    https://grouplens.org/datasets/movielens/ and
    https://www.imdb.com/interfaces/
    """

    def __init__(self, data_dir, recording_file, plots_folder):
        self.data_dir = data_dir
        self.recording_file = recording_file
        self.plots_folder = plots_folder

        # X,y will be intialized calling load()
        self.X = None
        self.y = None
        self.num_classes = None

        # fields for load_tab_transofrmer_data()
        self.categories = None  # number of unqiue ids per class
        self.continuous_features = 0  # number of continuous features
        self.X_cat = None
        self.X_cont = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # here we should randomly sample a feature to make RTD approach
        # select a class
        # num_cat_features = self.X_cat.shape[1]
        # category = np.random.randint(0, num_cat_features)

        # now get a random value from that category
        # num_unique_ids = self.categories[category]
        # random_value = np.random.randint(0, num_unique_ids)

        # replace value
        # self.X[idx, category] = random_value

        return self.X[idx, :], self.y[idx]

    def download_and_extract(self, movie_lens_zip_url, imdb_url):

        """Data Acquisition stage"""

        print("**** Data acquisition ****\n")
        # create folder for data and download dataset
        data_dir = Path(self.data_dir)
        if not data_dir.exists():
            makedirs(data_dir)

            # collect movielens data
            movie_lens_dir = join(data_dir, 'movielens')
            makedirs(movie_lens_dir)
            data_acquisition(movie_lens_dir, movie_lens_zip_url)
            print('\n')

            # collect imdb data
            imdb_dir = join(data_dir, 'imdb')
            imdb_files = ['title.crew.tsv.gz', 'title.basics.tsv.gz']
            makedirs(imdb_dir)
            for file in imdb_files:
                data_acquisition(imdb_dir, imdb_url + file)
                print('\n')
        else:
            print('Data already acquired ...')

    def create_dataframe(self):

        """
        First stage of Data Preprocessing where dataframes are analyzed
        and merged together to get final df
        """

        print('\n**** Data Preprocessing ****\n')
        with open(self.recording_file, 'a') as rec_file:
            rec_file.write('**** Data Preprocessing ****\n')

        # get movie lens csv files
        movie_lens_dir = join(self.data_dir, 'movielens', 'ml-25m')
        files = [join(movie_lens_dir, file) for file in listdir(movie_lens_dir) if isfile(join(movie_lens_dir, file))]
        csvs = [path for path in files if path[-4:] == '.csv']

        # get imdb tsv files
        imdb_dir = join(self.data_dir, 'imdb')
        files = [join(imdb_dir, file) for file in listdir(imdb_dir) if isfile(join(imdb_dir, file))]
        imdb_tsvs = [path for path in files if path[-4:] == '.tsv']

        csvs.extend(imdb_tsvs)

        # read csv files in order to plot some general information of each dataframe
        for path in csvs:
            print(f'Analyzing: {path} ...')
            # plots info of just some dfs
            if path.endswith('genome-tags.csv') or path.endswith('links.csv') or path.endswith('.tsv'):
                print_dataframe_infos(path, self.recording_file, self.plots_folder)
            else:
                plots_dir = join(self.plots_folder, path.split(sep)[-1][:-4])
                makedirs(plots_dir)
                print_dataframe_infos(path, self.recording_file, plots_dir)

        # plot some graphs combining datasets
        print('Merging dataframe to make more accurate analysis ...')
        plots_dir = join(self.plots_folder, 'mixed_dfs')
        makedirs(plots_dir)
        combine_dataframes(csvs, self.recording_file, plots_dir)

        # get final dataset to train models
        print('Merging all dataframes together to get final dataset:')
        plots_dir = join(self.plots_folder, 'final_df')
        makedirs(plots_dir)
        merge_dataframes(csvs, self.data_dir, plots_dir, self.recording_file)

    def visualization(self):

        """
        Data Visualization of final df before starting model definition,
        training and testing
        """

        print('Visualizing final dataframe and performing normalization/scaling...')
        df = pd.read_csv(join(self.data_dir, 'final.csv'))
        # Pandas default saves as a column the row index
        df.drop(df.columns[0], axis=1, inplace=True)

        print('\tplot some general info of final df ...')
        # plot final df infos (return df without movieId column)
        plots_dir = join(self.plots_folder, 'final_df')
        df = plots_final_df_infos(df, plots_dir, self.recording_file)

        # high dimensional visualization
        print('\tperforming normalization/scaling (visualization with TSNE) ...')
        labels = df['rating']
        df.drop('rating', axis=1, inplace=True)
        df = df.to_numpy()

        # first try TSNE with no normalization/scaling to get perplexity value
        plots_path = join(plots_dir, 'TSNE')
        makedirs(plots_path)

        print('\t\tdf no norm nor scaled')
        # no norm or scaling
        makedirs(join(plots_path, 'no_norm_no_scaling'))
        # -15 is for cosine metric
        perplexity_values = [7, 15, -15, 30]
        for p in perplexity_values:
            if p > 0:
                tsne = TSNE(n_components=2, init='random', random_state=42, learning_rate='auto', perplexity=p)
            else:
                tsne = TSNE(n_components=2, init='random', random_state=42, learning_rate='auto', perplexity=-p,
                            metric='cosine', square_distances=True)
            tsne_results = tsne.fit_transform(df)

            tsne_data = pd.DataFrame({'rating': labels.values})
            tsne_data['tsne-2d-one'] = tsne_results[:, 0]
            tsne_data['tsne-2d-two'] = tsne_results[:, 1]

            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x='tsne-2d-one', y='tsne-2d-two',
                hue='rating',
                palette='deep',
                data=tsne_data
            )
            if p > 0:
                plt.savefig(join(plots_path, 'no_norm_no_scaling', f'tsne_plot_perplexity{p}_.jpg'))
            else:
                plt.savefig(join(plots_path, 'no_norm_no_scaling', f'tsne_plot_perplexity{-p}_cosine_.jpg'))
            plt.clf()

        # try different kids of normalization and scaling
        normalizations = ['norm1', 'norm2', 'inf', 'sign_square_root_norm']
        for norm in normalizations:
            print(f'\t\tdf {norm} with:')
            df = normalize_dataset(df, norm)
            file_plot_path = join(plots_path, f'df_{norm}')
            makedirs(file_plot_path)
            high_dimensional_visualization(df, labels, file_plot_path)

    def load(self):

        """Final dataset loading for GBDT, SVM, LR and MLP models"""

        # final dataset after preprocessing
        print('Loading dataset ...')
        df = pd.read_csv(join(self.data_dir, 'final.csv'))

        # remove useless columns as row index and movieId
        df.drop(df.columns[0], axis=1, inplace=True)
        df.drop('movieId', axis=1, inplace=True)

        # encode target variable, resulting in [0, ..., 9] categories
        labels = df['rating']
        labels = LabelEncoder().fit_transform(labels)

        df.drop('rating', axis=1, inplace=True)

        df = df.to_numpy()

        self.X = df
        self.y = labels
        self.num_classes = len(np.unique(labels))

    def load_tab_transformer_data(self):

        """Final dataset loading for TabTransformer model"""

        # final dataset after preprocessing
        print('Loading tab transformer dataset ...')
        df = pd.read_csv(join(self.data_dir, 'final.csv'))
        df.drop(df.columns[0], axis=1, inplace=True)
        df.drop('movieId', axis=1, inplace=True)

        # get target variable
        labels = df['rating']
        labels = LabelEncoder().fit_transform(labels)

        df.drop('rating', axis=1, inplace=True)

        # separate categorical from continuous data
        cat_df = df[['genres', 'directors', 'tagId']]
        cont_df = df.drop(['genres', 'directors', 'tagId'], axis=1)

        self.continuous_features = cont_df.shape[1]
        self.categories = cat_df.nunique().to_numpy()
        self.num_classes = len(np.unique(labels))
        self.X_cat = cat_df.to_numpy()
        self.X_cont = cont_df.to_numpy()
        self.y = labels


def normalize_dataset(df, norm):

    """Apply different kind of normalization to final dataset"""

    # first get norm for each row (axis=1), then divide each row for its norm
    if norm == 'norm1':
        return df / np.linalg.norm(df, ord=1, axis=1)[:, None]

    elif norm == 'norm2':
        return df / np.linalg.norm(df, ord=2, axis=1)[:, None]

    elif norm == 'inf':
        return df / np.linalg.norm(df, ord=np.inf, axis=1)[:, None]

    else:
        return np.sign(df) * np.sqrt(np.abs(df))


class TorchDataset(Dataset):

    """Pytorch Dataset class for MLP"""

    def __init__(self, data, labels):
        self.X = torch.FloatTensor(data)
        self.y = torch.LongTensor(labels)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]


class TabTorchDataset(Dataset):

    """Pytorch Dataset class for TabTransformer"""

    def __init__(self, cat_data, cont_data, labels):
        self.X_cat = torch.LongTensor(cat_data)
        self.X_cont = torch.FloatTensor(cont_data)
        self.y = torch.LongTensor(labels)

    def __len__(self):
        return self.X_cat.shape[0]

    def __getitem__(self, idx):
        return self.X_cat[idx, :], self.X_cont[idx, :], self.y[idx]
