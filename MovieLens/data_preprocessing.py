import pickle
from os.path import join
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from utils import get_dataframe


def merge_dataframes(csvs, data_dir, plots_folder, recording_file):
    # first get infos from imdb data that are also in movie lens
    print('\tlinking together imdb data with movielens data ...')

    # merging imdb infos with movielens data
    df = merge_imdb_data(csvs, recording_file)

    print('\tmovies df preprocessing ...')
    df_movies = df_movies_preprocessing(df['movieId'], csvs, recording_file)

    df = pd.merge(df_movies, df, on=['movieId'], how='left')

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Links.csv + basics.tsv + crew.tsv + movies.csv df shape:\n')
        rec_file.write(str(df.shape) + '\n\n')

    print('\tadding ratings df infos ...')
    df_ratings = df_ratings_preprocessing(csvs, recording_file)

    common_movie_ids = df['movieId'].isin(df_ratings['movieId'])
    with open(recording_file, 'a') as rec_file:
        rec_file.write('Movie ids in movies.csv that have also an id in ratings.csv:\n')
        rec_file.write(str(common_movie_ids.value_counts()) + '\n\n')

    print('\t\tmerging ratings df with links/basics/crew/movies df ...')
    df = pd.merge(df_ratings, df, on=['movieId'], how='right')

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Links.csv + basics.tsv + crew.tsv + movies.csv + ratings.csv df shape:\n')
        rec_file.write(str(df.shape) + '\n\n')

    # since the purpose is to predict rating, let's dropna of that column
    df.dropna(subset=['rating'], inplace=True)

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Links.csv + basics.tsv + crew.tsv + movies.csv + ratings.csv df shape:\n')
        rec_file.write(str(df.head()) + '\n\n')
        rec_file.write(str(df.dtypes) + '\n\n')
        rec_file.write(str(df.shape) + '\n\n')
        rec_file.write(str(df.isnull().sum()) + '\n\n')

    print('\tadding number of user tag per movie ...')

    # use tags df only to count number of tags given to a movie
    df_tags = pd.read_csv(get_dataframe(csvs, 'tags.csv'),
                          usecols=['movieId', 'tag'])
    df_tags.dropna(inplace=True)
    tags_per_movie = df_tags.groupby('movieId')['tag'].count().reset_index()
    tags_per_movie = tags_per_movie.rename(columns={'tag': 'user_tags_count'})

    df = pd.merge(tags_per_movie, df, on=['movieId'], how='right')

    print('\tadding most relevant tag per movie ...')
    # add most relevant tag from genome scores/tags dataset
    df_genome_tags = genome_tags_handling(csvs, plots_folder, recording_file)

    df = pd.merge(df_genome_tags, df, on=['movieId'], how='right')

    def avg_rating(group):
        # weighted mean with user count
        mean = np.average(group['rating'], weights=group['user_count'])
        diff = np.round(mean - np.floor(mean), decimals=1)

        if diff >= 0.7:
            return np.ceil(mean)
        elif diff <= 0.3:
            return np.floor(mean)
        else:
            return np.trunc(mean) + 0.5

    # now df has lots of different ratings for the same film
    # so there are lots of samples that differs only on the label

    mean_rating_per_movie = df.groupby(by=['movieId'])[['rating', 'user_count']].apply(avg_rating)
    user_count_per_movie = df.groupby(by=['movieId'])['user_count'].sum()

    mean_rating_per_movie = pd.DataFrame({'movieId': mean_rating_per_movie.index,
                                          'rating': mean_rating_per_movie.values})
    user_count_per_movie = pd.DataFrame({'movieId': user_count_per_movie.index,
                                         'user_count': user_count_per_movie.values})

    df.drop('rating', axis=1, inplace=True)
    df.drop('user_count', axis=1, inplace=True)
    df.drop_duplicates(inplace=True)

    df = pd.merge(df, mean_rating_per_movie, on=['movieId'], how='left')
    df = pd.merge(df, user_count_per_movie, on=['movieId'], how='left')

    # null_values_count drops also NaN
    df = null_values_count(df, plots_folder)

    df = encode_categorical_data(df, plots_folder, recording_file)

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Final df head:\n')
        rec_file.write(str(df.head()) + '\n\n')

    # saving final df before making some visualization/encoding of data
    df.to_csv(join(data_dir, 'final.csv'))


def merge_imdb_data(csvs, recording_file):
    links = pd.read_csv(get_dataframe(csvs, 'links.csv'),
                        usecols=['movieId', 'imdbId'],
                        dtype={'movieId': 'int64',
                               'imdbId': 'int64'})

    links['imdbId'] = 'tt' + links['imdbId'].astype(str).apply(lambda x: x.zfill(7))
    links = links.rename(columns={'imdbId': 'tconst'})

    basics = pd.read_csv(get_dataframe(csvs, 'title.basics.tsv'),
                         sep='\t',
                         header=0,
                         usecols=['tconst',
                                  'runtimeMinutes'],
                         na_values='\\N',
                         keep_default_na=False,
                         low_memory=False
                         )
    basics.replace('\\N', np.nan, inplace=True)

    df = pd.merge(basics, links, on=['tconst'], how='right')

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Links.csv + basics.tsv df shape:\n')
        rec_file.write(str(df.shape) + '\n\n')

    crew = pd.read_csv(get_dataframe(csvs, 'title.crew.tsv'),
                       sep='\t',
                       header=0,
                       usecols=['tconst',
                                'directors'],
                       dtype={'tconst': str,
                              'directors': 'category'},
                       na_values='\\N',
                       keep_default_na=False
                       )
    crew.replace('\\N', np.nan, inplace=True)

    with open(recording_file, 'a') as rec_file:
        rec_file.write('crew.tsv df, directors column unique values:\n')
        rec_file.write(str(len(crew['directors'].unique())) + '\n\n')

    df = pd.merge(crew, df, on=['tconst'], how='right')

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Links.csv + basics.tsv + crew.tsv df shape:\n')
        rec_file.write(str(df.shape) + '\n\n')

    # no more use of imdb identifier for films
    df.drop('tconst', axis=1, inplace=True)

    return df


def df_movies_preprocessing(df, csvs, recording_file):
    df_movies = pd.read_csv(get_dataframe(csvs, 'movies.csv'),
                            usecols=['movieId', 'title', 'genres'],
                            dtype={'movieId': 'int64',
                                   'title': str,
                                   'genres': 'category'})
    # resolve alias in title by merging their genre
    df_movies = merge_duplicates(df_movies)
    # get year from title column
    year = df_movies['title'].str.extract('(.*)\\((\\d{4})\\)', expand=True)
    df_movies['year'] = np.array(year[1].astype('float64'))

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Movies dataframe shape:\n')
        rec_file.write(str(df_movies.shape) + '\n\n')

    df_movies.drop('title', axis=1, inplace=True)
    # drop movies without year
    df_movies.dropna(inplace=True)

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Movies dataframe shape without NaN:\n')
        rec_file.write(str(df_movies.shape) + '\n\n')

    print('\tadding to movies.csv, imdb infos ...')
    common_movie_ids = df.isin(df_movies['movieId'])

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Movie ids in links.csv that are also in movies.csv:\n')
        rec_file.write(str(common_movie_ids.value_counts()) + '\n\n')

    return df_movies


# ad-hoc function for movies dataframe since it's the only one with duplicates
def merge_duplicates(df_movies):
    def merge_genre(group):
        genres = list('|'.join(group).split('|'))
        for g in genres:
            g.lower()
        genres = set(genres)
        if '(no genres listed)' in genres and len(genres) > 1:
            genres.remove('(no genres listed)')
        return '|'.join(list(genres))

    # keep=false so we can get all duplicates
    duplicates = df_movies.loc[df_movies['title'].duplicated(keep=False), :]
    # now merge together rows since films are duplicate just for the title
    merged_genre = duplicates.groupby('title')['genres'].apply(merge_genre)
    merged_genre = pd.DataFrame({'title': merged_genre.index,
                                 'genres': merged_genre.values})
    merged_genre = pd.merge(df_movies, merged_genre, on=['title'], how='right')
    merged_genre.drop('genres_x', axis=1, inplace=True)
    merged_genre.drop_duplicates(subset=['title', 'genres_y'], keep='first', inplace=True)
    merged_genre.rename(columns={'genres_y': 'genres'}, inplace=True)
    df_movies.drop_duplicates(subset=['title'], keep=False, inplace=True)

    return pd.concat([df_movies, merged_genre]).sort_values(by='movieId')


def df_ratings_preprocessing(csvs, recording_file):
    df_ratings = pd.read_csv(get_dataframe(csvs, 'ratings.csv'),
                             usecols=['userId', 'movieId', 'rating'],
                             dtype={'userId': 'int64',
                                    'movieId': 'int64',
                                    'rating': 'float64'}
                             )

    # get number of users that gave a specific rating to a movie
    print('\t\tremoving ratings with small number of users ...')
    users_per_rating = df_ratings.groupby(by=['movieId', 'rating'])['userId'].count().reset_index()
    users_per_rating = users_per_rating.rename(columns={'userId': 'user_count'})

    df_1 = users_per_rating.loc[users_per_rating['user_count'] > 1]
    df_20 = users_per_rating.loc[users_per_rating['user_count'] > 20]
    df_50 = users_per_rating.loc[users_per_rating['user_count'] > 50]
    df_75 = users_per_rating.loc[users_per_rating['user_count'] > 75]
    df_100 = users_per_rating.loc[users_per_rating['user_count'] > 100]

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Number of rows in ratings.csv:\n')
        rec_file.write(str(users_per_rating.shape[0]) + '\n\n')
        rec_file.write('Number of movies with a rating with a number of users > 1:\n')
        rec_file.write(str(df_1.shape[0]) + '\n\n')
        rec_file.write('Number of movies with a rating with a number of users > 20:\n')
        rec_file.write(str(df_20.shape[0]) + '\n\n')
        rec_file.write('Number of movies with a rating with a number of users > 50:\n')
        rec_file.write(str(df_50.shape[0]) + '\n\n')
        rec_file.write('Number of movies with a rating with a number of users > 75:\n')
        rec_file.write(str(df_75.shape[0]) + '\n\n')
        rec_file.write('Number of movies with a rating with a number of users > 100:\n')
        rec_file.write(str(df_100.shape[0]) + '\n\n')

    df_ratings.drop('userId', axis=1, inplace=True)
    df_ratings.drop_duplicates(inplace=True)
    df_ratings = pd.merge(df_20, df_ratings, on=['movieId', 'rating'], how='left')

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Number of ratings in ratings.csv (removed ratings with n° users < 50):\n')
        rec_file.write(str(df_ratings.shape[0]) + '\n\n')

    return df_ratings


def genome_tags_handling(csvs, plots_folder, recording_file):
    # read genome score/tags dfs and pivot to get dense matrix
    df_genome_scores = pd.read_csv(get_dataframe(csvs, 'genome-scores.csv'))
    df_genome_tags = pd.read_csv(get_dataframe(csvs, 'genome-tags.csv'))
    df_genome = pd.merge(df_genome_tags, df_genome_scores, on=['tagId'], how='left')[['movieId', 'relevance', 'tag']]
    df_genome = df_genome.pivot(index='movieId', columns='tag', values='relevance')

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Genome scores dense matrix:\n')
        rec_file.write(str(df_genome) + '\n\n')

    most_relevant_tag_per_movie = df_genome.transpose().max()
    most_relevant_tag_per_movie = pd.DataFrame({'movieId': most_relevant_tag_per_movie.index,
                                                'relevance': most_relevant_tag_per_movie.values})

    most_relevant_tag_per_movie = pd.merge(most_relevant_tag_per_movie, df_genome_scores, on=['movieId', 'relevance'],
                                           how='left')
    most_relevant_tag_per_movie.drop_duplicates(subset=['movieId', 'relevance'], inplace=True)
    most_relevant_tag_per_movie = most_relevant_tag_per_movie[['movieId', 'tagId']]

    # hey, categorical data already encoded to int
    return most_relevant_tag_per_movie


def null_values_count(df, plots_folder):
    # null values
    null_values = df.isnull().sum()
    null_values = null_values[null_values.values > 0]
    ax = sns.barplot(x=null_values.index, y=null_values.values, palette='deep')
    ax.set_title('Null values count')
    ax.set(xlabel='columns', ylabel='count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(join(plots_folder, 'null_values_count.jpg'))
    plt.clf()

    # drop NaN in order to make plots
    df.dropna(inplace=True)

    return df


def encode_categorical_data(df, plots_folder, recording_file):
    # tagId column is already encoded

    dummies_data = df['genres'].str.get_dummies('|')
    count_movie_per_genre = dummies_data.sum()
    count_genre_per_movie = dummies_data.transpose().sum()

    # plots genres distribution
    ax = sns.barplot(x=count_movie_per_genre.index, y=count_movie_per_genre.values, palette='deep')
    ax.set_title('Movie Genres Distribution')
    ax.set(xlabel='genres', ylabel='count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(join(plots_folder, 'movie_genres_distribution(final df).jpg'))
    plt.clf()

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Number of distinct genres per movie:\n')
        rec_file.write(str(count_genre_per_movie.describe()) + '\n\n')

    ax = sns.histplot(data=count_genre_per_movie.values, palette='deep', binwidth=0.2)
    ax.set_title('Number of genres per movie')
    ax.set(xlabel='n°genres', ylabel='count')
    plt.tight_layout()
    plt.savefig(join(plots_folder, 'number_of_genres_per_movie(final df).jpg'))
    plt.clf()

    dummies_data = df['directors'].str.get_dummies(',')
    count_director_movie = dummies_data.sum()
    count_movie_director = dummies_data.transpose().sum()

    # plots director distribution
    ax = sns.histplot(data=count_director_movie.values, palette='deep')
    ax.set_title('Directors Distribution')
    ax.set(xlabel='director id', ylabel='count')
    plt.tight_layout()
    plt.savefig(join(plots_folder, 'directors_distribution(final df).jpg'))
    plt.clf()

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Number of directors per movie:\n')
        rec_file.write(str(count_movie_director.describe()) + '\n\n')

    ax = sns.histplot(data=count_movie_director.values, palette='deep', binwidth=0.2)
    ax.set_title('Number of directors per movie')
    ax.set(xlabel='n°genres', ylabel='count')
    plt.tight_layout()
    plt.savefig(join(plots_folder, 'number_of_directors_per_movie(final df).jpg'))
    plt.clf()

    genres = np.array(df['genres'])
    label_encoder = LabelEncoder()
    label_encoder.fit(genres)
    df['genres'] = label_encoder.transform(genres)
    # save label_encoder for embeddings visualization
    # pickle.dump(label_encoder, open(join(plots_folder, 'genres_label_encoder.pkl'), 'wb'))

    directors = np.array(df['directors'])
    df['directors'] = LabelEncoder().fit_transform(directors)

    tagId = np.array(df['tagId'])
    df['tagId'] = LabelEncoder().fit_transform(tagId)

    return df


def balance_dataset(X, y):
    # count samples of each target class
    counts_unique = np.unique(y, return_counts=True)

    # path = 'C:\\Users\\gaspa\\PycharmProjects\\DataAnalytics\\project_material\\plots\\imbalance'
    # ax = sns.barplot(x=counts_unique[0], y=counts_unique[1], palette='deep')
    # ax.set_title('Rating categories count')
    # ax.set(xlabel='rating category', ylabel='count')
    # plt.tight_layout()
    # plt.savefig(join(path, 'rating_categories_count.jpg'))
    # plt.clf()

    # first/second percentile chosen from plots
    first_percentile = np.percentile(y, 50)
    second_percentile = np.percentile(y, 50)
    num_samples_for_class = []

    for i in range(len(counts_unique[1])):
        if counts_unique[1][i] >= counts_unique[1][int(second_percentile)]:
            num_samples_for_class.append(counts_unique[1][int(second_percentile)])
        else:
            num_samples_for_class.append(counts_unique[1][i])

    # under sample to second percentile classes over represented
    under_strategy = dict(zip(counts_unique[0], num_samples_for_class))

    # apply Over/Under sampling
    under = RandomUnderSampler(random_state=42, sampling_strategy=under_strategy)
    X, y = under.fit_resample(X, y)

    counts_unique = np.unique(y, return_counts=True)
    num_samples_for_class = []

    for i in range(len(counts_unique[1])):
        if counts_unique[1][i] <= counts_unique[1][int(first_percentile)]:
            num_samples_for_class.append(counts_unique[1][int(first_percentile)])
        else:
            num_samples_for_class.append(counts_unique[1][i])

    # over sample to first percentile classes under represented
    over_strategy = dict(zip(counts_unique[0], num_samples_for_class))

    over = RandomOverSampler(random_state=42, sampling_strategy=over_strategy)
    X, y = over.fit_resample(X, y)

    # counts_unique = np.unique(y, return_counts=True)
    # ax = sns.barplot(x=counts_unique[0], y=counts_unique[1], palette='deep')
    # ax.set_title('Rating count (training set)')
    # ax.set(xlabel='rating class', ylabel='count')
    # plt.tight_layout()
    # plt.savefig(join(path, 'rating_count_after_balance.jpg'))
    # plt.clf()

    return X.astype(np.float32), y.astype(np.float32)


def normalize_scale_and_balance(model_type, y_train, train_idx_hpo, val_idx, normalizer=None, scaler=None, X_train=None,
                                X_train_cat=None, X_train_cont=None):

    """
    Last stage of Data Preprocessing.
    Correct unbalanced dataset with imbalanced-dataset library and apply
    min-max scaling to continuous features
    """

    if model_type != 'TABT':
        # balance classes distribution with imblearn
        X, y = balance_dataset(X_train[train_idx_hpo, :], y_train[train_idx_hpo])
        X_val, y_val = X_train[val_idx, :], y_train[val_idx]
        # try just to min-max scale data
        # normalizer.fit(X)
        # X = normalizer.transform(X)
        scaler.fit(X)
        X = scaler.transform(X)
        X_val = scaler.transform(X_val)
        # X_val = scaler.transform(normalizer.transform(X_val))

        return X, X_val, y, y_val

    else:
        X_cat, X_cont = X_train_cat[train_idx_hpo, :], X_train_cont[train_idx_hpo, :]
        y = y_train[train_idx_hpo]

        X, y = balance_dataset(np.concatenate([X_cat, X_cont], axis=1), y)

        # categorical data corresponds to the first three cols of X
        X_cat, X_cont = X[:, :3], X[:, 3:]

        # normalize and scale only continuous features
        # normalizer.fit(X_cont)
        # X_cont = normalizer.transform(X_cont)
        scaler.fit(X_cont)
        X_cont = scaler.transform(X_cont)

        X_cat_val, X_cont_val = X_train_cat[val_idx, :], X_train_cont[val_idx, :]
        X_cont_val = scaler.transform(X_cont_val)
        # X_cont_val = scaler.transform(normalizer.transform(X_cont_val))
        y_val = y_train[val_idx]

        return np.concatenate([X_cat, X_cont], axis=1), np.concatenate([X_cat_val, X_cont_val], axis=1), y, y_val
