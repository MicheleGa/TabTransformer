from os import sep
from os.path import join
from itertools import cycle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from utils import get_dataframe


def print_dataframe_infos(file_path, recording_file, plots_folder):

    """First exploration of just downloaded dataframes"""

    # get filename
    filename = file_path.split(sep)[-1]

    # plot some interesting graphs respect to each dataframe
    if filename == 'genome-scores.csv':
        data = pd.read_csv(file_path,
                           usecols=['movieId', 'tagId', 'relevance'],
                           dtype={'movieId': 'int64',
                                  'tagId': 'int64',
                                  'relevance': 'float64'})
        print_generic_infos(data, filename, recording_file)

        relevance = data['relevance']

        ax = sns.kdeplot(data=relevance, palette='deep')
        ax.set_title('Relevance distribution')
        ax.set(xlabel='relevance values', ylabel='count')
        plt.tight_layout()
        plt.savefig(join(plots_folder, 'relevance_distribution.jpg'))
        plt.clf()

    elif filename == 'genome-tags.csv':
        # nothing to show for tags since is just a list of unique identifiers,
        data = pd.read_csv(file_path,
                           usecols=['tagId', 'tag'],
                           dtype={'tagId': 'int64',
                                  'tag': 'category'})
        print_generic_infos(data, filename, recording_file)

    elif filename == 'links.csv':
        # drop tmdbId
        data = pd.read_csv(file_path,
                           usecols=['movieId', 'imdbId'],
                           dtype={'movieId': 'int64',
                                  'imdbId': 'int64'})
        print_generic_infos(data, filename, recording_file)

    elif filename == 'movies.csv':
        data = pd.read_csv(file_path,
                           usecols=['title', 'genres'],
                           dtype={'title': str,
                                  'genres': 'category'})
        print_generic_infos(data, filename, recording_file)

        # genres column adjustment
        dummies_data = data['genres'].str.get_dummies('|')
        count = dummies_data.sum()

        # plots genres distribution
        ax = sns.barplot(x=count.index, y=count.values, palette='deep')
        ax.set_title('Movie Genres Distribution')
        ax.set(xlabel='genres', ylabel='count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        plt.tight_layout()
        plt.savefig(join(plots_folder, 'movie_genres_distribution.jpg'))
        plt.clf()

        # get year from title column
        year = data['title'].str.extract('(.*)\\((\\d{4})\\)', expand=True)
        year = year[1].astype('float64')

        ax = sns.histplot(data=year, palette='deep')
        ax.set(title='Movie years distribution')
        plt.tight_layout()
        plt.savefig(join(plots_folder, 'movie_years_distributions.jpg'))
        plt.clf()

    # both ratings and tags has timestamp (seconds from 1/1/1970)
    elif filename == 'ratings.csv':
        data = pd.read_csv(file_path,
                           usecols=['userId', 'movieId', 'rating'],
                           dtype={'userId': 'int64',
                                  'movieId': 'int64',
                                  'rating': 'float64'})
        print_generic_infos(data, filename, recording_file)

        rating_dataframe_infos(data, recording_file, plots_folder)

    elif filename == 'tags.csv':
        # drop timestamp of tags
        data = pd.read_csv(file_path,
                           usecols=['movieId', 'tag'])
        print_generic_infos(data, filename, recording_file)

        # plot number of tags per movie
        tags_per_movie = data.groupby('movieId')['tag'].count()
        ax = sns.kdeplot(data=tags_per_movie, palette='deep')
        ax.set_title('Number of user tags per movie')
        ax.set(xlabel='movies', ylabel='count')
        plt.tight_layout()
        plt.savefig(join(plots_folder, 'number_of_user_tags_per_movie.jpg'))
        plt.clf()

    # imdb data
    elif filename == 'title.basics.tsv':
        data = pd.read_csv(file_path,
                           sep='\t',
                           header=0,
                           usecols=['tconst',
                                    'isAdult',
                                    'runtimeMinutes'],
                           na_values='\\N',
                           keep_default_na=False,
                           low_memory=False
                           )
        print_generic_infos(data, filename, recording_file)
    else:
        data = pd.read_csv(file_path,
                           sep='\t',
                           header=0,
                           usecols=['tconst',
                                    'directors'],
                           dtype={'tconst': str,
                                  'directors': object},
                           na_values='\\N',
                           keep_default_na=False
                           )

        print_generic_infos(data, filename, recording_file)


def print_generic_infos(data, filename, recording_file):
    # print column type, missing values and some example data
    with open(recording_file, 'a') as rec_file:
        rec_file.write('----' + filename + '----\n\n')
        rec_file.write('Example data:\n')
        rec_file.write(str(data.head()) + '\n\n')
        rec_file.write('Dataframe shape:\n')
        rec_file.write(str(data.shape) + '\n\n')
        rec_file.write('Columns dtypes:\n')
        rec_file.write(str(data.dtypes) + '\n\n')

        if filename.endswith('.csv'):
            rec_file.write('Missing values count:\n')
            rec_file.write(str(data.isnull().sum()) + '\n\n')
        else:
            data.replace('\\N', np.nan, inplace=True)
            rec_file.write('Missing values count:\n')
            rec_file.write(str(data.isnull().sum()) + '\n\n')

        if filename == 'movies.csv':
            rec_file.write('Duplicated values count:\n')
            rec_file.write(str(data.duplicated(subset=['title']).sum()) + '\n\n')
        elif filename == 'ratings.csv':
            rec_file.write('Duplicated values count:\n')
            # consider a duplicate a user rating the same movie with the same rating more times
            rec_file.write(str(data.duplicated(subset=['userId', 'movieId', 'rating']).sum()) + '\n\n')
        elif filename == 'tags.csv':
            rec_file.write('Unique user tag values count:\n')
            rec_file.write(str(len(data['tag'].unique())) + '\n\n')
        else:
            rec_file.write('Duplicated values count:\n')
            rec_file.write(str(data.duplicated().sum()) + '\n\n')


def rating_dataframe_infos(data, recording_file, plots_folder):
    # plot rating distribution
    rating = data[['rating']]
    ax = sns.histplot(x='rating', data=rating, palette='deep', binwidth=0.2)
    ax.set(title='Ratings distribution')
    plt.tight_layout()
    plt.savefig(join(plots_folder, 'ratings_distributions.jpg'))
    plt.clf()

    # plot number of users per rating categories [0.5,5.0] distribution
    users_per_category = data.groupby('rating')['userId'].count()
    # Create a donut
    plt.title('Users per rating category')
    plt.tight_layout()
    colors = sns.color_palette('deep')[:10]
    plt.pie(x=users_per_category.values, colors=colors)

    # add a circle at the center to transform it in a donut chart
    hole = plt.Circle((0, 0), 0.7, color='white')
    p = plt.gcf()
    p.gca().add_artist(hole)
    plt.legend(loc='best', labels=users_per_category.index, fontsize='medium')
    plt.savefig(join(plots_folder, 'users_per_rating_category.jpg'))
    plt.clf()

    # plot number of ratings per movie
    ratings_per_movie = data.groupby('movieId')['rating'].count()
    ax = sns.kdeplot(data=ratings_per_movie, palette='deep')
    ax.set_title('Ratings per movie')
    ax.set(xlabel='movies', ylabel='count')
    plt.tight_layout()
    plt.savefig(join(plots_folder, 'movie_per_rating.jpg'))
    plt.clf()

    # get average/variance of rating for each movie
    mean_rating = data.groupby(by='movieId')['rating']
    mean = mean_rating.mean()
    median = mean_rating.median()
    var = mean_rating.var()
    mean_average = mean.mean()
    median_average = median.mean()
    var_average = var.mean()

    with open(recording_file, 'a') as rec_file:
        rec_file.write('\n')
        rec_file.write('Mean rating per film:\n')
        rec_file.write(str(mean.head()) + '\n\n')
        rec_file.write('Rating median per film:\n')
        rec_file.write(str(median.head()) + '\n\n')
        rec_file.write('Rating variance per film:\n')
        rec_file.write(str(var.head()) + '\n\n')
        rec_file.write('Mean average:\n')
        rec_file.write(str(mean_average) + '\n\n')
        rec_file.write('Median average:\n')
        rec_file.write(str(median_average) + '\n\n')
        rec_file.write('Variance average:\n')
        rec_file.write(str(var_average) + '\n\n')

    # plot number of users that has given a rating per movie
    users_per_movie = data.groupby('movieId')['userId'].count()
    ax = sns.kdeplot(data=users_per_movie, palette='deep')
    ax.set_title('Users per movie with rating')
    ax.set(xlabel='movies', ylabel='count')
    plt.tight_layout()
    plt.savefig(join(plots_folder, 'users_per_movies_with_rating.jpg'))
    plt.clf()


# function intended to plot additional information coupling dfs
def combine_dataframes(csvs, recording_file, plots_folder):
    # print missing data with some infos
    df_tags = pd.read_csv(get_dataframe(csvs, 'tags.csv'))
    missing_data = df_tags.loc[df_tags['tag'].isna()]
    df_movies = pd.read_csv(get_dataframe(csvs, 'movies.csv'))
    missing_data = pd.merge(missing_data, df_movies, on=['movieId'])

    with open(recording_file, 'a') as rec_file:
        rec_file.write('------------\n')
        rec_file.write('Movies without user tags:\n')
        rec_file.write(missing_data.to_string() + '\n\n')

    # plot genome tags associated to each movie
    def get_genome_tag_id(group):
        return np.argmax(group['relevance']) + 1

    df_genome_scores = pd.read_csv(get_dataframe(csvs, 'genome-scores.csv'))
    relevance_per_movie = df_genome_scores.groupby(by='movieId')[['tagId', 'relevance']].apply(get_genome_tag_id)
    relevance_per_movie = pd.DataFrame(relevance_per_movie, columns=['tagId'])

    # now each has associated the most likely genome-tag
    df_genome_tags = pd.read_csv(get_dataframe(csvs, 'genome-tags.csv'))
    relevance_per_movie = pd.merge(df_genome_tags, relevance_per_movie, on=['tagId'], how='right')
    movies_per_tag = relevance_per_movie.groupby('tag')['tagId'].count()

    with open(recording_file, 'a') as rec_file:
        rec_file.write('------------\n')
        rec_file.write('Number of movies per tag:\n')
        rec_file.write(movies_per_tag.to_string() + '\n\n')

    # plot for each tag category number of movie
    ax = sns.barplot(x='tag', y='tagId', data=relevance_per_movie.iloc[:25, :], palette='deep')
    ax.set_title('Genome tags')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(join(plots_folder, 'movies_per_genome_tag.jpg'))
    plt.clf()

    df_ratings = pd.read_csv(get_dataframe(csvs, 'ratings.csv'))
    mean_rating = df_ratings.groupby(by='movieId')['rating'].mean()
    mean_rating = pd.DataFrame({'movieId': mean_rating.index,
                                'rating': mean_rating.values})

    df_movies = df_movies[['movieId', 'genres']]
    df = pd.merge(df_movies, mean_rating, on=['movieId'], how='right')
    dummies_data = df['genres'].str.get_dummies('|')
    genre_ratings = [dummies_data[col].multiply(df['rating']) for col in dummies_data]
    mean_rating = [genre[genre > 0.0].mean() for genre in genre_ratings]
    mean_rating = pd.DataFrame({'genre': dummies_data.columns,
                                'mean_rating': mean_rating})

    # Create a donut
    plt.title('Mean rating per genre')
    plt.tight_layout()
    colors = sns.color_palette('deep')[:10]
    plt.pie(x=mean_rating['mean_rating'], colors=colors, labels=mean_rating['genre'])

    # add a circle at the center to transform it in a donut chart
    hole = plt.Circle((0, 0), 0.7, color='white')
    p = plt.gcf()
    p.gca().add_artist(hole)
    plt.savefig(join(plots_folder, 'mean_rating_per_genre.jpg'))
    plt.clf()

    # plot mean rating per tag
    relevance_per_movie = df_genome_scores.groupby(by='movieId')[['tagId', 'relevance']].apply(get_genome_tag_id)
    relevance_per_movie = pd.DataFrame({'movieId': relevance_per_movie.index,
                                        'tagId': relevance_per_movie.values})

    mean_rating = df_ratings.groupby(by='movieId')['rating'].mean()
    mean_rating = pd.DataFrame({'movieId': mean_rating.index,
                                'rating': mean_rating.values})

    df = pd.merge(relevance_per_movie, mean_rating, on=['movieId'], how='right')
    # rating movie ids are inconsistent with genome-score movie ids
    df = df.dropna()
    df['tagId'] = df['tagId'].astype(int)
    # now each has associated the most likely genome-tag
    df_genome_tags = pd.read_csv(get_dataframe(csvs, 'genome-tags.csv'))
    df = pd.merge(df_genome_tags, df, on=['tagId'], how='right')
    df = df[['tag', 'rating']]
    mean_rating_per_tag = df.groupby('tag')['rating'].mean()
    mean_rating_per_tag = pd.DataFrame({'tag': mean_rating_per_tag.index,
                                        'mean_rating': mean_rating_per_tag.values})

    with open(recording_file, 'a') as rec_file:
        rec_file.write('------------\n')
        rec_file.write('Mean rating per tag:\n')
        rec_file.write(mean_rating_per_tag.to_string() + '\n\n')

    # plot for each tag mean rating
    ax = sns.barplot(x='tag', y='mean_rating', data=mean_rating_per_tag.iloc[:25, :], palette='deep')
    ax.set_title('Mean rating per tags')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(join(plots_folder, 'mean_rating_per_tag.jpg'))
    plt.clf()


def plots_final_df_infos(df, plots_folder, recording_file):

    """
    Data Visualization before model definition, training and testing
    in order to observe that features doesn't follow gaussian distribution
    """

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Final dataframe shape:\n')
        rec_file.write(str(df.shape) + '\n\n')
        rec_file.write('Statistic information about continuous features (no tag):\n')
        cols = ['user_tags_count', 'rating', 'user_count', 'year', 'runtimeMinutes']
        rec_file.write(str(df[cols].describe().transpose()) + '\n\n')

    # rating values
    rating_values_count = df.groupby(by=['rating'])['movieId'].count()
    ax = sns.barplot(x=rating_values_count.index, y=rating_values_count.values, palette='deep')
    ax.set_title('Rating categories count')
    ax.set(xlabel='rating category', ylabel='count')
    plt.tight_layout()
    plt.savefig(join(plots_folder, 'rating_categories_count.jpg'))
    plt.clf()

    # movieId column is no more useful
    df.drop('movieId', axis=1, inplace=True)

    # user count
    ax = sns.histplot(x='user_tags_count', data=df, palette='deep')
    ax.set(title='Number of users distribution')
    plt.tight_layout()
    plt.savefig(join(plots_folder, 'user_count.jpg'))
    plt.clf()

    # runtimeMinutes distribution
    ax = sns.kdeplot(x='runtimeMinutes', data=df, palette='deep')
    ax.set_title('RuntimeMinutes distribution')
    plt.tight_layout()
    plt.savefig(join(plots_folder, 'runtime_minutes_distribution.jpg'))
    plt.clf()

    # year distribution
    ax = sns.kdeplot(x='year', data=df, palette='deep')
    ax.set(title='Years distribution')
    plt.tight_layout()
    plt.savefig(join(plots_folder, 'years_distribution.jpg'))
    plt.clf()

    return df


def high_dimensional_visualization(df, labels, file_plot_path):
    def tsne_plot(dataframe, scaling):
        for p in [7, 15, -15, 30]:
            if p > 0:
                tsne = TSNE(n_components=2, init='random', random_state=42, learning_rate='auto', perplexity=p)
            else:
                tsne = TSNE(n_components=2, init='random', random_state=42, learning_rate='auto', perplexity=-p,
                            metric='cosine', square_distances=True)
            tsne_results = tsne.fit_transform(dataframe)

            tsne_data = pd.DataFrame({'tagId': labels.values})
            tsne_data['tsne-2d-one'] = tsne_results[:, 0]
            tsne_data['tsne-2d-two'] = tsne_results[:, 1]

            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x='tsne-2d-one', y='tsne-2d-two',
                hue='tagId',
                palette='deep',
                data=tsne_data
            )
            if p > 0:
                plt.savefig(join(file_plot_path, f'_{scaling}_' + f'tsne_visualization_perplexity{p}.jpg'))
            else:
                plt.savefig(join(file_plot_path, f'_{scaling}_' + f'tsne_visualization_perplexity{-p}_cosine_.jpg'))
            plt.clf()

    print('\t\t\tno scaling')
    tsne_plot(df, 'no_scaling')

    print('\t\t\tstandard scaling')
    standard_scaler = StandardScaler()
    df_std_scaled = standard_scaler.fit_transform(df)

    tsne_plot(df_std_scaled, 'standard_scaling')

    print('\t\t\tmin-max scaling')
    min_max_scaler = MinMaxScaler()
    df_min_max_scaled = min_max_scaler.fit_transform(df)

    tsne_plot(df_min_max_scaled, 'min_max_scaling')


def plot_roc_multiclass(y_test, y_pred_proba, model_type, fold, plots_dir, is_training=False):

    """
    This function returns ROC AUC score calculated with a 'one-vs-rest'
    approach. If called during training and validation it returns only
    ROC AUC score, else, if called during performance measure stage, it
    will also plots roc curves for each class.
    Code adapted from:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """

    classes = [i for i in range(y_pred_proba.shape[1])]

    # binarize: [2] -> [0,0,1,0,0,0,0,0,0]
    y_test = label_binarize(y_test, classes=classes)
    n_classes = y_test.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
        # roc_auc_score(y_test, y_pred_proba, multi_class='ovr') is the same as sum all auc (fpr[i], tpr[i])
        # and divide them for n_classes
        roc_auc[i] = auc(fpr[i], tpr[i])

    # first aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # ROC AUC score calculated during training, no plots needed
    if is_training:
        return roc_auc["macro"]

    # Plot all ROC curves at test time
    colors = cycle(['red',
                    'slategrey',
                    'firebrick',
                    'aqua',
                    'olive',
                    'darkorange',
                    'darkviolet',
                    'cornflowerblue',
                    'darkslategrey',
                    'gold'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic multiclass {model_type}')
    plt.legend(loc='lower right')
    plt.savefig(join(plots_dir, f'roc_curve_{model_type}{fold}'))
    plt.clf()

    return roc_auc["macro"]


def plot_final_demo_results(results, plots_path, recording_file):
    train_results = results[['model', 'train_roc_auc_mean', 'train_roc_auc_std']]
    test_results = results[['model', 'mean_test_auc_score', 'std_test_auc_score']]

    print(train_results)
    with open(recording_file, 'a') as rec_file:
        rec_file.write(str(train_results) + '\n\n')

    print(test_results)
    with open(recording_file, 'a') as rec_file:
        rec_file.write(str(test_results) + '\n\n')

    # plot training ROC AUC score
    fig, ax = plt.subplots(1, 1)
    ax.errorbar(results['model'], results['train_roc_auc_mean'], yerr=results['train_roc_auc_std'], capsize=7,
                elinewidth=3,
                ecolor='red', label='training ROC AUC score')
    # remove errorbars from legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc='upper right', numpoints=1)
    plt.ylim(0, 1)
    plt.savefig(join(plots_path, 'final_training_results.jpg'))
    plt.clf()

    # plot test ROC AUC score
    fig, ax = plt.subplots(1, 1)
    ax.errorbar(results['model'], results['mean_test_auc_score'], yerr=results['std_test_auc_score'], capsize=7,
                elinewidth=3, ecolor='red', label='test ROC AUC score')

    # remove errorbars from legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc='upper right', numpoints=1)
    plt.ylim(0.5, 1)
    plt.savefig(join(plots_path, 'final_test_results.jpg'))
    plt.clf()
