import time
start = time.time()
import pandas as pd
import numpy as np
from numpy import linalg as la
from sklearn.metrics import mean_squared_error
import sys

if __name__ == '__main__':
    n = int(sys.argv[1])
    train_data_path = sys.argv[2]
    test_data_path = sys.argv[3]
    # Loading the data
    data = pd.read_csv(train_data_path, header=None)
    data.columns = ['movie_id', 'user_id', 'rating']

    # using pivot to convert the data to rows of users and columns of movies
    pivoted_df_mat = pd.pivot_table(data, values='rating', index='user_id', columns='movie_id').fillna(0)
    pivoted_df_mat_numpy = pivoted_df_mat.copy().to_numpy()

    # calculating average user rating of a particular user
    average_user_rating_only_rated_movies = pivoted_df_mat_numpy.sum(axis=1) / (pivoted_df_mat_numpy > 0).sum(axis=1)
    pivoted_df_mat_numpy_centered_to_zero = pivoted_df_mat_numpy.copy().astype('float32')

    # subtracting average rating with non zero ratings
    pivoted_df_mat_numpy_centered_to_zero = np.subtract(pivoted_df_mat_numpy,average_user_rating_only_rated_movies.reshape(-1,1), out=pivoted_df_mat_numpy_centered_to_zero,  where= pivoted_df_mat_numpy_centered_to_zero>0)

    # Calculating the weight matrix
    # 1. Multuplying pivoted_df_mat_numpy_centered_to_zero with itself to get the numerator of the pearsons coefficient

    weight_mat = np.matmul(pivoted_df_mat_numpy_centered_to_zero, pivoted_df_mat_numpy_centered_to_zero.T)

    # Calculating the norm i.e. denominator
    norms = np.linalg.norm(pivoted_df_mat_numpy_centered_to_zero, ord=2 , axis=1).astype('float32')
    norm_mat = np.matmul(norms.reshape(-1,1), norms.reshape(1,-1))
    # Once the norm is calculated we divide the numerator with the denominator
    weight_mat = np.divide(weight_mat, norm_mat, out=np.zeros_like(norm_mat), where=norm_mat!=0)

    mat_f = np.zeros_like(pivoted_df_mat_numpy_centered_to_zero)

    # Looping through the users and movies and storing the predicted movie using
    for i,(user_avg,current_wt) in enumerate(zip(average_user_rating_only_rated_movies, weight_mat)):
        top_n_similar = np.argpartition(-current_wt, n)[:n]
        for j, movie_col in enumerate(pivoted_df_mat_numpy_centered_to_zero.T):
            # finding users having non-zero ratings for a movie
            avg_ar_zero = movie_col[top_n_similar].T
            weight_m = current_wt[top_n_similar]
            su_m = np.sum(np.abs(weight_m))
            # if the absolute sum is zero keep kappa as zero
            if su_m == 0:
                kappa = 0
            else:
                kappa = 1 / su_m

            # current movie rating using the formula
            current_movie_rating = user_avg + (kappa * np.dot(avg_ar_zero,weight_m))
            mat_f[i,j] = current_movie_rating

    # Loading the test data
    test_data = pd.read_csv(test_data_path, header=None)
    test_data.columns = ['movie_id', 'user_id', 'rating']
    pivoted_df_mat_test = pd.pivot_table(test_data, values='rating', index='user_id', columns='movie_id').fillna(0)
    test_d = pd.pivot_table(test_data, values='rating', index='user_id', columns='movie_id').fillna(0)
    pivoted_df_mat[:] = mat_f.copy()

    filter_index = test_d.index.values
    pivoted_df_mat = pivoted_df_mat[pivoted_df_mat.index.isin(filter_index)]

    filter_columns = test_d.columns.values
    pivoted_df_mat = pivoted_df_mat.loc[:, pivoted_df_mat.columns.isin(filter_columns)]

    test_d = test_d.to_numpy()
    pivoted_df_mat = pivoted_df_mat.to_numpy()

    actual = []
    predicted = []
    for row_test, row_train in zip(test_d,pivoted_df_mat):
        idx = np.where(row_test != 0)[0]
        actual.extend(list(row_test[idx]))
        predicted.extend(list(row_train[idx]))

    # Calculating RMSE
    rmse = mean_squared_error(actual, predicted, squared=False)
    # Calculating MAE
    mae = np.mean(np.absolute(np.array(actual) - np.array(predicted)))
    print("RMSE  = ", rmse)
    print("MAE = ", mae)
    print("Time taken  = {} seconds ".format(time.time()-start))
