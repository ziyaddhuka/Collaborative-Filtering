import time
start = time.time()
import pandas as pd
import numpy as np
from numpy import linalg as la
from sklearn.metrics import mean_squared_error
import sys

if __name__ == '__main__':
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
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

    # Multiplying the weight matrix with zero centered matrix
    matrix_calculated = np.matmul(pivoted_df_mat_numpy_centered_to_zero.T, weight_mat)

    # calculating kappa value
    wt_sum = np.absolute(weight_mat).sum(axis=0)
    wt_sum_inverse = np.reciprocal(wt_sum, out=np.zeros_like(wt_sum), where=wt_sum!=0)

    # dividing the kappa value with the matrix
    matrix_calculated = (matrix_calculated * wt_sum_inverse)

    # Adding average user rating to the calculated matrix
    matrix_calculated = matrix_calculated + average_user_rating_only_rated_movies.reshape(1,-1)

    # Loading the test data
    test_data = pd.read_csv(test_data_path, header=None)

    test_data.columns = ['movie_id', 'user_id', 'rating']

    pivoted_df_mat_test = pd.pivot_table(test_data, values='rating', index='user_id', columns='movie_id').fillna(0)

    test_d = pd.pivot_table(test_data, values='rating', index='user_id', columns='movie_id').fillna(0)

    # filling the pivot values of train dataframe with the calculated ratings
    pivoted_df_mat[:] = matrix_calculated.T

    # filtering index of the train data with the test data indexes
    filter_index = test_d.index.values
    pivoted_df_mat = pivoted_df_mat[pivoted_df_mat.index.isin(filter_index)]

    filter_columns = test_d.columns.values
    pivoted_df_mat = pivoted_df_mat.loc[:, pivoted_df_mat.columns.isin(filter_columns)]


    test_d = test_d.to_numpy()
    pivoted_df_mat = pivoted_df_mat.to_numpy()

    # Loop to store prediction and corresponding true value
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
