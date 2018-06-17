import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

path = 'C://Users//LENOVO//Desktop//Help_Me!//neobe_code_challenge.csv'


# --------------------------------------------------------------------------------------#
#----------------------------------Function definitions---------------------------------#
# --------------------------------------------------------------------------------------#

""" 
    Defines list of functions to be used for Preprocessing and model building and final prediction.
"""

"""
    function to return dataframe for the file provided.
"""
    
    
def read_file(path):
    return pd.read_csv(path)
    
""""
    Function to drop timestamp feature and also drop the rows
having target label as zero.
"""

    
def drop_attributes(dataframe):
    dataframe = dataframe.drop(['timestamp'], axis=1)
    dataframe = dataframe[dataframe['expected_result']!=0]
    dataframe.reset_index(inplace=True)
    return dataframe

"""
    Function to get the respective statistics{mean, max, min, median}
of each element(list={each list of column matrix}) of the matrix
column(input_attribute)
"""

    
def matrix_conv(input_attribute, apply_func):
    input_attribute = input_attribute.matrix
    input_split = input_attribute.split('],')
    result = []
    for element in input_split:
        input_replace = element.replace('[', '')
        input_change = input_replace.replace(']', '')
        input_update = input_change.split(',')
        input_modified = np.array([float(x) for x in input_update])
        result.append(apply_func(input_modified))
    return result
    
"""
    Create a single list containing mean of each dimension (total 27)
and then segregate them into individuals columns
"""


def add_stats_col(apply_func, col_initials, dataframe,
                  matrix_conv=matrix_conv):
    
    dataframe[col_initials] = dataframe.apply(matrix_conv, axis=1,
                                              apply_func=apply_func)

    col_names = [col_initials + str(idx) for idx in range(1, 28)]

    return pd.DataFrame(dataframe[col_initials].values.tolist(),
                        columns=col_names)

"""
    Function to drop all the derived column having very low variance
(~0.1) between its mean, median, max and min.
"""

    
def get_drop_column(column_name, drop_list_var):
    drop_col = []
    for name in column_name:
        for indx in drop_list_var:
            drop_col.append(name+str(indx))
    return (drop_col)

"""
    Function to split the dataframe based on the target class(positive or
negative and return the feature and target for each class.
"""

    
def split_user(dataframe, user_id):
    df_user = dataframe[dataframe['user_id'] == user_id]
    user_pos = df_user[df_user['expected_result'] == 1]
    user_neg = df_user[df_user['expected_result'] == -1]
    x_train, y_train = user_pos.iloc[:, :-2], user_pos['expected_result']
    x_test, y_test = user_neg.iloc[:, :-2], user_neg['expected_result']
    return x_train, x_test, y_train, y_test

"""
   Function to apply scaling operation on the input features and return
the transformed scaled features.
"""


def scaler_func(x_train, x_test):
    col = [x_train.columns]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_train)
    return (pd.DataFrame(scaler.transform(x_train), columns=col),
            pd.DataFrame(scaler.transform(x_test), columns=col))


"""
    Function to fit OneClassSVM on the positive class training set and
evaluate the results on negative class testing set and return the
calculated metrics.
Note - gamma values is choosen after hyper parameter tunning.
"""


def svm_class(user_id, x_train, x_test, y_train, y_test, gamma=0.71):
    clf = svm.OneClassSVM(nu=0.15, kernel="rbf", gamma=gamma)
    clf.fit(x_train)
    y_pred_pos = clf.predict(x_train)
    y_pred_neg = clf.predict(x_test)
    n_error_train = y_pred_pos[y_pred_pos == -1].size
    n_error_test = y_pred_neg[y_pred_neg == 1].size
    fap = float(n_error_test*100/y_test.size)
    frp = float(n_error_train*100/y_train.size)
    return ([user_id, y_train.size, n_error_train,
            y_test.size, n_error_test, fap, frp])

"""
    Function to call scaler function and pass on the scaled feature
to fit into OneClassSVM and fetch the metric for each user.
"""

    
def get_metric(data_frame, user_ids):
    col_name = ['user_id', 'Train size', 'Train Error', 'Test size',
                'Test Error', 'False Acceptance %', 'False Rejection %']
    data_metric = pd.DataFrame()
    metric_list = []
    for user_id in user_ids:
        x_train, x_test, y_train, y_test = split_user(data_frame, user_id)
        scaled_train, scaled_test = scaler_func(x_train, x_test)
        metric_list.append(svm_class(user_id, scaled_train, scaled_test,
                                     y_train, y_test))

    return pd.DataFrame(metric_list, columns=col_name)


# --------------------------------------------------------------------------------------#
#---------------------------End of Function definition----------------------------------#
# --------------------------------------------------------------------------------------#

"""
    Create a temporary dataframe to capture the variance of each calculated
columns (27 * 4 {4 statistics - Mean, Median, Max, Min})
"""
'''
    Read the input file
'''
data = read_file(path)


'''
    Drop the timestamp and rows having labels as zero
'''
data = drop_attributes(data)


'''
   Extract the mean, median, max and min of 27 features(Matrix feature) and 
club the same together 
'''
data_mean = add_stats_col(np.mean, 'Array_mean', data)
data_min = add_stats_col(np.min, 'Array_min', data)
data_max = add_stats_col(np.max, 'Array_max', data)
data_median = add_stats_col(np.median, 'Array_median', data)

'''
   extract the variance of each derieved columns(above) 
'''
index_value = [idx for idx in range(1, 28)]
temp_mean = pd.DataFrame(data_mean.apply(np.var, axis=0), columns=['mean'])
temp_min = pd.DataFrame(data_min.apply(np.var, axis=0), columns=['min'])
temp_max = pd.DataFrame(data_max.apply(np.var, axis=0), columns=['max'])
temp_median = pd.DataFrame(data_median.apply(np.var, axis=0),
                           columns=['median'])
temp_mean.index = temp_median.index = temp_min.index \
                = temp_max.index = index_value

template = pd.concat([temp_mean['mean'],temp_median['median'],
                      temp_min['min'],temp_max['max']],axis=1, ignore_index=False)


'''
   Create the list from Original 27 features having very less variance (<0.1) \
between mean, median, min and max of respective features. 
'''
drop_list_var = template[template.var(axis=1) < 0.1].index.tolist()
column_name = ['Array_min','Array_max','Array_mean','Array_median']


'''
    To check if there exist any collinearity {check specifically for mean}.
'''
corr_data = data_mean.corr()
corr_data[corr_data[corr_data.columns.values]>  0.8]
corr_data[corr_data[corr_data.columns.values]< -0.8]

'''
    Conclusion - Yes but those specified columns are already getting
dropped due to low variance across column statistics.
'''


'''
   Drop all the columns having very low variance or exhibiting multi-collinearity
and append user id and label to remainder dataframe
'''
final_drop_col = get_drop_column(column_name, drop_list_var)
df_club = pd.concat([data_mean, data_median, data_min, data_max],axis=1, ignore_index=False)
df_club = df_club.drop(final_drop_col, axis=1)
df_club = pd.concat([df_club,data[['expected_result', 'user_id']]] , axis=1)
user_ids = np.array(np.unique(data.user_id)).astype(int)

'''
Call the function which will do below steps for each user:-
Extract the data for specified user
split the data into two based on positive and negative labels (1, -1)
perform feature scaling (standardization) and feature transformation
train the complete positive instance on OneClassSVM
predict the response on both negative and positive instance
collect the error metrics
    
'''
print (get_metric(df_club, user_ids))