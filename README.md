mldeveloper is a collection of methods to construct a basic workflow for a machine learning task in just 5 steps. One can read data from file, pre-process it, feature engineer and train the model. The objective is to simplify the whole process and create a abstraction through which, people with little programming experience can still create a workflow right from data reading to training a model thorugh as little programming as possible. It avoids the need to search and import different libraries and program the syntax. By calling a few functions with necessary information as input, the whole workflow can be implemented.

Note: This provides only basic functionalities and not all capabilities are available, as the intention is to facilitate basic and common ML workflows through very little coding.

Description of important steps and how to implement them right from reading a file to training a model is as follows:

(i) MLReader

Reads the files available in given filepath. Creates seperate train, test, validation dataframes if needed. Filepath must be provided. Other params can be ignored if not needed. Create an object with theis class and call get_all_df() or get_all_data().

#Parameters:
#filepath - string specifying the folder in which all the file to be read are present.
#testfile_list - list of strings specifying the names of the files in the folder which must considered as testing data. Seperate datarame is created
#for these and returned.
#valfile_list - list of strings specifying the names of the files in the folder which must considered as validation data. Seperate datarame is created
#for these and returned.
#col_train - List containing names of columns that must be read from the file.
#col_lag - Dict containing names of columns and its correspondin lag numeric as value of that key. The given column is considered and new column with 
#lagged values of the specified column is created. The name of lagge column ends with '(t-1)'. The amount of lag depends on provided value to that 
#particular column. Example: col_lag = {"col1":1}. Here a new column with name "col1(t-1)" is created and filled with lagged values of "col1" 
#with timestep 1.
#seed- seed for reproducibility.

 get_all_df() - Read files seperately into train, test and validation dataframes as specified by seperate list of file names for test and/or validation.Only needed if they need to be seperately provided
Returns: train , test and validation dataframes.

 get_all_data() - Read all the files which satisfy the extension in the given filepath into a dataframe.
 Returns: the dataframe with read content. 

(ii) MLFilter

Filters the given dataframe according to criteria. Criteria is specified as a dict with the column name as key and condition as well as value to be compared(tuple) as the value of that key. The available conditions that can be specified are 'geq,leq,gt,lt,eq,neq'. If filtering based on more than one coliumn is needed, it can just be added to dict as another key,value pair in the above mentioned fashion. The filtering of dataframe occurs one after the other( filtered based on first item and resulting df is filtered  based on second item and so on). Create an object with theis class and call do_filter().
Example: filter_dict = {'col1':['eq'],1}. This returns rows of dataframe which had the value of column 'col1' equal to 1. Using other condition and values, results in filtering according to that flag and value. 

 do_filter() - Parameters: 
#df - Dataframe to be filtered. Should be a dataframe.
#param- The dict containing the column name as key and condition as well as nueric as the value of that dict. Multiple such items can be added to to dict and passed into this parameter.
Returns: Filtered dataframe

(iii) MLPreprocess

Pre-process the data. Functions like one-hot encoding, removing unnecessary features, dropping rows containing missing values are done.To be intialized with seed. The object created can be used to call standard_preprocess() to process the data.

 standard_process() - Parameters:
#list_dfs - list containing the dataframes that needs pre-processing. If only one dataframe needs to be pre-processed, then list could contain only one dataframe.
#remove_cols_analysis - default False. If True, features with less than 'n' unique values in them, multi-collinear features ábove given threshold and features with standard deviation less than given threshold will be removed. Thresholds must be specified seperately using other parameters.
#unique_thresh - default 1. Features with no of unique values less than or equal to threshold is removed.
#multicoll_thresh - default 0.98. Degree of multi-collinearity allowed, above which one of the featurs is removed.
#std_thresh - default 0.1 - Minimum standard deviation in each column allowed, below which removed
#encode - default False. Flag to one-hot-encode categorical features or not. If ture and encode_cols not provided, columns types other than int and float, are automatically encoded.
#encode_cols - default None. list of column names to encode.
#dropna - default False. Falg to drop rows with missing values. 
Returns - list containing pre-processed dataframes, available in the order they were given in as input.

(iv) MLFeatureGenerate

Generate features before training. Initialize the object of the class with necessary parameters and call process_for_training. Create an object of this class with required parameters as input and use this object to call process_for_training().

Parameters:
#shuffle: defalut True. If False, no shuffling takes place during train test split.
#poly: default False. Flag to specify if polynomial fetures needs to calculated.
#poly_deg: default 2: Degree of polynomial features to be calculated. Uses scikit-learn. 
#scale: default False: Flag to specify if scaling is needed.
#scaler: default minmax(Min Max Scaling). Type of scaling. Other option is 'std' which is standard scaling. Uses scikit-learn.
#chunksize: default 1000. Chunk size for partial fit of scaler. Useful when dealing with large amounts of data and less memory.
#pca: default False: Flag to specify if Principal Component Analysis is need. Uses scikit-learn.
#n_components: default 0.98. Ratio of cumulative variance to be considered when taking the top n components using pca.
#train_ratio: default=0.8. Ratio of training data needed when splitting into train and test sets. Remaining data will be test data.

 process_for_training() - Parameters:
#x - Input features
#y - Input target

Returns: #x_train - Resulting train data fetures
#x_test - Test data features
#y_train - Resulting train data target
#y_test - Test data target
#poly_feat - Object with which poly feats were created. None if not used.
#scaler_feat - Object with which scaling was done.  None if not used.
#pca_feat  - Object with which pca was done. None if not used.

(v) MLModel

Train the model using an object created using this class with below mention input params. Call train_test() using the object to train and get score for the model.
Parameters:
#estimator - default 'linreg'. Type of algorithm to be used to create the model. Available types are 'linreg' for linear regression, 'logreg' for logistic regression,'rf_reg' for random forest regressor, 'rf_class' for random forest classifier.
#type_problem - default 'regression'. Type of problem. Available options are 'regression' or 'classification'.
#n_trees - default 10. No of trees to build if random forest model is selected.
#seed - default 5. For reproducibility

 train_test() - Parameters:
#x_train - Resulting train data fetures
#x_test - Test data features
#y_train - Resulting train data target
#y_test - Test data target

Returns: #estimator - estimator trained
#mae - Mean absolute error of trained model on test data in case of 'regression'
#r2score - r2score of trained model on test data in case of 'regression'
#accuracy - accuracy of trained model on test data in case of 'classification'
#confusion_matrix - confusion matrix of trained model on test data in case of 'classification'

(vi) Various other utility functions like:

drop_cols() - Parameters: dataframe and list of col names
Returns: Dataframe without the columns. Columns are dropped inplace.

split_features_targets() - Parameters: dataframe and target col name.
Returns: Splits the dataframe into features and target and returns both as list of values.

shuffle_df() - shuffle a dataframe given a dataframe as input.

shuffle_array() - shuffle an array given an array as input.

shuffle_arr_unison() - shuffle two arrays in unison maintaining the relationship given two arrays as input. Input must be array

train_test_split_df() - split given dataframes into train and test dataset according to given train_ratio. Inputs must be dataframes or series. Parameters - #X_df - df of features
#Y_df - df of target
#train_ratio - Ratio of training data used for splitting. Default 0.8.
#shuffle - Flag for shuffle. default True.
#seed - for reproducibility
Returns: train_x, test_x, train_y, test_y

train_test_split_arr() - split given arrays into train and test dataset according to given train_ratio. Inputs must be arrays. Parameters - #X - df of features
#Y - df of target
#train_ratio - Ratio of training data used for splitting. Default 0.8.
#shuffle - Flag for shuffle. default True.
#seed - for reproducibility
Returns: train_x, test_x, train_y, test_y
