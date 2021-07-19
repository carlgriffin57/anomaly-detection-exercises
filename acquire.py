def acquire(file_name, column_names):
    '''
    This function reads in a csv file and certain column names within that file.
    '''
    return pd.read_csv(file_name, sep="\s", header=None, names=column_names, usecols=[0, 2, 3, 4, 5])
