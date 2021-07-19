def acquire(file_name, column_names):
    '''
    This function is designed to read a csv using a seperator and build a dataframe 
    from a specific list of columns specified by the user.
    '''
    return pd.read_csv(file_name, sep="\s", header=None, names=column_names, usecols=[0, 2, 3, 4, 5])
