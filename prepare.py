import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates #to format dates on our plots


def prep(df, user):
    '''
    This function takes the id of a particular user and returns the webpages
    that user viewed.
    '''
    df = df[df.user_id == user]
    df.date = pd.to_datetime(df.date)
    df = df.set_index(df.date)
    pages = df['page_viewed'].resample('d').count()
    return pages

def compute_pct_b(pages, span, weight, user):
    '''
    This function computes the %b of the webpages given the span and weight
    and returns a dataframe.
    '''
    midband = pages.ewm(span=span).mean()
    stdev = pages.ewm(span=span).std()
    ub = midband + stdev * weight
    lb = midband - stdev * weight
    bb = pd.concat([ub, lb], axis=1)
    my_df = pd.concat([pages, midband, bb], axis=1)
    my_df.columns = ['pages', 'midband', 'ub', 'lb']
    my_df['pct_b'] = (my_df['pages'] - my_df['lb'])/(my_df['ub'] - my_df['lb'])
    my_df['user_id'] = user
    return my_df

def plt_bands(my_df, user):
    '''
    This function takes a dataframe and the user id and plots
    the Bollinger bands
    '''
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(my_df.index, my_df.pages, label='Number of Pages, User: '+str(user))
    ax.plot(my_df.index, my_df.midband, label = 'EMA/midband')
    ax.plot(my_df.index, my_df.ub, label = 'Upper Band')
    ax.plot(my_df.index, my_df.lb, label = 'Lower Band')
    ax.legend(loc='best')
    ax.set_ylabel('Number of Pages')
    plt.show()
    
def find_anomalies(df, user, span, weight):
    '''
    This function takes a data frame and a user id, then searches for
    those values where %b is greater than one (meaning they mark
    anomalies).
    '''
    pages = prep(df, user)
    my_df = compute_pct_b(pages, span, weight, user)
#     plt_bands(my_df, user)
    return my_df[my_df.pct_b > 1]