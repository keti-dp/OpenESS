#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

def Separate_Date(df=None, date=None, interval=60):
    
    """
    df = DataFrame
    date = form of '0601'
    interval --> time interval (in seconds).
                    If the whole seconds of some specific date < interval
                        then, there is possibility can't catch that date
                    At that time, make interval small
    """
    
    
    TIME = df.loc[:,'TIMESTAMP']
    ind_min, ind_max = 0, 0
    for i in range( df.index[0], max(df.index), interval ) :
        try:
            if date == TIME[i].date().__str__()[-5:-3] + TIME[i].date().__str__()[-2:] :
                ind_min = i
                break
        except:
            continue

    while True:
        try:
            if date == TIME[ind_min].date().__str__()[-5:-3] + TIME[ind_min].date().__str__()[-2:] :
                ind_min -= 1
            else:
                ind_min += 1
                break
        except:
            ind_min +=1
            break
    
    ind_max = ind_min + 24*60*60 + 30
    while True:
        try:
            if date != TIME[ind_max].date().__str__()[-5:-3] + TIME[ind_max].date().__str__()[-2:] :
                ind_max -= 1
            else:
                break
        except:
            ind_max -=1

    return df.loc[ind_min:ind_max]



def Separate_DF(df):
    TIME = df.loc[:,'TIMESTAMP']
    # Get unique raw dates of data
    dates = []
    for i in range(0,len(TIME),60) :
        try:
            #print(i)
            dates.append( TIME[i].date().__str__()[-5:-3] + TIME[i].date().__str__()[-2:] )
        except:
            continue
    
    dates = list(set(dates))

    return { date : Separate_Date(df,date) for date in dates } , dates

