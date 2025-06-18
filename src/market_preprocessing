#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
import os
import regex as re
from scipy.stats import norm
from scipy.optimize import brentq


# In[2]:


#Data Download


# In[2]:


df=pd.read_csv("/Users/apple/Desktop/bam63i7gq8vszgdl.csv")
sp500=pd.read_csv("/Users/apple/Desktop/SP500.csv")


# In[3]:


sp500.rename(columns={"observation_date": "date"}, inplace=True)


# In[4]:


merged_df = pd.merge(df, sp500, on='date', how='left')


# In[5]:


df=merged_df


# In[6]:


df["strike_price"]=df["strike_price"]/1000


# In[7]:


df['date'] = pd.to_datetime(df['date'])
df['exdate'] = pd.to_datetime(df['exdate'])
df['DTE'] = (df['exdate'] - df['date']).dt.days
df = df[df['DTE'] >= 0].copy()


# In[8]:


call=df[df["cp_flag"]=="C"]
put=df[df["cp_flag"]=="P"]
put.drop("forward_price",axis=1,inplace=True)
call.drop("forward_price",axis=1,inplace=True)
call.dropna(inplace=True)
put.dropna(inplace=True)
call= call[~(call['symbol'].str.startswith('SPXW')) ].copy()
put= put[~(put['symbol'].str.startswith('SPXW')) ].copy()


# In[9]:


N = 10000  
daily_sampled_list = []
for current_date in df['date'].unique():
    daily_df = df[df['date'] == current_date].copy()
    puts_df = daily_df[
        (daily_df['cp_flag'] == 'P') &
        (daily_df['best_offer'] > 0) &
        (daily_df['strike_price'] > 0) &
        (daily_df['SP500'] > 0) &
        (daily_df['DTE'] <= 120)
    ].copy()

    if puts_df.empty:
        continue
    puts_df["moneyness"] = puts_df["SP500"] / puts_df["strike_price"]
    puts_df["log_moneyness"] = np.log(puts_df["moneyness"])
    dte_bins = [0, 30, 90,120]
    dte_labels = ['0-30D', '1-3M',">3M"]
    puts_df['DTE_bucket'] = pd.cut(puts_df['DTE'], bins=dte_bins, labels=dte_labels, right=True)

    mny_bins = [0, 0.7, 0.9, 1.0, 1.1, 1.3, np.inf]
    mny_labels = ['Deep ITM', 'ITM', 'ATM-ish', 'OTM', 'Far OTM', 'Extreme OTM']
    puts_df['moneyness_bucket'] = pd.cut(puts_df['moneyness'], bins=mny_bins, labels=mny_labels, right=True)

    valid_mny_buckets = ["Deep ITM",'ITM', 'ATM-ish',"OTM"]
    puts_df = puts_df[puts_df['moneyness_bucket'].isin(valid_mny_buckets)]
    sampled = (
        puts_df
        .groupby(['DTE_bucket', 'moneyness_bucket'], group_keys=False)
        .apply(lambda x: x.sample(n=min(N, len(x))))
    )

    sampled['date'] = current_date 
    daily_sampled_list.append(sampled)
sampled_options_all_days_put = pd.concat(daily_sampled_list).reset_index(drop=True)


# In[10]:


N = 10000  
daily_sampled_list = []
for current_date in df['date'].unique():
    daily_df = df[df['date'] == current_date].copy()
    puts_df = daily_df[
        (daily_df['cp_flag'] == 'C') &
        (daily_df['best_offer'] > 0) &
        (daily_df['strike_price'] > 0) &
        (daily_df['SP500'] > 0) &
        (daily_df['DTE'] <= 120)
    ].copy()

    if puts_df.empty:
        continue
    puts_df["moneyness"] = puts_df["SP500"] / puts_df["strike_price"]
    puts_df["log_moneyness"] = np.log(puts_df["moneyness"])
    dte_bins = [0, 30, 90,120]
    dte_labels = ['0-30D', '1-3M',">3M"]
    puts_df['DTE_bucket'] = pd.cut(puts_df['DTE'], bins=dte_bins, labels=dte_labels, right=True)

    mny_bins = [0, 0.7, 0.9, 1.0, 1.1, 1.3, np.inf]
    mny_labels = ['Deep ITM', 'ITM', 'ATM-ish', 'OTM', 'Far OTM', 'Extreme OTM']
    puts_df['moneyness_bucket'] = pd.cut(puts_df['moneyness'], bins=mny_bins, labels=mny_labels, right=True)

    valid_mny_buckets = ["Deep ITM",'ITM', 'ATM-ish',"OTM"]
    puts_df = puts_df[puts_df['moneyness_bucket'].isin(valid_mny_buckets)]
    sampled = (
        puts_df
        .groupby(['DTE_bucket', 'moneyness_bucket'], group_keys=False)
        .apply(lambda x: x.sample(n=min(N, len(x))))
    )

    sampled['date'] = current_date 
    daily_sampled_list.append(sampled)
sampled_options_all_days_call = pd.concat(daily_sampled_list).reset_index(drop=True)
