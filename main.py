##To parse real time data from Yahoo Finance(small dataset)

##import pandas as pd
##import requests
##from bs4 import BeautifulSoup
##url = 'https://finance.yahoo.com/quote/FB/history?p=FB' 
##df = pd.read_html(url)[0]
##df.drop(df.tail(1).index,inplace=True)
##df.rename(columns = {'Adj Close**':'Adj. Close'}, inplace = True)
##print(df.head())

##To use existing local dataset

##import pandas as pd
##data =  "./DATA/GOOGL.csv"
##df = pd.read_csv(data, sep = ',')
##df.rename(columns = {'Adj Close':'Adj. Close'}, inplace = True)
##print(df.head())

##To use dataset from quandl module
import quandl
import numpy as np
df = quandl.get("WIKI/AMZN")
print(df.head())
