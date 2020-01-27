# Tail Risk

## Motivation

## Description

## Theory

## Methodology

## Script: Tail Analysis

## Script: Cross Sectional Tail Analysis

In this section, we explain the cross-sectional tail analysis script. 

Firstly, we select the name of the database. In our test case, we use a sample database in xlsx format (Excel). Other formats are also supported but require to change the function pd.read_excel(filename). If the database is a comma-separated values (csv) file, the statement "read_csv" is used. 

Secondly, we select the number of cross-sectional time series and the corresponding ticker symbols. We choose the initial and final date as well as the rolling lookback window in days. Assume that the number of tickers is 10, the initial date is 01.01.2009, and the lookback is 252 days. Starting on 01.01.2009, we apply a daily rolling backwards-looking time window of 252 days, which is equivalent to two trading years. The returns are calculated for each time series separately. Next, the return time series are merged accordingly to the cross-sectional criteria (discussed below). In our example,  the ten individual time series have the same grouping criteria (country code: DE). On 01.01.2009, the cross-sectional return series has a length of 2520 observations. 

Thirdly, we choose the type of return. In our script, we have defined three types of returns. The log return is defined as log(P(t+δ)/P(t), where (P)  is the price at the time (t) and δ is the time scale. We can perform the tail analysis on different time scales, where δt=1 denotes daily log-returns, δt=5 weekly log-returns and δt= 22 monthly log-returns, for example. 


