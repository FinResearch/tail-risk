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

Thirdly, we choose the type of return. In our script, we have defined three types of returns. The log return is defined as log(P(t+δ)/P(t), where (P)  is the price at the time (t) and δ is the time scale. We can perform the tail analysis on different time scales, where δt=1 denotes daily log-returns, δt=5 weekly log-returns and δt= 22 monthly log-returns, for example. The relative return (second option) is defined as (P(t+δ)/P(t))-1. The term return (third option) defines price increments P(t+δ) - P(t). 

Forthly, ...

Fithly, ... 

Sixthly, ... 

Seventhly, we define the amplitude of the sliding time window. This option only applies to the rolling and the increasing time window, but it does not apply to the static analysis. In case the sliding time window is equivalent to one day, the tail statistics are calculated every day from the initial to the final date. In case the sliding time window is five days, the tail statistics are calculated on every fifth day (or only once a week). In simple words, if the difference between the final and initial date is 500 days, and the sliding time window is five days, we calculate 100 tail statistics on every fifth day.  



