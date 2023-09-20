from portfolio import Portfolio
from optimization import OptimizationP

import numpy as np

rf = 0.0037 #risk-free rate
names = ['SBER.ME','GAZP.ME', 'MTSS.ME', 'MGNT.ME', 'SNGS.ME', 'HYDR.ME', 'AFLT.ME', 'DSKY.ME', 'AAPL', 'IVV']
dates = ('2018-01-01', '2021-02-01')
n = len(names)

portfolio_opt = OptimizationP(names, dates)

## Possible Options:
##    1. maxReturn - maximize return of the portfolio
##    2. minRisk - minimize risk of the portfolio 
##    3. maxSharpe - maximize a Sharpe ratio
## to view the same run: 
# portfolio_opt.get_help() 

portfolio_opt.get_ts_csv()
portfolio_opt.resample_to_months()

## to view correlation matrix run:
# portfolio_opt.corr_matrix() 

p_return = portfolio_opt.get_month_return()
p_cov = portfolio_opt.get_month_COV()


#########################################################################################################

print("<---------Equal Allocation--------->\n")
print(f"Portfolio Weights: {portfolio_opt.w}")
print(f'Return: {round(portfolio_opt.get_returnP()*100, 3)} %\n')
print(f'Risk: {round(portfolio_opt.get_riskP()*100, 3)} %\n')
print("<---------------------------------->\n")


print("<---------Maximize Return--------->\n")
max_return = portfolio_opt.optimizeP('maxReturn', (0, 1), 0, rf)
print(max_return)

max_return_ret = round(np.dot(p_return, max_return.x), 3)*100
max_return_risk = round(np.dot(max_return.x, np.dot(p_cov, max_return.x)), 3)*100

print(f'Return: {round(max_return_ret, 3)} %\n')
print(f'Risk: {round(max_return_risk, 3)} %\n')
print("<---------------------------------->\n")


print("<---------Maximize Sharpe Ratio--------->\n")
max_sharpe = portfolio_opt.optimizeP('maxSharpe', (0, 1), 0.01, rf)
print(max_sharpe)

max_sharpe_ret = round(np.dot(p_return, max_sharpe.x)*100, 3)
max_sharpe_risk = round(np.dot(np.dot(max_sharpe.x, p_cov), max_sharpe.x)*100, 3)
max_sharpe_ratio = -max_sharpe.fun

print(f'Return: {round(portfolio_opt.get_returnP()*100, 3)} %\n')
print(f'Risk: {round(portfolio_opt.get_riskP()*100, 3)} %\n')
print(f'Sharpe ratio: {round(max_sharpe_ratio, 3)}\n')
print("<---------------------------------->\n")




print("<---------Minimize Risk--------->\n")
min_risk = portfolio_opt.optimizeP('minRisk', (0, 1), 0.01, rf)
print(min_risk)

min_risk_ret = round(np.dot(p_return, min_risk.x)*100, 3)
min_risk_risk = round(np.dot(np.dot(min_risk.x, p_cov), min_risk.x)*100, 3)

print(f'Return: {round(min_risk_ret, 3)} %\n')
print(f'Risk: {min_risk_risk} %\n')
print("<---------------------------------->\n")



