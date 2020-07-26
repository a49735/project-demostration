# Project Demostration
The SMU quantitative finance school project codes are demostrated here. Although all the projects are required to be finished in groups, the uploaded projects are predominately finished by myself. The projects are closely related to industry practices hence the skills developed during the learning processes could be directly applied in industry.
## Quantitative trading strategies
This project is based on Quantopian platform, which has a robust backtesting system to assess the profitability of the strategies. The trading universe for the strategies are US equities. The main trading framework is factor investing but it is called quantmental as it uses company fundamental factors in a quantitative trading context. The factor selection started from the 65 mostly well-cited factors in academics, and finished with 23 factors (some are compressed into single composite factor), which all passed the Alphenlens(Quantopian builtin factor analysis method)robustness tests. The factors cover the pespectives of quality, profitability, value, efficiency, safety and behavioural economics to maximize the diversification benefits. Fundamental factors are combined linearly but momentum is also used but combined nonlinearly. Conditional on those stocks perform well on fundamental factors, the momentum factor is used to pick up stocks among those stocks with good fundamental scores. Volatility will be a negative sign to rank stocks but will become a postive sign when volatility factor portfolios exhibit  statistically significant return momentum. The treatments towards volatility and momentum are based on researches which indicate the regime switching properties of the two factors.

Based on similar principles, one long only strategy and other three long short equity strategies, momentum weighted, equal weighted and cosine similarity(correlation) weighted are developed. The best performing strategy, cosine similarity LSE, which weights factors based on factor correlation, achieved 12% annualized return with 0.94 Sharpe ratio after transaction fees during the 13-year backtesting period. Around 50% of returns of the strategy come from specific sources. 
## GARCH volatility modelling
The pricing of market options requires the input of volatility into Black-Scholes formula, thus the modelling of volatility have been great interests for option market participants.There are some empiricial evidences for GARCH models to capture volatility smile and price option with limited errors. The innovation for this proejct is to incorporate exogenous economic variables into the GARCH model to test if the extra variables could improve the GARCH models in option pricing. The challenge is that there is no ready-made python package able to handle extra exogenous inputs for standard GARCH models. Since there are 1 variables in mean equation and more than 4 variables in volatility equition needed to be estimated simultaneously, oridinary maximum likelihood using scipy optimization will usually be rendered unsuccessful. Through studying several source codes of econometrics package, the problems have been overcome by creating variance bounds, choosing meaningful initial sigma, selecting starting values smartly. The covariance matrix was also produced numercially based on BHHH estimator so that statistical properties of estimators could be infered. The robustness tests to examine the stability of unconditional variance among subsamples has proved the estimation procedure is successful to a certain extent.

The option pricing procedure is using Monte Carlo simulation with Empirical martingale simulation as a control variate technique, the out of sample estimated prices is then compared with market prices and the errors are measured with mean squared errors. Trading volumes, US treasury 10 year yield, foreign exchange rate,crude oil, exchange rate volatility and gold futures have been tested and results show no significant evidences that those economic variables could improve GARCH in option pricing.
## Implied binomial Tree pricer
The intention of the construction of the implied binomial tree pricer is to accommodate volatility smiles observed in the market. The implied binomial tree is extracted from the smile so that it can be used to price any illiquid asset classess without losing consistancy from the market prices, the process to construct the tree follows the steps from "The volatility smile and its implied tree" by Derman and Kanji. For each node on each step, a option price will be given by the Black Scholes pricer with the volatility extracted from volatility surface. The arrow-debreu state price can be used to serve the purpose as the probability density function of the underlying, as a result, equations can be established and coresponding node values can be calculated. The tree construction follows the principles of CRR binomial tree that is recombining. Every even time step will have a central node value equal to S0 and on every odd time step the central nodes will follow the rule of $S_0^2=S_{k,k//2} \times S_{k,k//2+1}$. The implied binomial tree could be a powerful tool to price non-actively traded derivatives, especially for barrier options whose probability of striking the barrier is sensitive to the shape of the smile. The barrier options pricing and knock in - knock out parity with the implied binomial tree with smile are tested, the results show the model value barrier options appropriately and KIKO parity is not violated.
## Fixed income
