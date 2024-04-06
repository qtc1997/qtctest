#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:10:01 2024

@author: qiantianci
"""

# test1
import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, q, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes option price.

    Parameters:
    S : float
        Current stock price.
    K : float
        Option strike price.
    T : float
        Time to expiration (in years).
    r : float
        Risk-free interest rate (annualized).
    q : float
        股票的连续分红率 有分红为正
        如果是期货市场 贴水为负 升水为正 
        例如贴水率年化5%  则q = -0.05
    sigma : float
        Volatility of the underlying stock (annualized).
    option_type : str, optional
        Type of option, either 'call' or 'put' (default is 'call').

    Returns:
    float
        Option price.
    """
    d1 = (np.log(S / K) + ( r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) -  S * np.exp(-q * T)  * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Please choose either 'call' or 'put'.")

    return option_price

# Example usage:
if __name__ == "__main__":
    S = 100    # Current stock price
    K = 100    # Strike price
    T = 1      # Time to expiration (in years)
    q = -0.02
    r = 0.03   # Risk-free interest rate
    #r + q = mu = b
    sigma = 0.2  # Volatility

    call_price = black_scholes(S, K, T,q, r, sigma, option_type='call')
    put_price = black_scholes(S, K, T,q, r, sigma, option_type='put')

    print("Black-Scholes Call Price:", call_price)
    print("Black-Scholes Put Price:", put_price)


import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_stock_price(S0, mu, sigma, T, num_simulations, num_steps):
    """
    Generate stock price paths using Monte Carlo simulation.

    Parameters:
    S0 : float
        Initial stock price.
    mu : float
        Annualized risk-free interest rate.
    sigma : float
        Annualized volatility of the stock.
    T : float
        Time to maturity (in years).
    num_simulations : int
        Number of simulations.
    num_steps : int
        Number of time steps.

    Returns:
    ndarray
        Array of stock price paths with shape (num_simulations, num_steps+1).
    """
    dt = T / num_steps
    paths = np.zeros((num_simulations, num_steps + 1))
    paths[:, 0] = S0

    for i in range(1, num_steps + 1):
        z = np.random.normal(0, 1, num_simulations)
        paths[:, i] = paths[:, i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

    return paths

def monte_carlo_option_price(stock_paths,K,r,T,option_type):
    stock_prices = stock_paths[:,-1]
 # Calculate option payoff
    if option_type == 'call':
        option_payoff = np.maximum(stock_prices - K, 0)
    elif option_type == 'put':
        option_payoff = np.maximum(K - stock_prices, 0)
    else:
        raise ValueError("Invalid option type. Please choose either 'call' or 'put'.")

    # Discounted expected payoff
    option_price = np.exp(-r * T) * np.mean(option_payoff)

    return option_price
# Example usage:
if __name__ == "__main__":
    S0 = 100     # Initial stock price
    K = 100
    mu = 0.05    # draft rate
    r =  0.03    # Risk-free interest rate
    sigma = 0.2  # Annualized volatility
    T = 1        # Time to maturity (in years)
    num_simulations = 100000
    num_steps = 252  # Number of trading days in a year

    stock_paths = monte_carlo_stock_price(S0, mu, sigma, T, num_simulations, num_steps)
    
    call_price_mc = monte_carlo_option_price(stock_paths,K,r,T,option_type='call')
    put_price_mc = monte_carlo_option_price(stock_paths,K,r,T,option_type='put')


    print("Monte Carlo Call Price:", call_price_mc)
    print("Monte Carlo Put Price:", put_price_mc)