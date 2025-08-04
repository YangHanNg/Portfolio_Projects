import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

time = 60*60*24*1
### Market Marking Simulator ################################################################################################

timeset = 


# Create a class to randomly generate participant roles
class MarketRoleGenerator:
    def __init__(self, roles):
        self.roles = roles

    def generate_role(self):
        return np.random.choice(self.roles)

# Create a class to simulate participants in the market
class MarketParticipant:
    def __init__(self, name, initial_balance):
        self.name = name
        self.balance = initial_balance
        self.position = 0  # Number of units held

    def buy(self, price, quantity):
        cost = price * quantity
        if cost <= self.balance:
            self.balance -= cost
            self.position += quantity
            print(f"{self.name} bought {quantity} units at {price} each.")
        else:
            print(f"{self.name} cannot afford to buy {quantity} units at {price}.")

    def sell(self, price, quantity):
        if quantity <= self.position:
            revenue = price * quantity
            self.balance += revenue
            self.position -= quantity
            print(f"{self.name} sold {quantity} units at {price} each.")
        else:
            print(f"{self.name} does not have enough units to sell {quantity}.")

# Create a class to simulate a limit order book
class LimitOrderBook:
    def __init__(self):
        self.bids = []  # List of bid orders
        self.asks = []  # List of ask orders

    def add_bid(self, price, quantity):
        self.bids.append((price, quantity))
        self.bids.sort(reverse=True)  # Sort bids in descending order

    def add_ask(self, price, quantity):
        self.asks.append((price, quantity))
        self.asks.sort()  # Sort asks in ascending order

    def get_best_bid(self):
        return self.bids[0] if self.bids else None

    def get_best_ask(self):
        return self.asks[0] if self.asks else None

    def match_orders(self):
        while self.bids and self.asks and self.get_best_bid()[0] >= self.get_best_ask()[0]:
            bid_price, bid_quantity = self.get_best_bid()
            ask_price, ask_quantity = self.get_best_ask()

            trade_quantity = min(bid_quantity, ask_quantity)
            trade_price = (bid_price + ask_price) / 2  # Mid-price for the trade

            print(f"Trade executed: {trade_quantity} units at {trade_price}")

            # Update quantities
            if bid_quantity > trade_quantity:
                self.bids[0] = (bid_price, bid_quantity - trade_quantity)
            else:
                self.bids.pop(0)

            if ask_quantity > trade_quantity:
                self.asks[0] = (ask_price, ask_quantity - trade_quantity)
            else:
                self.asks.pop(0)

