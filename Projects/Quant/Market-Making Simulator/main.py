import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

TIME = 60*60*24*252  # Time unit for the simulation, e.g., seconds
PRICE = 100.0
PARTICIPANTS = 1000  # Number of market participants
CAPITAL = 80000
TICK_SIZE = 0.25  # Minimum price movement

time_step = np.arange(0, TIME, 1)  # Time steps for the simulation   

def initialise_participants():
    participants = []
    id = 0 
    for i in range(PARTICIPANTS):
        int1 = round(np.random.randint(0, PARTICIPANTS)/PARTICIPANTS * CAPITAL, 1) # Decides their starting capital
        int2 = np.random.randint(0, PARTICIPANTS)/PARTICIPANTS  # Decides their risk aversion
        int2 = True if int2 < 0.35 else False  # Convert to boolean for risk aversion
        int4 = np.random.randint(0, PARTICIPANTS)/PARTICIPANTS  # Decides their market participation
        int5 = np.random.randint(0, PARTICIPANTS)/PARTICIPANTS  # Decides their trading probability
        int5 = int5 - 0.30 if int5 > 0.40 else int5
        int5 = round(int5, 2)  # Round trading probability to two decimal places
        participants.append((i, int1, int2, int4, int5))

    participants = pd.DataFrame(participants, columns=['id', 'capital', 'risk_aversion', 'market_participation', 'trading_probability'])

    return participants

# I need a way to take the list of participants and assign their starting positions
# Then, I need a way to simulate their orders based on their risk aversion, market participation, and trading probability at this time step and price
# Orders are to go through a clearing house, which will process them and update the new market price as the mid price of the best ask and best bid


def place_order(participant, time, prices):
    # Simulate order placement based on participant's attributes
    if np.random.rand() < participant['market_participation'] and np.random.rand() < participant['trading_probability']: # Check if participant is active and willing to trade
        # Randomly decide order side, type, and size
        order_side = 'buy' if np.random.rand() < 0.5 else 'sell' # 50% chance of buying or selling
        order_type = 'limit' if np.random.rand() < 0.7 else 'market'  # 70% limit orders, 30% market orders
        order_size = np.random.rand()*100

        # Assign order price based on order type
        if order_type == 'market':
            order_price = prices 
        else:
            limit_price_raw = prices * (1 + np.random.uniform(-0.01, 0.01))  # Market orders can deviate from the current price
            order_price = round(limit_price_raw / TICK_SIZE) * TICK_SIZE  # Round to nearest tick size
        
        return {
            'time': time,
            'price': order_price,
            'side': order_side,
            'id': participant['id'],
            'order_type': order_type,
            'order_size': order_size,
            'order_status': 0 # 0 for queued, 1 for partially filled, 2 for filled, 3 for cancelled
        }
    return None

def simulate_market(participants):
    prices = np.zeros(TIME)  # Initialize prices array
    prices[0] = PRICE  # Set initial price
    for t in range(1, TIME):
        orders = []
        for _, participant in participants.iterrows():
            order = place_order(participant, t, prices[t-1])
            if order:
                orders.append(order)

        if orders:
            # Process orders and update price
            buy_orders = [o for o in orders if o['order_type'] == 'buy']
            sell_orders = [o for o in orders if o['order_type'] == 'sell']

            total_buy_size = sum(o['order_size'] for o in buy_orders)
            total_sell_size = sum(o['order_size'] for o in sell_orders)

            if total_buy_size > total_sell_size:
                price_change = (total_buy_size - total_sell_size) * 0.01  # Adjust price based on order imbalance
                prices[t] = prices[t-1] + price_change
            else:
                price_change = (total_sell_size - total_buy_size) * 0.01
                prices[t] = prices[t-1] - price_change
        else:
            prices[t] = prices[t-1]  # No trades, price remains the same