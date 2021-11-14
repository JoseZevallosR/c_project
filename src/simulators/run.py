
from ..utils import getStockDataVec, getState, formatPrice

stock_name = 'data_0'
window_size = 5
episode_count = 100
stock_name = str(stock_name)
window_size = int(window_size)
episode_count = int(episode_count)


data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32


def go(agent=None):
    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        state = getState(data, 0, window_size + 1)
        total_profit = 0
        agent.inventory = []
        for t in range(l):
            action = agent.act(state)
            # sit
            next_state = getState(data, t + 1, window_size + 1)
            reward = 0
            if action == 1: # buy
                agent.inventory.append(data[t])
                print("Buy: " + formatPrice(data[t]))
            elif action == 2 and len(agent.inventory) > 0: # sell
                bought_price = window_size_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price
                print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
            done = True if t == l - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                print("--------------------------------")
                print("Total Profit: " + formatPrice(total_profit))
                print("--------------------------------")
            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)
        if e % 10 == 0:
            agent.model.save(str(e))
