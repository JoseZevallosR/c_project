from tensorflow.keras.optimizers import Adam
from src.models.nn import FeedForward
from src.simulators.agents import Agent
from src.simulators.run import go

stock_name = 'data_0'
window_size = 5
episode_count = 100
stock_name = str(stock_name)
window_size = int(window_size)
episode_count = int(episode_count)
model_1 = FeedForward(input_size=window_size, output_size=3, opt=Adam, opt_param={"learning_rate":1e-6})
agent = Agent(state_size=window_size, model=model_1)

go(agent=agent)