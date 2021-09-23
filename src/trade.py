import os

from rl.environment import Stonks
from rl.ppo import PPO
from rl.utils.callbacks import TestOnEpochEndCallback, SaveLoggerCallback
from rl.utils.epochs import Epochs
from rl.utils.logger import Logger


def run(data_type, window_size, run_type):
    print(f"\n{data_type} with {window_size} window size\n")

    env = Stonks(
        currency="BTC",
        # use_sentiment=USE_SENTIMENT,
        window_size=window_size,
        training_dataset_filepath=path
    )

    testing_env = Stonks(
        currency="BTC",
        # use_sentiment=USE_SENTIMENT,
        window_size=window_size,
        testing=True,
        training_dataset_filepath=path
    )

    # print(f"{'No ' if not USE_SENTIMENT else ''}Sentiment")

    agent = PPO(
        input_dims=env.observation_shape,
        n_actions=env.actions,
        batch_size=BATCH_SIZE,
        alpha=LR,
        n_epochs=TRAIN_EPOCHS,
        dims=HIDDEN_DIMS,
        checkpoint_path=CHECKPOINT_PATH.format(data_type, window_size)
    )

    logger = Logger(plot=f" Trading with {run_type} ↗")
    testing_callback = TestOnEpochEndCallback(testing_env, agent, render=False, action=lambda a: a[0])
    saving_callback = SaveLoggerCallback(logger, log_path)

    for epoch, episode in Epochs(50, 25, callbacks=[testing_callback, saving_callback]):
        observation = env.reset()
        done = False
        while not done:
            action, prob, val = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            agent.remember(observation, action, prob, val, reward, done)
            logger.log(epoch, episode, reward, info["assets"], action, done)
            observation = observation_

LEARN_STEPS = 20
BATCH_SIZE = 5
TRAIN_EPOCHS = 4
LR = 0.0001

# WINDOW_SIZE = 30
HIDDEN_DIMS = 16
USE_SENTIMENT = False

RAW_PATH = "../data/trade/{}.csv"
LOG_FOLDER= "../data/trade/logs"
CRYPTO = "bitcoin"
CHECKPOINT_PATH = "../data/checkpoints/{}_{}.json"


if __name__ == '__main__':
    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)

    data_types = ['ta_sa_12', 'ta_sa_2', 'ta_sa_1',
                  'sa_12', 'sa_2', 'sa_1', 'ta', 'p']
    sizes = [10, 20, 30]
    sizes = [30]
    #
    for window in sizes:
        for data in data_types:
            run_type = f"{CRYPTO}_{data}"
            path = RAW_PATH.format(run_type)
            log_path = os.path.join(LOG_FOLDER, f"{run_type}.csv")

            run(data, window, run_type, path, log_path)


