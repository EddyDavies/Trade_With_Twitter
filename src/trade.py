from hodl.rl.environment import Stonks
from hodl.rl.ppo import PPO
from hodl.rl.utils.callbacks import TestOnEpochEndCallback
from hodl.rl.utils.epochs import Epochs
from hodl.rl.utils.logger import Logger

LEARN_STEPS = 20
BATCH_SIZE = 5
TRAIN_EPOCHS = 4
LR = 0.0001

WINDOW_SIZE = 8
HIDDEN_DIMS = 16
USE_SENTIMENT = False


if __name__ == '__main__':
    env = Stonks(
        currency="BTC",
        use_sentiment=USE_SENTIMENT,
        window_size=WINDOW_SIZE,
        training_dataset_filepath="../data/bitcoin_trading.csv"
    )

    testing_env = Stonks(
        currency="BTC",
        use_sentiment=USE_SENTIMENT,
        window_size=WINDOW_SIZE,
        testing=True,
        training_dataset_filepath="../data/bitcoin_trading.csv"
    )

    print(f"{'No ' if not USE_SENTIMENT else ''}Sentiment")

    agent = PPO(
        input_dims=env.observation_shape[0],
        n_actions=env.actions,
        batch_size=BATCH_SIZE,
        alpha=LR,
        n_epochs=TRAIN_EPOCHS,
        dims=HIDDEN_DIMS
    )

    logger = Logger(plot=f"StOnKs {'No ' if not USE_SENTIMENT else ''}Sentiment ↗")
    testing_callback = TestOnEpochEndCallback(testing_env, agent, render=False, action=lambda a: a[0])

    for epoch, episode in Epochs(50, 25, callbacks=[testing_callback]):
        observation = env.reset()
        done = False
        while not done:
            action, prob, val = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            agent.remember(observation, action, prob, val, reward, done)
            logger.log(epoch, episode, reward, info["assets"], action, done)
            observation = observation_
