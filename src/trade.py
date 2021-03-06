import os

from rl.environment import Stonks
from rl.ppo import PPO
from rl.utils.callbacks import TestOnEpochEndCallback, SaveLoggerCallback
from rl.utils.epochs import Epochs
from rl.utils.logger import Logger


def run(data_type,
        window_size,
        run_type,
        path,
        checkpoint_path,
        log_folder,
        no_window: list = [],
        show_fig=False,
        load_model=False
    ):

    if no_window is not []:
        run_type += "_no_window"
    log_path_csv = os.path.join(log_folder, f"{run_type}.csv")
    log_path_jpg = os.path.join(log_folder, f"{run_type}.jpg")

    checkpoint_path = checkpoint_path.format(run_type, window_size)


    print(f"\n{data_type} with {window_size} window size for all except {no_window}\n")

    if "1" not in run_type:
        no_window.remove('pos1')
        no_window.remove('neg1')

    if "2" not in run_type:
        no_window.remove('pos2')
        no_window.remove('neg2')


    env = Stonks(
        currency="BTC",
        # use_sentiment=USE_SENTIMENT,
        window_size=window_size,
        training_dataset_filepath=path,
        no_window=no_window
    )

    testing_env = Stonks(
        currency="BTC",
        # use_sentiment=USE_SENTIMENT,
        window_size=window_size,
        testing=True,
        training_dataset_filepath=path,
        no_window=no_window
    )

    # print(f"{'No ' if not USE_SENTIMENT else ''}Sentiment")

    agent = PPO(
        input_dims=env.observation_shape,
        n_actions=env.actions,
        batch_size=BATCH_SIZE,
        alpha=LR,
        n_epochs=TRAIN_EPOCHS,
        dims=HIDDEN_DIMS,
        checkpoint_path=checkpoint_path
    )

    # if load_model:
    #     agent.load_models()

    logger = Logger(plot=f" Trading with {run_type} ↗", log_path=log_path_jpg, show_fig=show_fig)
    testing_callback = TestOnEpochEndCallback(testing_env, agent, render=False, action=lambda a: a[0])
    saving_callback = SaveLoggerCallback(logger, log_path_csv)

    current_epoch = 1
    for epoch, episode in Epochs(50, 25, callbacks=[testing_callback, saving_callback]):
        observation = env.reset()
        done = False
        while not done:
            if current_epoch != epoch:
                current_epoch = epoch
                agent.save_models()

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

CHECKPOINT_PATH = "../data/checkpoints/{}_{}.pt"
RAW_PATH = "../data/trade/{}.csv"
LOG_FOLDER= "../data/trade/logs"
CRYPTO = "bitcoin"


if __name__ == '__main__':
    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)

    data_types = ['ta_sa_1',
        # 'ta_sa_12', 'ta_sa_2', 'ta_sa_1',
        #           'sa_12', 'sa_2', 'sa_1', 'ta', 'p'
                  ]
    sizes = [10, 20, 30]
    sizes = [10]

    for window in sizes:
        for data in data_types:
            no_window = ['pos1', 'neg1', 'pos2', 'neg2',
            "SMA_50", "SMA_200", "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0",
             "BBP_20_2.0", "RSI_14", "MACD_8_21_9", "MACDh_8_21_9", "MACDs_8_21_9", "VOLUME_SMA_20"]

            run_type = f"{CRYPTO}_{data}"
            path = RAW_PATH.format(run_type)

            run(data, window, run_type,
                path, CHECKPOINT_PATH, LOG_FOLDER,
                no_window=no_window, show_fig=True,
                load_model=True)


