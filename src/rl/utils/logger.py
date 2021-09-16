import time
import numpy as np
import matplotlib.pyplot as plt


class Logger:
    def __init__(
            self,
            plot="Training History"
    ):
        self.plot = plot
        self.fig = None
        self.__history = []

        self.current_epoch = None
        self.current_step = -1

        self.current_epoch_episodes = []
        self.current_assets = 0
        self.current_epochs_assets = []
        self.current_episode_reward = 0

        self.epoch_start = time.time()

    def __set_epoch_step(self, epoch, step):
        if self.current_epoch is None:
            self.current_epoch = epoch

        if self.current_epoch != epoch:
            self.current_epochs_assets.append(self.current_assets)
            epoch_time = time.time() - self.epoch_start

            # Log History
            self.__history += [{
                "min": min(self.current_epoch_episodes, default=0),
                "avg": np.mean(self.current_epoch_episodes) if len(self.current_epoch_episodes) > 0 else 0,
                "max": max(self.current_epoch_episodes, default=0),
                # "loss": np.mean(self.current_epoch_losses) if len(self.current_epoch_losses) > 0 else 0,
                "time": epoch_time
            }]
            self.__plot()
            self.epoch_start = time.time()

            print(end='\r')
            print("{}, Episodes: {}, Min: {:.2f}, Avg: {:.2f}, Max: {:.2f}, Assets: {:.5f}, Time: {:.2f}".format(
                self.current_epoch,
                len(self.current_epoch_episodes),
                min(self.current_epoch_episodes, default=0),
                np.mean(self.current_epoch_episodes) if len(self.current_epoch_episodes) > 0 else 0,
                max(self.current_epoch_episodes, default=0),
                np.mean(self.current_epochs_assets),
                epoch_time
            ))

            self.current_epoch_episodes = []
            self.current_epoch_losses = []

        self.current_epoch = epoch
        self.current_step = step

    def __step(self, reward, total_assets, action, done):
        self.current_episode_reward += reward
        self.current_assets = total_assets

        if done:
            self.current_epoch_episodes.append(self.current_episode_reward)
            self.current_episode_reward = 0

        actions = ["Hold", "Buy", "Sell"]

        print("\r{}:{}, reward: {:.2f}, avg. reward: {:.2f}, Action: {}, Assets: {:.5f}".format(
            self.current_epoch,
            self.current_step,
            self.current_episode_reward,
            np.mean(self.current_epoch_episodes + [self.current_episode_reward]),
            actions[action],
            total_assets,
        ), end="")

        return self.current_episode_reward

    def __plot(self):
        if self.plot not in [False, None, 0]:
            if self.fig is None:
                self.fig = plt.figure(figsize=(5, 3))
                self.fig.show()

            xs = [i + 1 for i in range(len(self.__history))]
            mins = [log["min"] for log in self.__history]
            avgs = [log["avg"] for log in self.__history]
            maxs = [log["max"] for log in self.__history]

            title = "Training History" if type(self.plot) != str else self.plot
            plt.title(title)
            plt.plot(xs, maxs, color="green", label="Best")
            plt.plot(xs, avgs, color="orange", label="Average")
            plt.plot(xs, mins, color="red", label="Worst")
            plt.xlabel("Epoch")
            plt.ylabel("Reward")
            plt.legend()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.0001)
            plt.clf()

    def log(self, epoch: int, step: int, reward: float, total_assets: float, action: float, done: bool):
        self.__set_epoch_step(epoch, step)
        return self.__step(reward, total_assets, action, done)

    def save(self, path: str):
        with open(path, "w+") as file:
            file.write("i, min, avg, max, loss, time\n")
            for i, log in enumerate(self.__history):
                # file.write(f"{i}, {log['min']}, {log['avg']}, {log['max']}, {log['loss']}, {log['time']}\n")
                file.write(f"{i}, {log['min']}, {log['avg']}, {log['max']}, {log['time']}\n")
