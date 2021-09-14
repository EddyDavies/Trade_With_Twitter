import hodl.data_collection.utils
from .logger import Logger


class EpochCallback:
    def on_epoch_end(self, e):
        pass

    def on_epoch_start(self, e):
        pass

    def on_step(self, e, n):
        pass


class SaveAgentCallback(EpochCallback):
    def __init__(self, agent, file_name):
        """
        "fixed_name"
        "lambda e, n: f'weights-{e}'"
        """
        self.__agent = agent
        self.file_name = file_name

    def on_epoch_end(self, e):
        hodl.data_collection.utils.save(self.file_name(e) if type(self.file_name) != str else self.file_name)


class TestOnEpochEndCallback(EpochCallback):
    def __init__(self, env, agent, render=True, action=None):
        self.env = env
        self.agent = agent
        self.__action_convertor = action
        self.render = render

    def on_epoch_end(self, _):
        state = self.env.reset()
        done = False
        while not done:
            action = self.agent.choose_action(state)

            if self.__action_convertor is not None:
                action = self.__action_convertor(action)

            state, reward, done, info = self.env.step(action)

            if self.render:
                self.env.render()
        print(f"\rTotal Reward: {info['assets']}")


class SaveLoggerCallback(EpochCallback):
    def __init__(self, logger: Logger, file_name):
        self.logger = logger
        self.file_name = file_name

    def on_epoch_end(self, e):
        self.logger.save(self.file_name(e) if type(self.file_name) != str else self.file_name)
