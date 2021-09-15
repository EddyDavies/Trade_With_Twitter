from typing import List, Optional, Tuple

from .callbacks import EpochCallback
from .epsilon import Epsilon, FixedEpsilon


class Epochs:
    def __init__(
            self,
            epochs=100,
            steps_per_epoch=50_000,
            initial_epochs=0,
            epsilon: Optional[Epsilon] = None,
            callbacks: Optional[List[EpochCallback]] = None
    ):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        self.epsilon: Optional[Epsilon] = epsilon

        self.__initial_epochs = initial_epochs
        self.__epoch = self.__initial_epochs
        self.__step = 0

        self.callbacks = [] if callbacks is None else callbacks

    def __iter__(self):
        self.__epoch = self.__initial_epochs
        self.__step = 0
        return self

    def __next__(self):
        if self.__epoch < self.epochs:
            self.__step += 1
            n = self.__step
            e = self.__epoch

            if self.__step >= self.steps_per_epoch:
                self.__epoch += 1
                self.__step %= self.steps_per_epoch

            if self.__step == 1:
                if self.__epoch > self.__initial_epochs:
                    for callback in self.callbacks: callback.on_epoch_end(self.__epoch)
                for callback in self.callbacks: callback.on_epoch_start(self.__epoch + 1)

            if self.epsilon is not None:
                return e + 1, n, self.epsilon.get(e * self.steps_per_epoch + n)
            return e + 1, n

        raise StopIteration
