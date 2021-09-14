class Epsilon:
    def __init__(self):
        pass

    def get(self, _):
        raise NotImplementedError


class FixedEpsilon(Epsilon):
    def __init__(self, value=0.05):
        super().__init__()
        self.value = value

    def get(self, _):
        return self.value


class LinearEpsilon(Epsilon):
    def __init__(self, start, decay=0, min_epsilon=0):
        super().__init__()
        self.start = start
        self.decay = decay
        self.min_epsilon = min_epsilon

    def get(self, interval: int) -> float:
        epsilon = self.start - (self.decay * interval)
        return max(self.min_epsilon, epsilon)


class ExponentialEpsilon(Epsilon):
    def __init__(self, start, decay=1, min_epsilon=0):
        super().__init__()
        self.start = start
        self.decay = decay
        self.min_epsilon = min_epsilon

    def get(self, interval: int) -> float:
        epsilon = self.start * (self.decay ** interval)
        return max(self.min_epsilon, epsilon)
