class GameAiConfig:
    def __init__(self):
        self.numSimulations = 4000
        self.usbExploreBase = 19652
        self.usbExploreInit = 1.25
        self.root_dirichlet_alpha = 0.5 # 0.3 for chess

        self.root_exploration_fraction = 0.25
        self.momentum = 0.9
        self.learning_rate_schedule = {
            0: 0.02,
            100000: 0.002,
            300000: 0.0002,
            500000: 0.0002
        }
