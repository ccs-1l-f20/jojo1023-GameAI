class GameAiConfig:
    def __init__(self):
        self.numSimulations = 100
        self.usbExploreBase = 19652
        self.usbExploreInit = 1.25
        self.root_dirichlet_alpha = 0.35  # 0.3 for chess

        self.gameBatchQueueSize = 100000
        self.batchSize = 4096 # 4096
        self.root_exploration_fraction = 0.25
        self.momentum = 0.9
        self.learning_rate_schedule = {
            0: 0.02,
            10000: 0.002,
            100000: 0.0002,
            300000: 0.00002,
            500000: 0.000002
        }
        self.amountOfTrainingSteps = 700000
        self.checkpointInterval = 50
