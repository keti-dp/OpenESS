from battery_simulator.battery_models.experiment.experiment import (
    Experiment as BaseExperiment,
)


class Experiment(BaseExperiment):
    def __init__(self, steps):
        super().__init__(steps=steps, nominal_capacity=4.1, v_max=4.25, v_min=3.1)
