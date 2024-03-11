from enum import Enum


class PrivacyMechanismIndex(Enum):
    DiscreteGaussian = "discrete gaussian mechanism"
    Skellam = "skellam mechanism"
    PoissonBionomial = "poisson binomial mechanism"
    Binomial = "binomial mechanism"
    ContinuousGaussian = 'continuous gaussian mechanism'
