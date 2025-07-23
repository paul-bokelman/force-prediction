from typing import TypedDict

# ---------------------------------- metrics --------------------------------- #

class CandidateInfo(TypedDict):
    hash: str
    identifier: str
    architecture: str
    version: int
    hyperparameters: dict[str, dict[str, str | int | float]]

class MetricsGroup(TypedDict):
    baseline: float
    candidate: float
    improvement: bool

class PredictionsPlot(TypedDict):
    best: str
    worst: str
    average: str

class MetricsPlot(TypedDict):
    number_of_neurons: str
    data_volume: str

class Plots(TypedDict):
    training_history: str
    predictions: PredictionsPlot
    metrics: MetricsPlot

type ComparisonMetrics = dict[str, MetricsGroup]

class CandidateReport(TypedDict):
    candidate: CandidateInfo
    metrics: ComparisonMetrics
    plots: Plots