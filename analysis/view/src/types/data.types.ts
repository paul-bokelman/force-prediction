export interface CandidateReport {
  candidate: {
    hash: string;
    identifier: string;
    architecture: string;
    version: number;
    hyperparameters: {
      preprocessing: Record<string, string | number>;
      training: Record<string, string | number>;
    };
  };
  metrics: Record<string, { baseline: number; candidate: number; improvement: boolean }>;
  plots: {
    training_history: string;
    predictions: {
      best: string;
      worst: string;
      average: string;
    };
    metrics: {
      number_of_neurons: string;
      data_volume: string;
    };
  };
}
