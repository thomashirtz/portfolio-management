import torch
from portfolio_management.soft_actor_critic.evaluators import BasicEvaluator


def test_basic_evaluator():
    num_samples = 1

    num_symbols = 3
    num_features = 4
    num_observations = 5

    num_outputs = 10
    num_intermediate_outputs = 4

    basic_evaluator = BasicEvaluator(
        num_observations,
        num_features,
        num_outputs,
        num_symbols,
        num_intermediate_outputs
    )
    market_observation = torch.rand((num_samples, num_symbols, num_features, num_observations))
    portfolio_observation = torch.rand((num_samples, 2))
    proportion_observation = torch.rand((num_samples, num_symbols))

    basic_evaluator(market_observation, portfolio_observation, proportion_observation)


if __name__ == '__main__':
    test_basic_evaluator()