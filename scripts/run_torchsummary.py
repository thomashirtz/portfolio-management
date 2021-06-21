from torchsummary import summary

from portfolio_management.paths import get_models_folder_path
from portfolio_management.io_utilities import read_json_file
from portfolio_management.supervised.utilities import get_sequential
from portfolio_management.soft_actor_critic.utilities import get_device


if __name__ == '__main__':
    model_name = 'model_0'

    models_path = get_models_folder_path(None)
    path_config = models_path.joinpath(model_name).joinpath('config.json')
    json_config = read_json_file(path_config)

    device = get_device()
    sequential = get_sequential(json_config['network_config']['module_list']).to(device)

    input_shape = tuple([9, 50])  # Channels * Number of Observations
    summary(sequential, input_size=input_shape)
