import sys
import argparse

from portfolio_management.soft_actor_critic.train import train
from portfolio_management.soft_actor_critic.evaluate import evaluate


def get_parser(formatter_class=argparse.RawTextHelpFormatter):
    parser = argparse.ArgumentParser(
        description='PyTorch Soft Actor-Critic',
        usage='Use "python soft_actor_critic --help" for more information',
        formatter_class=formatter_class
    )

    # choice of the subparser
    subparsers = parser.add_subparsers(help='Selection of the mode to perform', dest='mode')

    # parent parser
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '--env-name', default="LunarLanderContinuous-v2", type=str, metavar='',
        help='Gym environment to train on (default: %(default)s)'
    )
    parent_parser.add_argument(
        '--hidden-units', nargs='+', type=int, default=[256, 256], metavar='',
        help='List of networks\' hidden units (default: %(default)s)'
    )
    parent_parser.add_argument(
        '--directory', default='../runs/', type=str, metavar='',
        help='Root directory in which the run folder will be created (default: %(default)s)'
    )
    parent_parser.add_argument(
        '--seed', default=1, type=int, metavar='',
        help='Seed used for pytorch, numpy and the environment (default: %(default)s)'
    )

    # train parser
    parser_train = subparsers.add_parser(
        "train", parents=[parent_parser],
        help='Train an agent'
    )
    parser_train.add_argument(
        '--run-name', default=None, type=str, metavar='',
        help='Name used for saving the weights and the logs (default: generated using the "get_run_name" function)'
    )
    parser_train.add_argument(
        '--batch-size', default=256, type=int, metavar='',
        help='Batch size used by the agent during the learning phase (default: %(default)s)'
    )
    parser_train.add_argument(
        '--memory-size', default=1_000_000, type=int, metavar='',
        help='Size of the replay buffer (default: %(default)s)'
    )
    parser_train.add_argument(
        '--learning-rate', default=3e-4, type=float, metavar='',
        help='Learning rate used for the networks and entropy optimization (default: %(default)s)'
    )
    parser_train.add_argument(
        '--gamma', default=0.99, type=float, metavar='',
        help='Discount rate used by the agent (default: %(default)s)'
    )
    parser_train.add_argument(
        '--tau', default=0.005, type=float, metavar='',
        help='Value used for the progressive update of the target networks (default: %(default)s)'
    )
    parser_train.add_argument(
        '--num-steps', default=1_000_000, type=int, metavar='',
        help='Number training steps (default: %(default)s)'
    )
    parser_train.add_argument(
        '--start-step', default=1_000, type=int, metavar='',
        help='Step after which the agent starts to learn (default: %(default)s)'
    )
    parser_train.add_argument(
        '--alpha', default=0.2, type=float, metavar='',
        help='Starting value of the entropy (alpha) (default: %(default)s)'
    )

    # eval parser
    parser_eval = subparsers.add_parser(
        "eval", parents=[parent_parser],
        help='Evaluate the performance of an already trained agent'
    )
    parser_eval.add_argument(
        '--run-name', type=str, metavar='',
        help='Run name of an already trained agent located in the "--directory" directory'
    )
    parser_eval.add_argument(
        '--num-episodes', default=3, type=int, metavar='',
        help='Number of episodes to run (default: %(default)s)'
    )
    parser_eval.add_argument(
        '--deterministic', default=False, action='store_true',
        help='Toggle deterministic behavior of the agent when interacting with the environment'
    )
    parser_eval.add_argument(
        '--render', default=False, action='store_true',
        help='Toggle the rendering of the episodes'
    )
    parser_eval.add_argument(
        '--record', default=False, action='store_true',
        help='Toggle the recording of the episodes (toggling "record" would also toggle "render")'
    )
    return parser


def main(mode, **kwargs):
    if mode == 'train':
        print('Training SAC with the following arguments:')
        print(kwargs)
        train(**kwargs)
    elif mode == 'eval':
        print(f'Evaluating SAC of the run {kwargs["run_name"]} the following arguments:')
        print(kwargs)
        evaluate(**kwargs)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    elif arguments.mode == 'eval' and arguments.run_name is None:
        parser.error('The --run-name argument is required in eval mode')
    kwargs = vars(arguments)
    mode = kwargs.pop('mode', 'train')
    exit(main(mode, **kwargs))
