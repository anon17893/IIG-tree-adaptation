import yaml
import argparse


from agents.utils import ExperimentGenerator


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        help='Path to configuration file.')
    args = parser.parse_args()

    config_path = args.config

    exp_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)        

    exp_generator = ExperimentGenerator(**exp_config)
    tuning_parameters=exp_config.get('tuning_parameters')
    if tuning_parameters is not None and tuning_parameters.get('tune_parameters'):
        exp_generator.tune_rates()
    exp_generator.run()