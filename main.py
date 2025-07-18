import argparse
from config import Config
from fed_same import data_utils, fed_same, baselines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--method', type=str, default='fedsame')
    parser.add_argument('--config', type=str, default='config.py')
    args = parser.parse_args()

    config = Config()
    train_loaders, test_loaders, client_task_map, input_shape, num_classes = data_utils.load_data(args.dataset, config)
    models = fed_same.init_models(args.dataset, config, input_shape, num_classes)

    if args.method == 'fedsame':
        fed_same.train(train_loaders, test_loaders, models, config, client_task_map)
    else:
        baselines.train(args.method, train_loaders, test_loaders, models, config, client_task_map)

if __name__ == '__main__':
    main()