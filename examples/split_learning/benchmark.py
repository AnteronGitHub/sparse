import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', default='aio', type=str)
    parser.add_argument('--model-name', default='VGG_unsplit', type=str)
    parser.add_argument('--batches', default=64, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--log-file-prefix', default='training-benchmark', type=str)
    return parser.parse_args()

def run_aio_benchmark(args):
    print('All-in-one benchmark suite')
    print('--------------------------')

    from datasets import DatasetRepository
    from models import ModelTrainingRepository
    from nodes.all_in_one import AllInOne

    dataset, classes = DatasetRepository().get_dataset(args.model_name)
    model, loss_fn, optimizer = ModelTrainingRepository().get_model(args.model_name)

    AllInOne(dataset, classes, model, loss_fn, optimizer).train(args.batches,
                                                                args.batch_size,
                                                                args.epochs,
                                                                args.log_file_prefix)

def run_offload_datasource_benchmark(args):
    print('Offload data source benchmark suite')
    print('-----------------------------------')

    import asyncio
    from datasets import DatasetRepository
    from nodes.training_data_source import TrainingDataSource

    dataset, classes = DatasetRepository().get_dataset(args.model_name)
    asyncio.run(TrainingDataSource(dataset, classes, 'VGG_unsplit').train(args.batch_size,
                                                                          args.batches,
                                                                          args.epochs,
                                                                          args.log_file_prefix))
def run_offload_intermediate_benchmark(args):
    print('Offload intermediate node benchmark suite')
    print('-----------------------------------------')

    from nodes.split_training_intermediate import SplitTrainingIntermediate
    from models import ModelTrainingRepository

    model, loss_fn, optimizer = ModelTrainingRepository().get_model(args.model_name)

    SplitTrainingIntermediate(model=model, loss_fn=loss_fn, optimizer=optimizer).start()

def run_offload_final_benchmark(args):
    print('Offload final node benchmark suite')
    print('----------------------------------')

    from nodes.split_training_final import SplitTrainingFinal

    model, loss_fn, optimizer = ModelTrainingRepository().get_model(args.model_name)

    SplitTrainingFinal(model=model, loss_fn=loss_fn, optimizer=optimizer).start()

if __name__ == '__main__':
    args = parse_arguments()

    if args.suite == 'offload-source':
        run_offload_datasource_benchmark(args)
    elif args.suite == 'offload-intermediate':
        run_offload_intermediate_benchmark(args)
    elif args.suite == 'offload-final':
        run_offload_final_benchmark(args)
    else:
        run_aio_benchmark(args)
