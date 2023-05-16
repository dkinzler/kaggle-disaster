import argparse
import disaster.train_transformer as tf
import disaster.evaluate as evaluate
import disaster.extract_embeddings as ee
import disaster.train_roberta as rbt

def main() -> None:
    """Parse command line arguments and execute appropriate command."""

    parser = argparse.ArgumentParser(
        prog="KaggleDisasterTrainer",
        description="Train a neural network for the Kaggle disaster tweet classification challenge.")

    subparsers = parser.add_subparsers(
        help="Available commands", dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train a Transformer model")
    train_parser.add_argument("--dataset", action="store",
                              default="data/train.csv", help="file containing training data")
    train_parser.add_argument("--token_dict", action="store",
                              default="token_dict.pt", help="file containing token dictionary")
    train_parser.add_argument("--pretrained_embeddings", action="store",
                              default="embeddings.pt", help="file containing pretrained token embeddings")
    train_parser.add_argument(
        "-c", "--checkpoint", action="store", help="resume training from checkpoint file")
    train_parser.add_argument("-b", "--batch_size", action="store",
                              type=int, default=32, help="training batch size")
    train_parser.add_argument("-e", "--epochs", action="store",
                              type=int, default=20, help="number of epochs to train for")
    train_parser.add_argument("--use_all_data", action="store_true", default=False,
                              help="train on entire dataset, no split into training/validation")
    train_parser.add_argument("--log_freq", action="store", type=int,
                              default=10, help="log loss/accuracy every LOG_FREQ steps")
    train_parser.add_argument("--checkpoint_freq", action="store", type=int,
                              default=10, help="save checkpoint every CHECKPOINT_FREQ epochs")
    train_parser.add_argument("--checkpoint_dir", action="store",
                              default="ckpts", help="directory to save checkpoints in")
    train_parser.add_argument("--checkpoint_name", action="store", default="model",
                              help="base name for checkpoint files, epoch number will be appended")
    train_parser.add_argument("--tensorboard_log_dir", action="store",
                              default="logs", help="directory to write tensorboard log files to")
    train_parser.add_argument(
        "--profile", action="store_true", default=False, help="run with profiler")

    roberta_parser = subparsers.add_parser(
        "train_roberta", help="Train a simple model based on RoBERTa")
    roberta_parser.add_argument("--dataset", action="store",
                                default="data/train.csv", help="file containing training data")
    roberta_parser.add_argument(
        "-b", "--batch_size", action="store", type=int, default=32, help="training batch size")
    roberta_parser.add_argument("-e", "--epochs", action="store",
                                type=int, default=20, help="number of epochs to train for")
    roberta_parser.add_argument(
        "-c", "--checkpoint", action="store", help="resume training from checkpoint file")
    roberta_parser.add_argument("--use_all_data", action="store_true", default=False,
                                help="train on entire dataset, no split into training/validation")
    roberta_parser.add_argument("--log_freq", action="store", type=int,
                                default=10, help="log loss/accuracy every LOG_FREQ steps")
    roberta_parser.add_argument("--checkpoint_freq", action="store", type=int,
                                default=10, help="save checkpoint every CHECKPOINT_FREQ epochs")
    roberta_parser.add_argument("--checkpoint_dir", action="store",
                                default="ckpts", help="directory to save checkpoints in")
    roberta_parser.add_argument("--checkpoint_name", action="store", default="model",
                                help="base name for checkpoint files, epoch number will be appended")
    roberta_parser.add_argument("--tensorboard_log_dir", action="store",
                                default="logs", help="directory to write tensorboard log files to")

    test_parser = subparsers.add_parser(
        "test", help="Evaluate a model on the test set, outputs predictions to file")
    test_parser.add_argument("checkpoint", action="store",
                             help="checkpoint file to load")
    test_parser.add_argument("--dataset", action="store",
                             default="data/test.csv", help="file containing test data")
    test_parser.add_argument("--token_dict", action="store",
                             default="token_dict.pt", help="file containing token dictionary")
    test_parser.add_argument("-o", "--output", action="store",
                             default="result.csv", help="output file for predictions")
    test_parser.add_argument(
        "-b", "--batch_size", action="store", type=int, default=32, help="batch size")

    inference_parser = subparsers.add_parser(
        "inference", help="Run inference on input sentences from the command line")
    inference_parser.add_argument(
        "checkpoint", action="store", help="checkpoint file to load")
    inference_parser.add_argument("--token_dict", action="store",
                                  default="token_dict.pt", help="file containing token dictionary")

    extract_parser = subparsers.add_parser(
        "extract", help="Extract pretrained fastText word embeddings")
    extract_parser.add_argument(
        "input_file", action="store", help="fastText word vector file")
    extract_parser.add_argument("--output_embeddings", action="store",
                                default="embeddings.pt", help="output file for word embeddings")
    extract_parser.add_argument("--output_token_dict", action="store",
                                default="token_dict.pt", help="output file for token dictionary")
    extract_parser.add_argument("--num_words", action="store", type=int,
                                default=100000, help="extract the first NUM_WORDS words, default 100000")

    args = parser.parse_args()
    match args.command:
        case "train":
            tf.train(tf.TrainConfig(
                dataset=args.dataset,
                token_dict=args.token_dict,
                pretrained_embeddings=args.pretrained_embeddings,
                use_all_data=args.use_all_data,
                checkpoint=args.checkpoint,
                checkpoint_freq=args.checkpoint_freq,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_name=args.checkpoint_name,
                tensorboard_log_dir=args.tensorboard_log_dir,
                profile=args.profile,
                log_freq=args.log_freq,
                batch_size=args.batch_size,
                epochs=args.epochs,
            ))
        case "train_roberta":
            rbt.train_roberta(rbt.RobertaConfig(
                dataset=args.dataset,
                use_all_data=args.use_all_data,
                checkpoint=args.checkpoint,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_freq=args.checkpoint_freq,
                checkpoint_name=args.checkpoint_name,
                log_freq=args.log_freq,
                tensorboard_log_dir=args.tensorboard_log_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
            ))
        case "test":
            evaluate.test(evaluate.TestConfig(
                checkpoint=args.checkpoint,
                dataset=args.dataset,
                token_dict=args.token_dict,
                output=args.output,
                batch_size=args.batch_size,
            ))
        case "inference":
            evaluate.inference(evaluate.InferenceConfig(
                checkpoint=args.checkpoint,
                token_dict=args.token_dict,
            ))
        case "extract":
            ee.extract_embeddings(ee.ExtractEmbeddingsConfig(
                input_file=args.input_file,
                num_words=args.num_words,
                output_embeddings=args.output_embeddings,
                output_token_dict=args.output_token_dict,
            ))
        case _:
            print("unknown command")


if __name__ == "__main__":
    main()
