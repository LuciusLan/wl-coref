""" Runs experiments with CorefModel.

Try 'python run.py -h' for more details.
"""

import argparse
from contextlib import contextmanager
import datetime
import random
import sys
import time

import numpy as np  # type: ignore
import torch        # type: ignore

from coref import CorefModel
from coref.sep_span_predictor import SpanModel
from coref.utils import init_logger


@contextmanager
def output_running_time():
    """ Prints the time elapsed in the context """
    start = int(time.time())
    try:
        yield
    finally:
        end = int(time.time())
        delta = datetime.timedelta(seconds=end - start)
        print(f"Total running time: {delta}")


def seed(value: int) -> None:
    """ Seed random number generators to get reproducible results """
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)           # type: ignore
    torch.backends.cudnn.deterministic = True   # type: ignore
    torch.backends.cudnn.benchmark = False      # type: ignore


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", choices=("train", "eval", "train_sp", "eval_sp"))
    argparser.add_argument("experiment")
    argparser.add_argument("--config-file", default="config.toml")
    argparser.add_argument("--data-split", choices=("train", "dev", "test"),
                           default="test",
                           help="Data split to be used for evaluation."
                                " Defaults to 'test'."
                                " Ignored in 'train' mode.")
    argparser.add_argument("--batch-size", type=int,
                           help="Adjust to override the config value if you're"
                                " experiencing out-of-memory issues")
    argparser.add_argument("--warm-start", action="store_true",
                           help="If set, the training will resume from the"
                                " last checkpoint saved if any. Ignored in"
                                " evaluation modes."
                                " Incompatible with '--weights'.")
    argparser.add_argument("--weights",
                           help="Path to file with weights to load."
                                " If not supplied, in 'eval' mode the latest"
                                " weights of the experiment will be loaded;"
                                " in 'train' mode no weights will be loaded.")
    argparser.add_argument("--word-level", action="store_true",
                           help="If set, output word-level conll-formatted"
                                " files in evaluation modes. Ignored in"
                                " 'train' mode.")
    args = argparser.parse_args()

    

    if args.warm_start and args.weights is not None:
        print("The following options are incompatible:"
              " '--warm_start' and '--weights'", file=sys.stderr)
        sys.exit(1)

    seed(2020)
    coref_model = CorefModel(args.config_file, args.experiment)
    coref_model.logger = init_logger(log_file=f'{coref_model.config.data_dir}.log')
    if args.batch_size:
        coref_model.config.a_scoring_batch_size = args.batch_size
    
    span_model = SpanModel(args.config_file, args.experiment)
    span_model.logger = init_logger(log_file=f'{coref_model.config.data_dir}.log')
    if args.batch_size:
        span_model.config.a_scoring_batch_size = args.batch_size



        #args.weights = "data/chunk_roberta_(e1_2023.03.13_08.33).pt"
    coref_model.load_weights(path="data/chunk_roberta_split.pt", map_location="cpu",
                        ignore={"bert_optimizer", "general_optimizer",
                                "bert_scheduler", "general_scheduler"})
    span_model.load_weights(path="data/chunk_roberta_best.pt", map_location="cpu",
                        ignore={"bert_optimizer", "general_optimizer",
                                "bert_scheduler", "general_scheduler"})
    coref_model.evaluate(data_split=args.data_split,
                    word_level_conll=args.word_level, sp_model=span_model)
