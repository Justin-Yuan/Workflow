""" entry point for execution 
"""

import os 
import sys 
import argparse 

from workflow.common.config import cfg


def parse_args():
    """get main arguments from config file and command line 
    """
    parser = argparse.ArgumentParser(description="Project name")
    parser.add_argument(
        "-c", "--config-file",
        default="configs/example.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg 


def main(config):
    pass 




if __name__ == "__main__":
    config = parse_args()
    main(config)
