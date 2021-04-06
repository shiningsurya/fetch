#!/usr/bin/env python3

import logging
import os
import string
import argparse

from fetch.utils import preget_weights

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Extragalactic Transient Candiate Hunter (FETCH)",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='Be verbose', action='store_true')
    parser.add_argument('-m', '--model', help='Index of the model to train', required=True)
    args = parser.parse_args ()

    logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    if args.model not in list(string.ascii_lowercase)[:11]:
        raise ValueError(f'Model only range from a -- k.')

    preget_weights (args.model)
