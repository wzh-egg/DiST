import argparse
from pathlib import Path


def parse_args(description):
    parser=argparse.ArgumentParser(description=description);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--micro',type=int,default=16);parser.add_argument('--eval-batch',type=int,default=16)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--data', type=Path)
    inputs.add_argument('--stations', type=Path)
    parser.add_argument('--grid', type=Path)
    parser.add_argument('--geometry',type=Path,required=True)
    args = parser.parse_args()
    if (args.stations is None) != (args.grid is None):
        parser.error('Supply --stations and --grid together')
    if args.data is not None:
        args.stations = args.data / 'aqi_2016_2019' / 'gwn_candidate_v1'
        args.grid = args.data / 'air_grid_cache'
    return args
