import argparse
import os

def make_parser():
    parser = argparse.ArgumentParser(description='Exoplanet detection CNN: Local or global data only')
    parser.add_argument('--param', default="param/param_local.json", type=str, help="file name for json attributes.")
    parser.add_argument('--res-path', help='path to save the test outputs at')

    args = parser.parse_args()

    if not os.path.exists('plots'):
        os.makedirs('plots')

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    return args












