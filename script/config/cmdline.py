import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Motion capture form Internet Videos')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='config/config_example.yml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--exp', type=str, default=None, help='specify the output name')
    parser.add_argument('--gpu_id', type=int, default=None, help='specify the gpu numbers')
    parser.add_argument('--path', type=str, default='output/serve')
    parser.add_argument('--vis', type=str, choices=['kpts', 'match', 'scene', 'all','mesh'], 
        default='kpts', help='match type')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fps25', action='store_true')
    parser.add_argument('--usecpp', action='store_true')
    parser.add_argument('--eval0', action='store_true')
    parser.add_argument('--tcc', action='store_true')
    args = parser.parse_args()
    return args
