import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--city',                 default="NY",       help='City name, can be NY or Chi or SF')
parser.add_argument('--data_path',                    default='./NewYork')
args = parser.parse_args()
if args.city == 'NY':
    parser.add_argument('--data_path',                    default='./NewYork')
else:
    parser.add_argument('--data_path',                    default='./Chicago')