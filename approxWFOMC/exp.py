import os
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str)
parser.add_argument('--repeats', type=int, default=10)
parser.add_argument('--domain_sizes', type=str, default="2,3,4,5,6")
parser.add_argument('--input', type=str)
parser.add_argument('--args', type=str, default="")

args = parser.parse_args()

if not os.path.exists(args.log_dir):
    os.system('mkdir -p {}'.format(args.log_dir))

domain_sizes = args.domain_sizes.split(',')
for domain_size in domain_sizes:
    for i in range(args.repeats):
        print(domain_size, i)
        os.system('python approxwfomc.py {} {} --debug --log {}/domain_{}_{}.log {} > /dev/null 2>&1'.format(
            domain_size, args.input, args.log_dir, domain_size, i, args.args
        ))
