import sys, os
import numpy as np
import json, argparse, sys
import csv
import random

pwd = os.getcwd()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reduce Dataset')

    parser.add_argument('--num',default=100, type=int, help="Number of TCE's for dataset")
    parser.add_argument('--input',default="full_dr24_tce.csv", type=str, help="Input csv file")
    parser.add_argument('--output',default="dr24_tce.csv", type=str, help='path to save test plots including filename eg plots/fig1.png')
    
    args = parser.parse_args()
    
    N = args.num
    input_file = args.input
    output_file = args.output
    
    f = open(pwd+'/'+input_file)
    csv_reader = csv.reader(f)
    
    n = open(pwd+'/'+output_file,'w')
    writer = csv.writer(n)
    
    for i in list(csv_reader)[0:N]:
        writer.writerow(i)
        
    n.close()
