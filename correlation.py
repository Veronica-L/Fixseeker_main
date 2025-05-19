import csv, json
import os.path
import re
import pandas as pd
from hunk import get_hunk_from_diff, retrieve_commit_content
import subprocess
from hunk_relation import extract_relation
from multiprocessing import Pool, Manager
import logging
import time, signal

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Process program parameters')

    parser.add_argument('--dataset_path',
                        type=str,
                        default='c_imbalance.json',
                        help='Path to the dataset file')

    parser.add_argument('--program_language',
                        type=str,
                        default='c',
                        help='Programming language to process')

    parser.add_argument('--type',
                        type=str,
                        default='nonvul',
                        help='Type of vulnerability')

    parser.add_argument('--balance_type',
                        type=str,
                        default='imbalance',
                        help='Balance type of the dataset')

    args = parser.parse_args()
    return args

args = parse_args()
dataset_path = f'./data/{args.dataset_path}.json'
PROGRAM_LANGUAGE = args.program_language
TYPE = args.type
BALANCE_TYPE = args.balance_type
output_json_path = f"data/correlations/{PROGRAM_LANGUAGE}_{BALANCE_TYPE}.json"
logging.basicConfig(filename=f'log/Correlation_{PROGRAM_LANGUAGE}_{BALANCE_TYPE}.log', filemode='a', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TimeOutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeOutError("func out of time")

def run_with_timeout(func, args=(), kwargs={}, timeout=60):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        hunks, relations = extract_relation(*args, **kwargs)
        signal.alarm(0)
        return hunks, relations
    except TimeOutError:
        print(f"timed out waiting for {timeout} seconds")
        return None, None


def is_source_code_file(filename):
    # make sure the target patch is from source code file.
    # return True if filename is a source code file
    return filename.endswith('.c') or filename.endswith('.cpp') \
           or filename.endswith('.cxx') #or filename.endswith('.h') or filename.endswith('.java') or filename.endswith('.py')

def process_url(args):
    index, (url, label) = args
    print(index, url)
    owner, repo, hash = url.split('/')[-4], url.split('/')[-3], url.split('/')[-1]
    hunks, relations = extract_relation(owner, repo, hash)
    if hunks == []:
        return None
    os.system('rm -rf joern/workspace/*')
    try:
        json_str = json.dumps({url: {"index": index, "hunks": hunks, "realtions": relations, "label": label}})

        with open(output_json_path, 'a+') as f:
            f.write(json_str + '\n')
    except:
        return None

    return url, {"hunks": hunks, "realtions": relations, "label": label}

def get_data():
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    vul_list = data["vul"]
    nonvul_list = data["nonvul"]

    url_to_label = {}

    for url in vul_list:
        url_to_label[url] = 1
    for url in nonvul_list:
        url_to_label[url] = 0

    print(len(url_to_label))

    for index, url in enumerate(url_to_label):
        owner, repo, hash = url.split('/')[-4], url.split('/')[-3], url.split('/')[-1]

        #hunks, relations = extract_relation(owner, repo, hash)
        hunks, relations = run_with_timeout(extract_relation, args=(owner, repo, hash), timeout=120)
        if hunks is None and relations is None:
            continue

        label = url_to_label[url]
        if hunks == []:
            continue
        os.system('rm -rf joern/workspace/*')
        try:
            json_str = json.dumps({url: {"index": index, "hunks": hunks, "realtions": relations, "label": label}})

            with open(output_json_path, 'a+') as f:
                f.write(json_str + '\n')  
            logging.info("Successfully saved json to {}".format(output_json_path))
        except:
            print('Fail')
            continue


        #json_str = json.dumps(commit_dict, indent=4)
        #with open('sub_python_data.json', 'w') as json_file:
            #json_file.write(json_str)

def train():
    print("Reading dataset...")
    get_data()


if __name__ == '__main__':
    train()