# merge results from all log files.

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge log files.')
    parser.add_argument("--result_dir", type=str, required=True)
    args = parser.parse_args()

    all_log_files = [l.strip() for l in open(args.result_dir + '/log_files.txt').readlines()]
    merged_data = []
    for log_file in all_log_files:
        with open(log_file, 'r') as f:
            data = f.readlines()
            merged_data.extend(data)
    
    with open(args.result_dir + '/merged_log.txt', 'w') as f:
        f.writelines(merged_data)
