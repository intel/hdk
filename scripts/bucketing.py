#!/usr/bin/python3

import subprocess
import argparse
import pathlib
from string import whitespace
import yaml
import signal


def signal_to_name(sig: int):
    return signal.Signals(sig).name


def generate_test_run_options(gtest_output: str):
    # Gtest prints a 'starting' string and a summary at the end
    # FIXME: The ending is specific to ArrowBasedExecuteTest
    lines = gtest_output.splitlines()[1:-2]
    run_opts = []
    base = lines[0]
    for line in lines:
        if line.startswith(tuple(w for w in whitespace)):
            run_opts += [base + line.strip()]
        else:
            base = line
    return run_opts


parser = argparse.ArgumentParser(prog='bucketing',
                                 usage="""%(prog)s <filename> [options]\nExample: ./scripts/%(prog)s.py build/omniscidb/Tests/ArrowBasedExecuteTest -o report.yaml""",
                                 description="""This is an utility script to collect gtest 
                                    tests and generate a status report by running tests one 
                                    by one and bucketing any discovered issues.""")
parser.add_argument('testname', help='Path to the gtest executable.')
parser.add_argument('--no-run', default=False, action='store_true',
                    help="Don't run the tests, just print their names.")
parser.add_argument('-a', '--advanced-bucketing', default=False,
                    action='store_true', help="Use fine-grained bucketing. No effect if --no-run is enabled.")
parser.add_argument('-o', '--output', type=argparse.FileType('w'),
                    help="Re-route output to a yaml file.")
args = parser.parse_args()

executable = pathlib.Path(args.testname)
if not executable.is_file():
    print('File not found:', executable)
    exit(1)

p = subprocess.run([executable, '--gtest_list_tests'],
                   capture_output=True, encoding='utf-8')
p.check_returncode()

tests = generate_test_run_options(p.stdout)

if args.no_run:
    if args.output:
        [args.output.write(str(t) + '\n') for t in tests]
    else:
        [print(t) for t in tests]
    exit(0)


report = {}


def add_to_bucket(category: dict, bucket, test, stderr=None):
    record = test if not stderr else {'name': test, 'output': stderr}
    if bucket not in category.keys():
        category[bucket] = [record]
    else:
        category[bucket].append(record)


def add_to_report(bucket, test):
    add_to_bucket(report, bucket, test)


def basic_bucketize(p: subprocess.CompletedProcess):
    if p.returncode < 0:
        add_to_report(signal_to_name(-p.returncode), test)
    elif p.returncode == 0:
        add_to_report('PASS', test)
    elif p.returncode > 0:
        add_to_report('ERROR', test)
    else:
        assert (False)


def advanced_bucketize(p: subprocess.CompletedProcess):
    if p.returncode == 0:
        add_to_report('PASS', test)
    else:
        top_level_bucket = signal_to_name(
            -p.returncode) if p.returncode < 0 else 'ERROR'
        if top_level_bucket not in report.keys():
            report[top_level_bucket] = {}
        if 'Check failed: Data_Namespace::CPU_LEVEL' in p.stderr:
            add_to_bucket(report[top_level_bucket],
                          'MEMORY_LEVEL', test)
        elif 'Abort /opt/src/l0_gpu_driver/shared/source/memory_manager/memory_manager.cpp' in p.stdout:
            add_to_bucket(report[top_level_bucket], 'DRIVER_FAILURE', test)
        elif 'Check failed: func' in p.stderr:
            add_to_bucket(report[top_level_bucket], 'CODEGEN_FAILURE', test)
        elif 'Check failed: hash_table_cache_' in p.stderr:
            add_to_bucket(report[top_level_bucket], 'HASH_TABLE_CACHE', test)
        elif 'SQLiteComparator' in p.stdout:
            add_to_bucket(report[top_level_bucket], 'INCORRECT_RESULT', test)
        else:
            add_to_bucket(report[top_level_bucket], 'OTHER', test, p.stderr)


for test in tests:
    if args.advanced_bucketing:
        p = subprocess.run([executable, '--gtest_filter=' + test],
                           capture_output=True, encoding='utf-8')
        print(p.stdout, p.stderr)
        advanced_bucketize(p)
    else:
        p = subprocess.run([executable, '--gtest_filter=' + test])
        basic_bucketize(p)

if args.output:
    yaml.dump(report, args.output)
else:
    print(yaml.dump(report))
