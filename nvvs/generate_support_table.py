#!/usr/bin/env python3

import argparse
import csv
import re
import sys
import xml.etree.cElementTree as ET

try:
    import yaml
except ImportError:
    print('Please install pyyaml')
    raise

NAME_ID = ['name', 'id']
CSV = 'csv'
DITA = 'dita'


def main():
    parser = argparse.ArgumentParser(description='Generate Diag support table')
    parser.add_argument('--in', help='Diag config YAML',
                        dest='in_', required=True)
    parser.add_argument('--out', help='Destination CSV', required=True)

    args = parser.parse_args()

    config_name = args.in_
    destination = args.out

    extension_match = re.search(r'\.(\w+)$', destination)
    if extension_match is None:
        print('No extension found')
        sys.exit(1)

    out_type = str.lower(extension_match.group(1))

    with open(config_name) as config_yaml, open(destination, 'w') as out_file:
        config = yaml.safe_load(config_yaml)

        rows = []
        all_tests = set()

        parse_skus(rows, all_tests, config['skus'])
        all_tests_list = sorted(all_tests)

        if out_type == DITA:
            print('Outputting dita xml')
            print_dita(out_file, rows, all_tests_list)
        elif out_type == CSV:
            print('Outputting csv')
            print_csv(out_file, rows, all_tests_list)
        else:
            print('Unrecognized extension')
            sys.exit(1)


def parse_skus(rows, all_tests, skus):
    for sku in skus:
        if 'name' not in sku:
            continue

        row = {'tests': set()}
        for prop in sku:
            if prop == 'id':
                raw_id = str(sku[prop])
                # If len(id) > 4, split with a colon for readability
                _id = raw_id if len(
                    raw_id) <= 4 else raw_id[:4] + ':' + raw_id[4:]
                row[prop] = _id
            elif prop == 'name':
                row[prop] = str(sku[prop])
            else:
                row['tests'].add(prop)
                all_tests.add(prop)
        rows.append(row)


def print_csv(out_file, skus, all_tests_list):
    out_csv = csv.writer(out_file)

    out_csv.writerow(NAME_ID + all_tests_list)

    for sku in skus:
        tests = sku['tests']
        sku_list = [sku['name'], sku['id']] + \
            ['x' if test in tests else '' for test in all_tests_list]
        out_csv.writerow(sku_list)


def print_dita(out_file, skus, all_tests_list):
    row = None

    n_cols = len(NAME_ID + all_tests_list)

    table = ET.Element('table')
    tgroup = ET.SubElement(table, 'tgroup', cols=str(n_cols))

    # Metadata
    for col_name in (NAME_ID + all_tests_list):
        ET.SubElement(tgroup, 'colspec', colname=col_name)

    # Header
    thead = ET.SubElement(tgroup, 'thead')
    row = ET.SubElement(thead, 'row')

    for col_name in (NAME_ID + all_tests_list):
        ET.SubElement(row, 'entry').text = col_name

    # Body
    tbody = ET.SubElement(tgroup, 'tbody')
    for sku in skus:
        row = ET.SubElement(tbody, 'row')
        ET.SubElement(row, 'entry').text = sku['name']
        ET.SubElement(row, 'entry').text = sku['id']
        for test in all_tests_list:
            ET.SubElement(
                row, 'entry').text = 'x' if test in sku['tests'] else ''

    table_tree = ET.ElementTree(table)

    # Pretty-print
    ET.indent(table_tree)

    table_tree.write(out_file, encoding='unicode')


if __name__ == '__main__':
    main()
