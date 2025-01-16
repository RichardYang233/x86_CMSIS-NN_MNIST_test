#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import shutil


def write_csv_rows(csv_file, data):
    writer = writer = csv.writer(csv_file)
    writer.writerows(data)

def write_csv_row(csv_file, data):
    writer = csv.writer(csv_file)
    writer.writerow(data)

def copy_csv_file(src_path, drt_path): 
    shutil.copy(src_path, drt_path)
    print(f"File copied from {src_path} to {drt_path}")