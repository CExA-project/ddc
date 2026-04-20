#!/usr/bin/env python3

# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

"""
Script to check if files are missing a newline at the end.
"""

import argparse
import sys


def check_missing_newline(file_path):
    """
    Checks if the given file is missing a newline at the end.
    """
    try:
        with open(file_path, "rb") as file:
            if file.seek(0, 2) == 0:  # Skip empty files
                return False
            file.seek(-1, 2)  # Move to last byte
            if file.read(1) != b"\n":
                print(f"Missing newline at end of file: {file_path}")
                return True
    except OSError as exception:
        print(f"Error checking {file_path}: {exception}")
    return False


def main():
    """
    Parses command-line arguments and checks files for missing newlines.
    """
    parser = argparse.ArgumentParser(description="Check for missing newline at end of file.")
    parser.add_argument("files", type=str, nargs="+", help="Files to analyze.")
    parser.add_argument("--Werror", action="store_true", help="If set, treat warnings as errors")
    args = parser.parse_args()

    missing_newline = any(check_missing_newline(file) for file in args.files)

    if args.Werror and missing_newline:
        sys.exit(1)


if __name__ == "__main__":
    main()
