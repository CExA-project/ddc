#!/usr/bin/env python3

# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

"""
Compare the difference in preprocessor macros between two source files
(one with a library and one without), and check if that diff matches a reference.

Uses compile_commands.json to simulate compilation and extract preprocessor macros.
The reference file must be a JSON file containing a list of expected #define macros.
"""

import json
import re
import subprocess
import shlex
from pathlib import Path
import argparse
import sys


def extract_command(entry):
    """
    Modify compile command to only run preprocessor: remove -c/-o, add -dM -E.

    Args:
        entry (dict): An entry from compile_commands.json

    Returns:
        tuple[list[str], Path]: The modified command and working directory
    """
    parts = shlex.split(entry["command"])
    new_parts = []
    skip_next = False
    for part in parts:
        if skip_next:
            skip_next = False
            continue
        if part == "-c":
            continue
        if part == "-o":
            skip_next = True
            continue
        new_parts.append(part)
    new_parts += ["-dM", "-E", entry["file"]]
    return new_parts, Path(entry["directory"])


def run_preprocessor(command, cwd):
    """
    Run the preprocessor and return unique macro lines.

    Args:
        command (list[str]): Compiler command
        cwd (Path): Working directory

    Returns:
        set[str]: Set of macro lines
    """
    result = subprocess.run(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(command)}\n{result.stderr}")
    return set(result.stdout.splitlines())


def find_entry(commands, source_path):
    """
    Find compile command for the given source file.

    Args:
        commands (list[dict]): compile_commands.json content
        source_path (Path): Path to source file

    Returns:
        dict: Matching compile command entry
    """
    for entry in commands:
        if Path(entry["file"]) == source_path:
            return entry
    raise ValueError(f"Source file {source_path} not found in compile_commands.json")


def prepare_commands(args):
    """
    Load compilation database and extract preprocessor commands for both source files.

    Args:
        args (argparse.Namespace): Parsed command-line arguments

    Returns:
        tuple[set[str], set[str]]: Macros with and without the library
    """
    commands = json.loads(args.db.read_text())

    entry_with = find_entry(commands, args.with_src.resolve())
    entry_without = find_entry(commands, args.without_src.resolve())

    cmd_with, dir_with = extract_command(entry_with)
    cmd_without, dir_without = extract_command(entry_without)

    print(f"Running preprocessor for: {args.with_src.name}")
    macros_with = run_preprocessor(cmd_with, dir_with)

    print(f"Running preprocessor for: {args.without_src.name}")
    macros_without = run_preprocessor(cmd_without, dir_without)

    macros_with_diff = macros_with - macros_without
    macros_without_diff = macros_without - macros_with

    if macros_without_diff:
        print(macros_without_diff)
        raise ValueError("Set should be empty")

    return macros_with_diff, macros_without_diff


def extract_macro_diff(macros_with, macros_without):
    """
    Compute the difference in #define macros between with/without source.

    Args:
        macros_with (set[str]): Macros when library is included
        macros_without (set[str]): Macros without library

    Returns:
        list[str]: Sorted list of macro differences
    """
    macro_pattern = r"#define\s+([A-Za-z_][A-Za-z0-9_]*)(\s+.*)?"
    define_with = set()
    for line in macros_with:
        match = re.match(macro_pattern, line.strip())
        if match:
            define_with.add(str(match.group(1)))
        else:
            raise ValueError(f"Unhandled macro {line}")
    define_without = set()
    for line in macros_without:
        match = re.match(macro_pattern, line.strip())
        if match:
            define_without.add(str(match.group(1)))
        else:
            raise ValueError(f"Unhandled macro {line}")
    return sorted(define_with - define_without)


def compare_macro_diff(actual_diff, reference_file):
    """
    Compare computed macro diff to expected one from JSON reference file.

    Args:
        actual_diff (list[str]): Macros computed by diff
        reference_file (Path): Path to JSON reference

    Returns:
        bool: True if matches, False otherwise
    """
    try:
        ref_macros = set(json.loads(reference_file.read_text()))
    except Exception as exception:
        raise ValueError(f"Error reading reference JSON file: {exception}") from exception

    added = sorted([m for m in actual_diff if m not in ref_macros])
    removed = sorted([m for m in ref_macros if m not in actual_diff])

    has_error = False

    if added:
        print("\n--- Unexpected Macros (in diff, not in reference) ---")
        print("\n".join(added))
        has_error = True

    if removed:
        print("\n--- Missing Macros (in reference, not in diff) ---")
        print("\n".join(removed))
        has_error = True

    if not has_error:
        print("\n✅ Macro diff matches the reference exactly.")
    return not has_error


def main():
    """Main entry point: run preprocessor and compare macro diff with reference."""
    parser = argparse.ArgumentParser(
        description="Compare macro diff against a JSON reference file."
    )
    parser.add_argument("--with-src", required=True, type=Path, help="Source file with the library")
    parser.add_argument(
        "--without-src", required=True, type=Path, help="Source file without the library"
    )
    parser.add_argument(
        "--reference", required=True, type=Path, help="Reference JSON file with expected macro diff"
    )
    parser.add_argument(
        "--db",
        default="build-install-test/compile_commands.json",
        type=Path,
        help="Path to compile_commands.json",
    )

    args = parser.parse_args()

    macros_with, macros_without = prepare_commands(args)
    actual_diff = extract_macro_diff(macros_with, macros_without)
    success = compare_macro_diff(actual_diff, args.reference)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
