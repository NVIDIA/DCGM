#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reports on tests in tests_mock that do not mock anything (Orphans) and tests
# in tests that are not mocked or trampolined as non-hardware (Hardware Only).
# Run without arguments in the common parent directory of tests and tests_mock.

import ast
import os
import sys
from pathlib import Path


def _is_trampoline_only(func_def: ast.FunctionDef) -> bool:
    body = func_def.body
    # Skip an optional leading docstring.
    if (body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Expr):
        return False
    call = body[0].value
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == 'trampoline'
    )


def _parse_tests(directory: str, mocked_label: str, hardwareless_label: str) -> list[tuple[str, str]]:
    """Return (qualified_name, label) pairs for all top-level test_* functions."""
    dir_path = Path(directory)
    entries = []
    for py_file in sorted(dir_path.rglob('test*.py')):
        prefix = (
            str(py_file.relative_to(dir_path))
            .replace(os.sep, '.')
            .removesuffix('.py') + '.'
        )
        try:
            tree = ast.parse(py_file.read_text(
                encoding='utf-8'), filename=str(py_file))
        except SyntaxError as exc:
            print(f"# SyntaxError in {py_file}: {exc}", file=sys.stderr)
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                label = hardwareless_label if _is_trampoline_only(
                    node) else mocked_label
                entries.append((f"{prefix}{node.name}", label))
    return entries


def check_tests() -> None:
    tests_entries = _parse_tests('tests', 'NORMAL', 'NORMAL')
    mock_entries = _parse_tests('tests_mock', 'Mocked', 'Non-Hardware')

    tests_keys = {key for key, _ in tests_entries}
    mock_keys = {key for key, _ in mock_entries}

    lines = [f"{key}: {label}" for key, label in mock_entries]

    for key, _ in mock_entries:
        if key not in tests_keys:
            lines.append(f"{key}: Orphan")
    for key, _ in tests_entries:
        if key not in mock_keys:
            lines.append(f"{key}: Hardware Only")

    for line in sorted(lines):
        print(line)


if __name__ == '__main__':
    check_tests()
