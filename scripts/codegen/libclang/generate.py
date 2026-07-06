#!/usr/bin/env python3

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import multiprocessing
import sys
from pathlib import Path
import yaml


class Generator:
    def generate(self):
        script_dir = Path(__file__).resolve().parent
        project_dir = script_dir.parent.parent.parent
        yaml_filepath = script_dir / "files.yml"

        p = argparse.ArgumentParser(description="""
Run the libclang-based code generators

The script reads the files.yml in the same directory. The document is a list of
files to process. See file for format details
                                    
Example usage:
python3 scripts/codegen/libclang/generate.py \
    --binary-dir _out/build/<dir> \
    --code-generator-dir ../libclang-code-generators
""")
        p.add_argument(
            "--binary-dir", type=Path,
            action="store",
            required=True,
            help="Path to the binary directory",
        )
        p.add_argument(
            "--code-generator-dir", type=Path,
            action="store",
            required=True,
            help="Path to the cloned code generator project directory",
        )
        args = p.parse_args()

        # Resolve paths
        args.binary_dir = args.binary_dir.resolve()
        args.code_generator_dir = args.code_generator_dir.resolve()

        # Import generators
        code_generator_dir_str = str(args.code_generator_dir / "src")
        if code_generator_dir_str not in sys.path:
            sys.path.insert(0, code_generator_dir_str)
        try:
            import generate_any_value
            import generate_any_ptr
            import generate_any_unique_ptr
            import generate_concept
        except ImportError as e:
            raise SystemExit(
                f"Could not import generator modules. Check your code generator path. Error: {e}") from e
        generator_fn_map = {
            "any_value": generate_any_value.run_generator,
            "any_ptr": generate_any_ptr.run_generator,
            "any_unique_ptr": generate_any_unique_ptr.run_generator,
            "concept": generate_concept.run_generator,
        }

        # Load yaml entries
        with open(yaml_filepath, encoding="utf-8") as f:
            yaml_obj = yaml.safe_load(f)
        if not isinstance(yaml_obj, dict):
            raise SystemExit(f"{yaml_filepath}: expected a YAML mapping")
        if "files" not in yaml_obj:
            raise SystemExit(f"{yaml_filepath}: missing required key 'files'")
        files = yaml_obj["files"]
        if not isinstance(files, list):
            raise SystemExit(f"{yaml_filepath}: key 'files' is not a list")

        # If files is empty, do nothing
        if len(files) == 0:
            return

        for i, file in enumerate(files):
            # Check block contents
            if not isinstance(file, dict):
                raise SystemExit(
                    f"{yaml_filepath}: file entry at index {i} is not a mapping")
            for entry in ["generator_type", "filepath", "destination_dir", "class_name", "policies"]:
                if entry not in file:
                    raise SystemExit(
                        f"{yaml_filepath}: file entry at index {i} missing required key '{entry}'")

            # Prepend the project directory to each file path
            abs_filepath = project_dir / file["filepath"]
            abs_destination_dir = project_dir / file["destination_dir"]
            # Sanity check
            if not abs_filepath.is_file():
                raise SystemExit(
                    f"File {file['filepath']} does not exist. Double check that {project_dir} is the correct project directory")
            file["filepath"] = abs_filepath
            file["destination_dir"] = abs_destination_dir

            # Insert binary directory for convenience later
            file["binary_dir"] = args.binary_dir

        with multiprocessing.Pool() as pool:
            # Pass additional data to run_generator
            try:
                run_generator_args = [(file, generator_fn_map)
                                      for file in files]
                pool.map(Generator.run_generator, run_generator_args)
            except Exception as e:
                raise SystemExit(
                    f"Generator failed to run. Error: {e}") from e

    # Run the generators in parallel
    @staticmethod
    def run_generator(args):
        file = args[0]
        generator_fn_map = args[1]

        # Prepare arguments and pass to generator
        libclang_path = "/usr/lib/llvm-21/lib/libclang.so"
        generator_fn = generator_fn_map.get(file["generator_type"])
        if generator_fn is None:
            raise ValueError(
                f"Unknown generator type: {file['generator_type']}")
        generator_fn(
            file["filepath"], file["class_name"],
            file["binary_dir"], file["destination_dir"], file["policies"], libclang_path
        )


if __name__ == "__main__":
    generator = Generator()
    generator.generate()
