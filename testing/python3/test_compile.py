import os
import option_parser
import fnmatch
import ast
import re
import json
import logger

from importlib import import_module

# global
script_directory = os.path.dirname(__file__) 
test_directory = "/tests"
compiled_file_name = "test_compiled_all.py"

# load from JSON
file_path = script_directory + "/config_tests.json"
with open(file_path, "r") as file:
    json_data = json.load(file)

# Get test files and test functions to exclude from amortized decoration.
files_to_not_parse = json_data["modules_to_run_independent"]
test_funcs_to_not_parse = json_data["tests_to_run_independent"]

def get_file_name():
    return compiled_file_name

def find_files(path, mask = "*", skipdirs=None, recurse=True):
    skipdirs = skipdirs or []
    if recurse:
        for root, dirnames, filenames in os.walk(path):
            if skipdirs is not None:
                [dirnames.remove(skip) for skip in skipdirs if skip in dirnames]  # don't visit directories in skipdirs list
            for filename in fnmatch.filter(filenames, mask):
                yield os.path.abspath(os.path.join(root, filename))
    else:
        #Just list files inside "path"
        filenames = os.listdir(path)
        for filename in fnmatch.filter(filenames, mask):
            yield os.path.abspath(os.path.join(path, filename))

# unwrap
#   This unwraps decorators from a function.
#
# Arguments:
#    func: function to unwrap.
#
# Returns:
#    unwrapped function.
#
def unwrap(func):
    if not hasattr(func, '__wrapped__'):
        return func

    return unwrap(func.__wrapped__)

# get_all_functions
#
# Returns a decorator tuple to function tuple map over the supplied file name
# list. Functions that are not to have their decorator sets shared with other
# functions that have similar decorators are mapped under an empty decorator
# tuple.
#
# Arguments:
#     test_file_names: list of file names to parse
#
# Returns:
#
#    A map of amortized decorator tuples of a list of amortized decorator
# function names to a list of decorators and list of tuples related to
# functions is returned. Each function tuple consists of the original source
# file name of the function, the function name, and the function itself.
#
def get_all_functions(test_file_names):
    decorators_map = {} # indexed by decorator sequence
    amortized_decorators_map = {} # indexed by amortized decorator function name
    decorator_index = 0;

    for file in test_file_names:
        # getting file name and path
        file_path = script_directory + "/" + file + ".py" 
        file_name = file.split("/")[-1]

        if file_name in files_to_not_parse:
            continue

        # open each file and read it
        with open(file_path, 'r') as f:
            """
            lines = f.readlines()
            """

            data = f.read()

        # Build an index of line number to file offset

        line_map=[]
        line_start = True

        for i in range(len(data)):
            if line_start:
                line_map.append(i)
                line_start = False

            if data[i] == '\n':
                line_start = True

        # Parse file

        tree = ast.parse(data)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name[0:4] != "test":
                    continue

                function_name = node.name

                if function_name in test_funcs_to_not_parse:
                    decorator_tuple = ()
                else:
                    decorators = []

                    # Start of function definition. We parse this way to
                    # be compatible with ast 3.6. Version 3.8 has end_lineno
                    # and end_col_offset which makes this much easier, and
                    # version 3.9 has unparse() but we have to support ast 3.6
                    # for RHEL 8.
                    #
                    # Basically, we keep track of the start of each decorator
                    # and extract the text between decorators and between the
                    # last decorator and the function definition. This can
                    # pick up comments (which do not matter), and trailing
                    # white space (which we strip, but which would not matter).

                    node_lineno = node.lineno
                    node_col_offset = node.col_offset
                    last_lineno = -1
                    last_col_offset = -1

                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            if last_lineno >= 0:
                                # This does not remove trailing comments on the
                                # decorator, but they do not matter.
                                decoratorCall = data[line_map[last_lineno - 1] + last_col_offset - 1 : line_map[decorator.lineno - 1] + decorator.col_offset - 1].strip()
                                decorators.append(decoratorCall)

                            last_lineno = decorator.lineno
                            last_col_offset = decorator.col_offset

                    if last_lineno != -1:
                        # Some old ASTs start the node line as the line of the first decorator
                        # We presume the last decorator is all on one line (they all usually are)
                        if last_lineno >= node_lineno:
                            node_lineno = last_lineno + 1
                            node_col_offset = 0
                            
                        decoratorCall = data[line_map[last_lineno - 1] + last_col_offset - 1 : line_map[node_lineno - 1] + node_col_offset].strip()
                        decorators.append(decoratorCall)
                        
                    decorator_tuple = tuple(decorators)

                if decorator_tuple in decorators_map:
                    amortized_decorator_name = decorators_map[decorator_tuple]
                else:
                    amortized_decorator_name = "test_custom_function_"+ str(decorator_index)
                    # map decorator tuple to amortized decorator name
                    decorators_map[decorator_tuple] = amortized_decorator_name

                    # map amortized decorator name to functions
                    amortized_decorators_map[amortized_decorator_name] = [decorator_tuple, []]
                    decorator_index += 1

                module_name = file.replace("/", ".")
                mod = import_module(module_name)
                function = getattr(mod, function_name)

                if option_parser.options.filter_tests:
                    if option_parser.options.filter_tests.search(mod.__name__ + "." + function_name) is None:
                        # Skip tests that don't match provided filter test regex
                        continue

                amortized_decorators_map[amortized_decorator_name][1].append([file, function_name])

    logger.info("Tests compiled.")

    return amortized_decorators_map

# generate_function_calls
#
# Arguments:
#
#    amortized_decorators_map : map of amortized function names to decorators
# and list of function tuples using them.
#
def generate_function_calls(amortized_decorators_map):
    file_content = ""

    # Prepare for custom functions.
    count = 0

    for cust_name, function_data in amortized_decorators_map.items():
        decorators = function_data[0]
        functions = function_data[1]
        function_len = len(functions)

        if function_len == 0:
            continue

        if len(decorators) != 0:
            # Add decorators.
            for decorator in decorators:
                file_content += f"{decorator}\n"

        # Add custom function.
        file_content += "def {}".format(cust_name)
        file_content += "(test_data_obj, cbFun, exception = None, *args, **kwargs):\n"
        file_content += "    wrapped_functions = [\n"

        function_idx = 0
        function_len = function_len - 1

        while function_idx < function_len:
            file_content += "        {},\n".format(functions[function_idx])
            function_idx += 1

        file_content += "        {}\n".format(functions[function_len])
        file_content += "    ]\n\n"

        if len(decorators) == 0:
            file_content += "    unwrap = False\n\n"
        else:
            file_content += "    unwrap = True\n\n"

        file_content += "    cbFun(test_data_obj, exception, wrapped_functions, unwrap, *args, **kwargs)\n\n"

    return file_content

def run_compilation():
    # Get all test file names
    logger.debug("getting all test names")
    test_files = find_files(script_directory + test_directory, mask = "test*.py", recurse=True)

    # Filter test file names.
    test_file_names = []
    for fname in test_files:
        if compiled_file_name.split(".")[0] not in fname:
            test_file_names.append(os.path.splitext(os.path.relpath(fname, script_directory))[0].replace(os.path.sep, "/"))

    # Sort test file names.
    test_file_names.sort()

    # File content holder.
    new_file_content = ""

    # Get imports.
    for test_file_name in test_file_names:
        new_file_content += "import {}\n".format(test_file_name.replace("/", "."))

    new_file_content += "\n"

    # Get all functions mapped under common decorators, as well as those to
    # call wrapped, individually.
    logger.debug("getting decorators and test functions")
    amortized_decorators_map = get_all_functions(test_file_names)

    # Add imports that decorator requiresments require.
    new_file_content += "import os\n\n"                    # import utilities

    # Add imports that we require.
    new_file_content += "import test_utils\n"              # import utilities
    new_file_content += "import dcgm_structs\n\n"          # import structs
    new_file_content += "from _test_helpers import *\n"    # import helpers

    # Add import of test_globals to set values for globals referenced in
    # test decorator arguments.
    new_file_content += "from test_globals import *\n\n"   # import test globals

    # Generate Amortized Decorator test calls.
    new_file_content += generate_function_calls(amortized_decorators_map) + "\n"

    # Write the compiled file.
    logger.debug("writing new file")
    with open(script_directory + test_directory + "/" + compiled_file_name, "w") as f:
        f.write(new_file_content)

    logger.debug(f"Compiled file [{compiled_file_name}] generated")

    return compiled_file_name

if __name__ == '__main__':
    run_compilation()
