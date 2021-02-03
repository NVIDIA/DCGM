# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import os
import sys
import subprocess
import time
import multiprocessing
import functools
import copy
from distutils.version import LooseVersion # pylint: disable=import-error,no-name-in-module
from multiprocessing.pool import ThreadPool

import test_utils
import utils
import logger

PYLINT_MIN_VERSION = '1.5.4'

# create a file with this suffix for every python file that passes linting
PYLINT_PASS_SUFFIX = 'pylint-passed'

# directories that should not be linted
PYLINT_BLACKLISTED_DIRS = ['_out']

# filenames that should not be linted
PYLINT_BLACKLISTED_FILENAMES = ['topology_gen_LR.py', 'topology_gen.py']

def _pylint_pass_filepath(filepath):
    '''Given a filepath, construct the filepath for the hidden "pylint-passed" file'''
    dirPath, file = os.path.split(filepath)
    return os.path.join(dirPath, '.%s.%s' % (file, PYLINT_PASS_SUFFIX))
    
def _should_lint_file(filepath):
    # only lint python files
    if not filepath.endswith('.py'):
        return False
    
    lintFilepath = _pylint_pass_filepath(filepath)
    
    # lint if they haven't been linted before
    if not os.path.isfile(lintFilepath):
        return True
    
    fileModTime = os.path.getmtime(filepath)
    fileLintTime = os.path.getmtime(lintFilepath)
    
    # lint if the file has changed since it was linted last
    if fileModTime > fileLintTime:
        return True
    else:
        return False
    
def _remove_blacklisted_dirs(dirnames):
    for dir in PYLINT_BLACKLISTED_DIRS:
        try:
            blacklistDirIndex = dirnames.index('_out')
        except ValueError:
            pass
        else:
            del dirnames[blacklistDirIndex]

def _pylint_root_dir():
    '''
    Return the root directory to lint all python files under.

    This used to run at the dcgm level (../..). Now it runs from the current directory
    to avoid running python2 lint against python3 and vice versa.
    
    This attempts to walk up the current directory until the development "dcgm" 
    directory is found so that it can:
      1. lint all files included in DCGM
      2. track which files have passed linting, speeding up the process next time
    
    If the dev environment is not found then it simply will have to make do 
    with only linting files at or below the current directory.
    '''
    curDir = os.path.dirname(os.path.realpath(__file__))
    pylintRoot = curDir
    return curDir

def _get_py_files_to_lint(rootDir):
    pythonFiles = []
    for dirpath, dirnames, filenames in os.walk(rootDir, topdown=True): 
        _remove_blacklisted_dirs(dirnames)
        
        for file in filenames:
            #Don't lint files that are blacklisted
            if file in PYLINT_BLACKLISTED_FILENAMES:
                continue

            fullFile = os.path.join(rootDir, dirpath, file)
            if _should_lint_file(fullFile):
                pythonFiles.append(fullFile)
                
    return pythonFiles

def _lint_file(filepath, pylintExecFile, rcfile, pythonPath):
    '''
    Returns 1 if there were errors or 0 if there were none. Logs errors.
    If this file had no errors than create/update its matching PYLINT_PASS_SUFFIX file.
    '''
    env = os.environ
    env['PYTHONPATH'] = env.get('PYTHONPATH', '') + ':' + pythonPath
    
    proc = subprocess.Popen([sys.executable, pylintExecFile, '--rcfile=%s' % rcfile, filepath], 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=env)
    pylintMsgs, _ = proc.communicate()

    pylintMsgs = ''.join("%s\n" % line for line in filter(lambda x: "Using config file" not in x
                         and "No config file found" not in x, pylintMsgs.split('\n'))).strip()
    
    if pylintMsgs:
        logger.error(pylintMsgs)
        return 1
    else:
        # update the hidden pylint-passed file for this file 
        pylintFile = _pylint_pass_filepath(filepath)
        with open(pylintFile, 'a'):
            os.utime(pylintFile, None)
        return 0

def _is_pylint_installed():
    try:
        import pylint
    except ImportError:
        return False
    else:
        return True

def _pylint_version():
    import pylint
    return pylint.__version__

def _create_python_path_env_var(pythonFilepaths):
    '''
    Given a list of python files, construct a string that can be used
    as a PYTHONPATH environment variable to allow importing of all the python files.
    
    This function returns a non-redundant PYTHONPATH where every directory in it 
    is the top directory of a python module.  This is done by assuming that of all the 
    filepaths under a filepath A in "pythonFilepaths" are part of a python module 
    that starts at A's directory.  This means that everything under A's directory must have
    __init__.py files.
    '''
    uniqueDirs = set(os.path.split(filepath)[0] for filepath in pythonFilepaths)
    
    # We can't just add every unique directory from pythonFilepaths to the PYTHONPATH 
    # since this would make it too long for the OS and result in 
    # "OSError: [Errno 7] Argument list too long"
    
    # construct a tree with a dict where every node represents a directory
    # and stores whether or not that dir level has python files
    blankNode = {
        'contains-py-files': False,
        'sub-dirs': {}
    }
    
    dirTree = copy.deepcopy(blankNode)
    curTreeNode = dirTree
    
    for dirPath in uniqueDirs:
        
        # for each dir path, start inserting dir nodes to the dir tree at the root
        curTreeNode = dirTree
        
        dirs = dirPath.split(os.sep)
        for i, dir in enumerate(dirs):
            
            if dir not in curTreeNode['sub-dirs']:
                curTreeNode['sub-dirs'][dir] = copy.deepcopy(blankNode)
                
            # end of the directory path, this dir has a python file
            if i == len(dirs) - 1:
                curTreeNode['sub-dirs'][dir]['contains-py-files'] = True
                
            # iterate to next level
            curTreeNode = curTreeNode['sub-dirs'][dir]
            
    # extract the highest level dirs that have python files.  We don't care
    # about any dirs below them that have python files since we only need to specify 
    # the top dir for PYTHONPATH as long as intermediate dirs all have an __init__.py
    # file (this is assumed to be true)
    curTreeNode = dirTree
    
    # prime the toVisitQueue with the root directory to start the Breadth-First-Search
    toVisitQueue = [{
        'dirPath': os.path.sep,
        'dirNode': dirTree
    }]
    curDirPath = ''
    topPyDirs = []
    
    while len(toVisitQueue) > 0:
        
        curDirPath  = toVisitQueue[0]['dirPath']
        curTreeNode = toVisitQueue[0]['dirNode']
        toVisitQueue.pop(0)
        
        if curTreeNode['contains-py-files']:
            topPyDirs.append(curDirPath)
            continue    # don't recurse, this is a top dir with py files
            
        
        toVisitQueue += [
            {
                'dirPath': os.path.join(curDirPath, subDir), 
                'dirNode': subDirNode
            } 
            for subDir, subDirNode in curTreeNode['sub-dirs'].items()
        ]
        
    return ':'.join(topPyDirs)

def pylint_dcgm_files():
    '''
    This lints python files, logs errors, and the returns the number of files
    that had pylint errors.
    All python files IN YOUR DEVELOPMENT ENVIRONMENT that have changed 
    since the last time that they were linted will have pylint run against them.  
    If this script is not somewhere underneath the development "dcgm" folder then 
    only python files in the current directory and below will be linted.
    '''
    
    if not _is_pylint_installed():
        raise EnvironmentError('pylint is not installed.  Version %s or later is needed.  ' % PYLINT_MIN_VERSION + 
                               'You can install it with "sudo pip install pylint".  ' + 
                               'If you do not have pip (python package manager) you can ' +
                               'install it with "sudo apt-get install python-pip.')
        
    pylintVer = LooseVersion(_pylint_version())
    if pylintVer < LooseVersion(PYLINT_MIN_VERSION):
        raise EnvironmentError('pylint version %s is required but the currently installed version is %s.  ' 
                               % (PYLINT_MIN_VERSION, pylintVer) +
                               'Please update pylint.')
         
    curDir = os.path.dirname(os.path.realpath(__file__))
    pylintRoot = _pylint_root_dir()
    pythonFilepaths = _get_py_files_to_lint(pylintRoot)
    pythonPath = _create_python_path_env_var(pythonFilepaths)
    
    if len(pythonFilepaths) == 0:
        logger.info('No changed python files to lint')
        return 0
    
    logger.info('Starting to lint %d python files under "%s"...' 
                % (len(pythonFilepaths), pylintRoot))
    
    try:
        workerCount = multiprocessing.cpu_count()
    except NotImplementedError:
        workerCount = 8
        
    pool = ThreadPool(workerCount)
    
    pylintrcFilepath = os.path.join(curDir, 'pylintrc')
    if not os.path.isfile(pylintrcFilepath):
        raise EnvironmentError('Could not find pylintrc file, expected to find it in the current directory')
    
    try:
        pylintExecFile = subprocess.check_output(['which', 'pylint']).strip()
    except subprocess.CalledProcessError:
        pylintExecFile = subprocess.check_output(['which', 'pylint2']).strip()
    
    pylintFn = functools.partial(_lint_file, 
                                 pylintExecFile=pylintExecFile, 
                                 rcfile=pylintrcFilepath, 
                                 pythonPath=pythonPath)
    
    fileErrorCount = sum(pool.map(pylintFn, pythonFilepaths))
    
    logger.debug('Done linting python files')
    if fileErrorCount == 0:
        logger.info('SUCCESS: pylint found no errors')
    else:
        logger.error('Found FATAL ERRORS in %d python files. You MUST fix these before checking in code!  ' 
                     % fileErrorCount
                     + 'For further explanations of error messages try `pylint --list-msgs` or ' 
                     + '`pylint --help-msg=SOME_ERROR_CODE`.  The error code will be in the error messages.')
    return fileErrorCount

def clear_lint_artifacts():
    _clear_lint_artifacts(_pylint_root_dir())
    
def _clear_lint_artifacts(pylintRoot):
    artifactCount = 0
    for dirpath, dirnames, filenames in os.walk(pylintRoot, topdown=True): 
        _remove_blacklisted_dirs(dirnames)
        
        for file in filenames:
            fullFile = os.path.join(pylintRoot, dirpath, file)
            if fullFile.endswith('.' + PYLINT_PASS_SUFFIX):
                os.remove(fullFile)
                artifactCount += 1
    
    logger.debug('%d lint file artifacts were removed' % artifactCount)
                
    
