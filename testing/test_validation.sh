#!/bin/sh
# Test Validation config for diag-skus.yaml.in
#
# ./test_validation.sh config
#
# config is the name a file containing the config entry.
#
CONFIG=$1
APP_PATH=`python3 -c "import utils; print(utils.get_testing_framework_library_path())"`
killall nv-hostengine
NVVS_BIN_PATH=`pwd`/apps/nvvs $APP_PATH/nv-hostengine
$APP_PATH/dcgmi diag -i 0 -r 1 -c $CONFIG
$APP_PATH/dcgmi diag -i 0 -r 2 -c $CONFIG
$APP_PATH/dcgmi diag -i 0 -r 3 -c $CONFIG
$APP_PATH/dcgmi diag -i 0 -r 4 -c $CONFIG
killall nv-hostengine
