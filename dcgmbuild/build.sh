#!/usr/bin/env bash

set -ex

DIR=$(dirname $(realpath $0))

docker buildx build --squash -t dcgmbuild $DIR
