#!/usr/bin/env bash

set -ex

DIR=$(dirname $(realpath $0))

docker build --force-rm --no-cache --rm -t dcgmmetabuild:latest -f Metabuild.dockerfile $DIR/
docker run \
    -v /var/run/docker.sock:/var/run/docker.sock:rw \
    -v "$DIR":"$DIR" \
    -w "$DIR" \
    -i \
    dcgmmetabuild:latest \
        bash -c "set -ex; \
            docker build -t dcgmbuild $DIR/; \
            docker-squash -t dcgmbuild dcgmbuild"

docker container prune -f
docker image prune -f

