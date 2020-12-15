#!/usr/bin/env bash
set -euo pipefail

DIR="$(dirname $(realpath $0))"
source "$DIR/_common.sh"

pushd "$DCGM_DIR"

LONG_OPTS=help,no-build
! PARSED=$(getopt --options= --longoptions=${LONG_OPTS} --name "${0}" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "Failed to parse arguments. Consult with $0 --help"
    exit 1
fi
eval set -- "${PARSED}"

NOBUILD=

function usage() {
USAGE=$(cat <<EOF
Usage:
$0 [--no-build]

Options:
  - --no-build: skip deleting _out and building packages

This script
  - deletes _out
  - creates a fresh DCGM build for all platforms
  - generates package lists for all platforms (both deb and rpm)
EOF
)
printf "%s" "$USAGE"
}

while true; do
    case "$1" in
        --help)
            usage
            exit 0
            ;;
        --no-build)
            NOBUILD=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unrecognized argument: $1"
            exit 1
    esac
done

function confirm_delete_out() {
    read -p "This will delete the contents of _out. Do you want to proceed? [y/N]? " -n 1 -r
    echo

    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        "$@"
    else
        echo "Did not receive user confirmation. Aborting"
        exit 1
    fi
}

if [[ -z "$NOBUILD" ]]; then
    confirm_delete_out
    echo 'Deleting _out'
    rm -rf _out
    echo 'amd64 aarch64 ppc64le' | xargs -d' ' -I'{}' -P3 bash -c 'set -ex; ./build.sh --release --deb --rpm --arch {}'
fi

ls _out/Linux-*-release/*rpm | egrep -v test | xargs -n1 ./scripts/verify_package_contents/generate_package_contents.sh --rpm
ls _out/Linux-*-release/*deb | egrep -v test | xargs -n1 ./scripts/verify_package_contents/generate_package_contents.sh --deb

popd
