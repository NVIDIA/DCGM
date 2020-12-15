#!/usr/bin/env bash
set -euo pipefail

DIR="$(dirname $(realpath $0))"
source "$DIR/_common.sh"

run_in_dcgmbuild "$@"

LONG_OPTS=help,deb:,rpm:

! PARSED=$(getopt --options= --longoptions=${LONG_OPTS} --name "${0}" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "Failed to parse arguments"
    exit 1
fi

eval set -- "${PARSED}"

function usage() {
   echo "$0 [--deb <debfile>|--rpm <rpmfile>]"
}

function verify_deb() {
    DEB="$1"
    CONTENTS="$(normalize_deb_name "$DEB")"
    diff "$CONTENTS" <(list_deb_contents "$1")
}

function verify_rpm() {
    RPM="$1"
    CONTENTS="$(normalize_rpm_name "$RPM")"
    diff "$CONTENTS" <(list_rpm_contents "$1")
}

while true; do
    case "$1" in
        --help)
            usage
            exit 0
            ;;
        --deb)
            verify_deb "$2"
            echo "Passed: $2"
            shift 2
            ;;
        --rpm)
            verify_rpm "$2"
            echo "Passed: $2"
            shift 2
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

