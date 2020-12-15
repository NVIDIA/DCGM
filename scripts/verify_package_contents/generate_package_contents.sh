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

function generate_deb_contents() {
    DEB="$1"
    CONTENTS="$(normalize_deb_name "$DEB")"
    list_deb_contents "$DEB" > "$CONTENTS"
}

function generate_rpm_contents() {
    RPM="$1"
    CONTENTS="$(normalize_rpm_name "$RPM")"
    list_rpm_contents "$RPM" > "$CONTENTS"
}

while true; do
    case "$1" in
        --help)
            usage
            exit 0
            ;;
        --deb)
            generate_deb_contents "$2"
            echo "Generated contents for: $2"
            shift 2
            ;;
        --rpm)
            generate_rpm_contents "$2"
            echo "Generated contents for: $2"
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
