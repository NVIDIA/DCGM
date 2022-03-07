#!/usr/bin/env bash
# This is not a script. It's shared functionality loaded by the
# verify_package_contents scripts

DCGM_DIR="${DCGM_DIR:-$(realpath $DIR/../..)}"
RELDIR="$(echo $DIR | rev | cut -d'/' -f-2 | rev)"
SCRIPT_NAME="$(basename $0)"

# prints the DCGM version escaped for sed
function get_escaped_version() {
    # The sed script below:
    # - extracts the version from the project line
    # replaces "." with "\."
    egrep 'project\(dcgm' < "$DCGM_DIR/CMakeLists.txt" | sed 's/project(.* \([0-9\.]*\).*)/\1/;s/\./\\./g'
}

ESCAPED_VERSION=$(get_escaped_version)

function run_in_dcgmbuild() {
    if ! [[ "${DCGM_BUILD_INSIDE_DOCKER:-}" = 1 ]]; then
        "$DCGM_DIR/intodocker.sh" -- bash -c "$RELDIR/$SCRIPT_NAME $*"
        exit $?
    fi

    if ! [[ -f "$DCGM_DIR/intodocker.sh" ]]; then
        echo "Could not find intodocker.sh. Make sure DCGM_DIR is properly configured"
    fi
}

function normalize_deb_name() {
    # The sed command below does three things:
    # - replaces the version (e.g. 2.2.0) with the string "VERSION"
    # - prefixes the filename with this script's directory
    # - adds the extension ".txt" at the end
    basename "$1" | sed -E "s/(datacenter-gpu-manager(-config)?)_([^_]*)/\1_VERSION/;s~^~$DIR/~;s/\$/.txt/"
}

function list_deb_contents() {
    dpkg -c "$1" | awk '{print $6}' | sed "s/$ESCAPED_VERSION/\${VERSION}/" | LC_ALL=C sort
}

function normalize_rpm_name() {
    # The sed command below does three things:
    # - replaces the version (e.g. 2.2.0) with the string "VERSION"
    # - prefixes the filename with this script's directory
    # - adds the extension ".txt" at the end
    basename "$1" | sed -E "s/(datacenter-gpu-manager(-config)?)-([^-]*)/\1_VERSION/;s~^~$DIR/~;s/\$/.txt/"
}

function list_rpm_contents() {
    rpm -qlp "$1" 2>/dev/null | sed "s/$ESCAPED_VERSION/\${VERSION}/" | LC_ALL=C sort
}
