#!/usr/bin/env bash

set -ex -o nounset

die() {
    echo "$@" >&2
    exit 1
}

# Download a file
# Params: (url, output_file)
download_url(){
    local url=$1; shift
    local outputfile=$1; shift

    #
    # Cannot use curl until 7.68: https://github.com/curl/curl/pull/4623
    #
    # curl -fL "${url}" -o "${outputfile}"
    #
    wget -O "${outputfile}" -nv -t10 --unlink --no-check-certificate "${url}"
}
