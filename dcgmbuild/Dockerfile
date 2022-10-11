ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY --chown=root:root scripts_host /root/.build/scripts_host
RUN set -ex; find /root/.build/scripts_host -iname '*.sh' -exec chmod a+x {} \;
WORKDIR /root/.build/scripts_host
RUN bash -c 'set -ex -o pipefail; find . -iregex "^\.\/[0-9]+_.*" | sort | xargs -n1 -I {} bash -c "{} || exit 255"'

COPY --chown=root:root scripts /root/.build/scripts/
RUN set -ex; find /root/.build/scripts -iname '*.sh' -exec chmod a+x {} \;
WORKDIR /root/.build/scripts

RUN bash -c 'set -ex -o pipefail; export TARGET=x86_64-linux-gnu; find . -iregex "^\.\/[0-9]+_.*" | sort | xargs -n1 -I {} bash -c "{} || exit 255"'
RUN bash -c 'set -ex -o pipefail; export TARGET=aarch64-linux-gnu; find . -iregex "^\.\/[0-9]+_.*" | sort | xargs -n1 -I {} bash -c "{} || exit 255"'
RUN bash -c 'set -ex -o pipefail; export TARGET=powerpc64le-linux-gnu; find . -iregex "^\.\/[0-9]+_.*" | sort | xargs -n1 -I {} bash -c "{} || exit 255"'

RUN mkdir -p /workspaces
RUN chmod a+rw /workspaces

RUN rm -rf /root/.build
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /root
