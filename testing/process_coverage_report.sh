#!/bin/sh
#
# This script process the genhtml-created coverage files with the
# process-coverage.awk script to allow filtering those below a given limit
# (currently less than 70% functional coverage  by default -- we ignore the
# line coverage with line_limit=0.0).
#
for f in `find . -name index.html`
do
    awk -f ./process_coverage_report.awk -v path=$f -v line_limit=0.0 <$f
done
