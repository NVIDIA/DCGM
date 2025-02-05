#!/bin/sh

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
