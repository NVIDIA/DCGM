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
# test_times.sh <test framework log file>
set +x +e
LOG=$1
grep "start time:" $LOG | awk '{ print $5, $6, $2 }' \
    | uniq -f 2 \
    > /tmp/test_times_$$_1
grep "end time:" $LOG | awk '{ print $5, $6, $2 }' \
    | uniq -f 2 \
    >/tmp/test_times_$$_2
join -j 3 /tmp/test_times_$$_1 /tmp/test_times_$$_2 >/tmp/test_times_$$_3

awk '{ print $1,
       substr($5,1,2) * 3600 + substr($5,4,2) * 60 + substr($5,7) - \
       substr($3,1,2) * 3600 - substr($3,4,2) * 60 - substr($3,7) \
     }' \
    < /tmp/test_times_$$_3 \
    | awk '{ printf "%11.6f %s\n",  $2, $1 }' \
    | sort -n \
    > /tmp/test_times_$$_4

TOTAL_TIME=`awk '{ total += $1 } END { print total }' < /tmp/test_times_$$_4`

awk -v total=$TOTAL_TIME \
    'BEGIN \
     { printf " Percentage  Total Pct.    Run Time Test Name\n" \
     } \
     { sum += $1*100/total; \
       printf "%11.4f %11.4f %11.4f %s\n", \
              $1*100/total, sum, $1, $2 \
     }' \
    < /tmp/test_times_$$_4

rm /tmp/test_times_$$_[1234]

