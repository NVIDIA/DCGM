#!/bin/bash
# Generate Validation config for diag-skus.yaml.in
#
# ./gen_validation.sh name comment
#
# name is the name of the prospective diag-skus.yaml.in record.
# comment is a record comment (use double quotes if necessary).
#
# N.B. The boilerplate is designed to pass nvvs validation, and some fields may
# not reflect the status of GPU being probed. This is a preliminary tool unti
# one that uses more stable bindings is developed in Python.
#
set -euxo pipefail # pipefail requires bash
NAME=$1
COMMENT=$2
APP_PATH=`python3 -c "import utils; print(utils.get_testing_framework_library_path())"`
nvidia-smi -q -i 0 | grep "Device Id" 2>&1 | tee /tmp/validate-$$.out > /tmp/gen_validation-$$.out
DEV_ID=`cat /tmp/validate-$$.out | sed -e "s/^.*: 0x//;s/\(....\).*/\\1/" | tr A-F a-f`
$APP_PATH/dcgmproftester12 -i 0 -t 1007 -d 7.5 -r 0.5 --cublas 2>&1 | tee /tmp/validate-$$.out >> /tmp/gen_validation-$$.out
FIELD_1007=`cat /tmp/validate-$$.out | grep ", dcgm " | sed -e "1,5d;s/^.*(//;s/ gflops.*//" | (while read n; do printf "%f\n" $n ; done) | awk '{S=S+$1; N=N+1} END {printf "%f\n", int(S/N)}' | sed -e "s/\..*//"`
$APP_PATH/dcgmproftester12 -i 0 -t 1005 -d 7.5 -r 0.5 --cublas 2>&1 | tee /tmp/validate-$$.out >> /tmp/gen_validation-$$.out
FIELD_1005=`cat /tmp/validate-$$.out | grep ", dcgm " | sed -e "1,5d;s/^.*(//;s/ GiB.*//" | (while read n; do printf "%f\n" $n ; done) | awk '{S=S+$1; N=N+1} END {printf "%f\n", int(S*1000/N)}' | sed -e "s/\..*//"`
nvidia-smi -q -i 0 | grep Power 2>&1 | tee /tmp/validate-$$.out >> /tmp/gen_validation-$$.out
POWER_LEVEL=`cat /tmp/validate-$$.out | grep "Default Power Limit.*W" | sed -e "s/^.*: //;s/ W//"`
nvidia-smi -q -i 0 | grep -B 2 -A 2 -i width 2>&1 | tee /tmp/validate-$$.out >> /tmp/gen_validation-$$.out
LINK_GENERATION=`cat /tmp/validate-$$.out | grep "Host Max" | sed -e "s/^.*: //"`
LINK_WIDTH=`cat /tmp/validate-$$.out | grep "  Max " | sed -e "s/^.*: //;s/x//"`
rm /tmp/validate-$$.out

FIELD_1007_75=$(expr $FIELD_1007 \* 3 / 4)
FIELD_1005_75=$(expr $FIELD_1005 \* 3 / 4)

echo "DEV_ID = $DEV_ID" >> /tmp/gen_validation-$$.out
echo "FIELD_1007 = $FIELD_1007" >> /tmp/gen_validation-$$.out
echo "FIELD_1007_75 = $FIELD_1007_75" >> /tmp/gen_validation-$$.out
echo "FIELD_1005 = $FIELD_1005" >> /tmp/gen_validation-$$.out
echo "FIELD_1005_75 = $FIELD_1005_75" >> /tmp/gen_validation-$$.out
echo "POWER_LEVEL = $POWER_LEVEL" >> /tmp/gen_validation-$$.out
echo "LINK_GENERATION = $LINK_GENERATION" >> /tmp/gen_validation-$$.out
echo "LINK_WIDTH = $LINK_WIDTH" >> /tmp/gen_validation-$$.out
cat <<EOF >/tmp/gen_nvalidation-$$.txt
  # $COMMENT
  # N.B. The boilerplate from which this is produced is designed to pass nvvs
  # validation, and some fields may not reflect the status of GPU being probed.
  - name: $NAME
    id: $DEV_ID
    targeted_power:
      is_allowed: true
      starting_matrix_dim: 1024
      target_power: $POWER_LEVEL
      use_dgemm: false
    targeted_stress:
      is_allowed: true
      use_dgemm: false
      #  Copied from sm stress target
      target_stress: $FIELD_1007_75
    sm_stress:
      is_allowed: true
      # dcgmproftester -t 1007 measures ~$FIELD_1007. Multiply by .75 to get $FIELD_1007_75
      target_stress: $FIELD_1007_75
      use_dgemm: false
    pcie:
      is_allowed: true
      h2d_d2h_single_pinned:
        min_pci_generation: 3.0
        min_pci_width: 16.0
      h2d_d2h_single_unpinned:
        min_pci_generation: 3.0
        min_pci_width: 16.0
    memory:
      is_allowed: true
    # l1cache_size_kb_per_sm: 164.0 ## This can vary and is an example
    diagnostic:
      is_allowed: true
    memory_bandwidth:
      is_allowed: true
      # dcgmproftester -t 1005 shows $FIELD_1005. Multiply by .75 to get $FIELD_1005_75
      minimum_bandwidth: $FIELD_1005_75
    pulse_test:
      is_allowed: true
EOF
cat /tmp/gen_nvalidation-$$.txt
rm /tmp/gen_nvalidation-$$.txt
