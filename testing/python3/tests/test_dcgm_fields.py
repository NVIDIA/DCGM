#
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import dcgm_fields
import warnings


def test_deprecated_aliases_resolve_to_canonical_field_ids():
    """Verify each deprecated field name resolves to the same ID as its canonical name."""
    if dcgm_fields.DCGM_DEPRECATED == 0:
        return

    for old, new in dcgm_fields._DEPRECATED_ALIASES.items():
        assert getattr(dcgm_fields, old) == getattr(dcgm_fields, new), \
            f"{old} should equal {new}"


def test_deprecated_alias_targets_exist():
    """Every canonical name in _DEPRECATED_ALIASES exists as a real attribute."""
    for old, new in dcgm_fields._DEPRECATED_ALIASES.items():
        assert hasattr(dcgm_fields, new), \
            f"deprecated alias {old} points to missing canonical {new}"


def test_unknown_attribute_raises():
    """Accessing a non-existent attribute raises AttributeError."""
    try:
        dcgm_fields.DCGM_FI_THIS_DOES_NOT_EXIST
    except AttributeError:
        return
    assert False, "expected AttributeError"


def test_deprecated_access_emits_warning():
    """Accessing a deprecated alias emits a DeprecationWarning pointing at the new name."""
    if dcgm_fields.DCGM_DEPRECATED == 0:
        return
    old, new = next(iter(dcgm_fields._DEPRECATED_ALIASES.items()))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(dcgm_fields, old)

    deprecations = [
        w for w in caught
        if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecations) == 1, \
        f"expected 1 DeprecationWarning, got {len(deprecations)}"
    assert old in str(deprecations[0].message), f"warning should mention {old}"
    assert new in str(deprecations[0].message), f"warning should mention {new}"
