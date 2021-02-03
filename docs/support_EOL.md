# Product Lifecycle
- Each DCGM release follows semver - X.Y.z.
- DCGM releases two (rev’ing X or Y) times in a year (linked to driver LLB lifecycle)
    - Each DCGM release is validated against ‘active’ driver branches.
    - The most recent of these is the current release, and the older is the maintenance release.
- DCGM is free to release patch (.z) releases every quarter (similar to TRDs)
- Each Y release is supported for a year (max overlap releases is therefore 2 active releases at any given point in time). After a year, these releases will be considered legacy releases.
- We don't follow the driver LTSB model (where drivers are supported up to 3 years). DCGM is user mode software, so based on customer feedback, it should be easier to upgrade.
- At arch boundaries, we are likely to rev X (to allow for breaking APIs).
- To be consistent with semver: breaking API changes are only made in major releases, new APIs are added in minor releases,and bug fixes are made in patch releases.

## Current Release
The current release is the primary place from which patch releases will be made. Fixes and small features which fit within semver best practices will be added to this branch and made available as patch releases.

## Maintenance Release
The maintenance release is still supported, and it is still eligible for being updated via patch releases. However, if a fix is already available in the current release, we prefer that customers upgrade instead of waiting for a patch release of the maintenance version. Similarly, if a bug has a simple workaround or a small impact, then the fix would normally not be prioritized for a maintenance release.

## Legacy Releases
Patch releases are not created for legacy releases.. In the event that new bugs are reported against releases that have already transitioned to legacy status, the bug will be fixed in either the current or the maintenance release.

