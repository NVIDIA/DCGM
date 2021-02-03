# Introduction

NVIDIA Data Center GPU Manager (DCGM) is a suite of tools for managing and monitoring NVIDIA GPUs in cluster environments. It includes active health monitoring, comprehensive diagnostics, system alerts and governance policies including power and clock management. It is maintained by NVIDIA, and we are pleased to work with the open-source community.

# Architecture

As previously stated, DCGM is a tool used to monitor the health and telemetry of NVIDIAGPUs and their related components. We use a modular design to accomplish this task. The core of DCGM consists of:
 - APIs for interacting with the DCGM agent (nv-hostengine).
 - A cache for the known telemetry.
 - Modules that perform different functions:
    - Health: passively monitors fields that give a high-level picture of readiness to perform workload.
    - Configuration: assists in configuring the GPU(s) for usage.
    - Policy alerting: allows the user to set automatic reactions to different conditions of the GPU(s).
    - Job / process stats: a pre-packaged way to monitor how a process is using GPU(s).
    - Diagnostics: active stress tests for GPU(s) and related components.
    - NVSwitch: interacts with NVSwitches on the system.
    - Profiling: closed source\* module for providing profiling information about the usage of GPUs.

\*This module is not part of open-source DCGM.

# License

DCGM is licensed under the Apache 2.0 license.

# Communication Is Key

Early communication is essential for successfully contributing to DCGM, and the larger the change is, the more critical communication becomes. The same applies to the degree of complexity, as well as risk factors such as refactoring, behavioral changes, large changes, or other similarly impactful changes. Naturally, the most impactful and or risky a change is, the harder it is to plan as part of a release, and changes in behavior must be consistent with overall DCGM goals in order to be accepted and merged into the product as a whole. A good rule of thumb is that if there’s a significant amount of work involved, reach out to us early on.

# Testing Your Merge Request
All code must be thoroughly tested before a merge request can be accepted. It will be easiest to accept merge requests that have thorough testing coverage. If the test is fixing a bug, ideally the change will include a test that fails without the applied change and passes with the adopted change. Our unit tests are written using [Catch2](https://github.com/catchorg/Catch2) and are run when the code is compiled. For examples of existing unit tests, please look in dcgmlib/src/tests/. Our integration tests are written in python. Examples of existing integration tests can be found in testing/python/tests/. Changes will not be accepted if they do not have sufficient testing.

Additionally, the code must be tested to verify that it doesn’t cause any regressions in current DCGM testing. Current testing includes the integration tests and the unit tests, and regressions in either will have to be resolved before a merge request can be accepted.

# Merging Changes and Release
DCGM follows semver best practices, so the nature of the change will determine whether it is released in a major, minor, or patch release. A total of two minor releases are typically made per year, and some years one of those is replaced with a major release. Patch releases happen much more frequently. As a result, bug fixes are likely to be released quickly, and more complicated changes will have to fit into a release cycle for their target release.

Please note: we are currently snapshotting DCGM code for the public repo, so changes that are submitted will be merged first there and then become visible in the public repo later. At the time they are accepted, we should still be able to communicate a targeted release for the merge.

## Conditions that Can Cause a Change to Not Be Accepted
 - Inadequate testing / test regressions: all submissions must pass existing tests and include adequate testing coverage and functionality before they can be accepted.
 - Too complex: changes that are overly complicated, insufficiently decomposed, tightly coupled, or otherwise have a complexity that makes the code a burden to support will not be accepted.
 - Inconsistent methodology: examples of methodology that must be followed include DCGM’s modular design and versioned structs in our APIs. Changes that do not conform to these designs and their requirements cannot be adopted.
 - Insufficient time: changes that cannot be reviewed in time will be deferred to another release. Similarly, if a change is risky it must be adopted early enough in the release cycle to ensure that it hasn’t caused any regressions. Risky changes submitted later in a release cycle will also be deferred to the next release.
 - Incompleteness: for example, a change that updates C APIs, but doesn’t modify the Python bindings would be considered incomplete.
 - Changes that are not signed cannot be accepted.
 - Code must meet the DCGM coding best practices mentioned in dcgm_best_practices.md, as well as following the style that is checked for by the pre-commit hooks if properly configured. 

## The Mechanics of Contributing
1. Create a github issue which explains the change that you desire to make.
    1. Simple changes may not require this step, but generally speaking merge requests will be more productive and accepted faster if all parties have a clear understanding of why the change is needed or desired.
2. Clone the repository and configure the pre-commit hooks using install_git_hooks.sh.
3. Create a branch from main or a currently maintained branch.
4. Develop and test the desired change.
5. Submit an appropriately-signed merge request to github.
6. Correspond over github as needed until the merge request is accepted.

## Signing Your Work
The sign-off is a simple line at the end of the explanation for the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch. The rules are pretty simple: if you can certify the below (from developercertificate.org):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:
```Signed-off-by: Joe Smith <joe.smith@email.com>```

You need to use your real name to contribute (sorry, no pseudonyms or anonymous contributions).
If you set your `user.name` and `user.email` git configs, you can sign your commit automatically with `git commit -s`.

