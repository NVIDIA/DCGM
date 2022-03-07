# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
# Sample script to test python bindings for DCGM

import dcgm_structs
import dcgm_agent_internal
import dcgm_agent
import logger
import test_utils
import dcgm_fields
import apps
import time

@test_utils.run_only_on_linux()
@test_utils.run_only_on_bare_metal()
def test_nv_hostengine_app():
    """
    Verifies that nv-hostengine can be lauched properly and 
    can run for whatever timeout it's given in seconds
    """

    # Start nv-hostengine and run for 15 seconds
    nvhost_engine = apps.NvHostEngineApp()
    nvhost_engine.start(timeout=15)  

    # Getting nv-hostenging process id
    pid = nvhost_engine.getpid()
        
    # Cleanning up
    time.sleep(5)
    nvhost_engine.terminate()
    nvhost_engine.validate()

    logger.debug("nv-hostengine PID was %d" % pid)


@test_utils.run_only_on_linux()
@test_utils.run_only_on_bare_metal()
def test_dcgmi_app():
    """
    Verifies that dcgmi can be lauched properly with 
    2 parameters at least
    """

    # Start dcgmi and start collecting data from nv-hostengine
    dcgmi_app = apps.DcgmiApp(["127.0.0.1", "0"])
    dcgmi_app.start()
        
    # Getting nv-hostenging process id
    pid = dcgmi_app.getpid()


    # Cleanning up dcgmi run
    time.sleep(3)
    dcgmi_app.terminate()
    dcgmi_app.validate()   

    logger.debug("dcgmi PID was %d" % pid)

@test_utils.run_only_on_linux()
@test_utils.run_only_on_bare_metal()
@test_utils.run_only_with_all_supported_gpus()
@test_utils.skip_blacklisted_gpus(["GeForce GT 640"])
def test_dcgm_unittests_app(*args, **kwargs):
    """
    Runs the testdcgmunittests app and verifies if there are any failing tests
    """

    # Run testsdcgmunittests 
    unittest_app = apps.TestDcgmUnittestsApp()
    unittest_app.run(1000)
        
    # Getting testsdcgmunittests process id
    pid = unittest_app.getpid()
    logger.debug("The PID of testdcgmunittests is %d" % pid)

    # Cleanning up unittests run
    unittest_app.wait()
    unittest_app.validate()
    assert unittest_app._retvalue == 0, "Unittest failed with return code %s" % unittest_app._retvalue

