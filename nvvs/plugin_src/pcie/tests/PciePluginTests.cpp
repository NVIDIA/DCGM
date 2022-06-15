#include <catch2/catch.hpp>

#include <Pcie.h>
#include <PcieMain.h>
#include <sstream>

unsigned int failingGpuId    = DCGM_MAX_NUM_DEVICES;
unsigned int failingSwitchId = DCGM_MAX_NUM_SWITCHES;

TEST_CASE("Pcie: pcie_gpu_id_in_list")
{
    dcgmDiagPluginGpuList_t gpuInfo = {};
    gpuInfo.numGpus                 = 2;
    gpuInfo.gpus[0].gpuId           = 1;
    gpuInfo.gpus[1].gpuId           = 2;

    CHECK(pcie_gpu_id_in_list(0, gpuInfo) == false);
    for (unsigned int i = 3; i < 10; i++)
    {
        CHECK(pcie_gpu_id_in_list(i, gpuInfo) == false);
    }

    for (unsigned int i = 0; i < gpuInfo.numGpus; i++)
    {
        CHECK(pcie_gpu_id_in_list(gpuInfo.gpus[i].gpuId, gpuInfo) == true);
    }
}

/*
 * Spoof this dcgmlib function so we can control program execution
 */
dcgmReturn_t dcgmGetNvLinkLinkStatus(dcgmHandle_t handle, dcgmNvLinkStatus_v2 *linkStatus)
{
    memset(linkStatus, 0, sizeof(*linkStatus));
    linkStatus->numGpus                    = 1;
    linkStatus->gpus[0].entityId           = 0;
    linkStatus->gpus[0].linkState[0]       = DcgmNvLinkLinkStateUp;
    linkStatus->numNvSwitches              = 1;
    linkStatus->nvSwitches[0].entityId     = 0;
    linkStatus->nvSwitches[0].linkState[0] = DcgmNvLinkLinkStateUp;

    if (failingGpuId < DCGM_MAX_NUM_DEVICES)
    {
        linkStatus->gpus[0].entityId     = failingGpuId;
        linkStatus->gpus[0].linkState[0] = DcgmNvLinkLinkStateDown;
    }

    if (failingSwitchId < DCGM_MAX_NUM_SWITCHES)
    {
        linkStatus->nvSwitches[0].entityId     = failingSwitchId;
        linkStatus->nvSwitches[0].linkState[0] = DcgmNvLinkLinkStateDown;
    }

    return DCGM_ST_OK;
}

TEST_CASE("Pcie: pcie_check_nvlink_status")
{
    dcgmHandle_t handle = {};
    BusGrindGlobals bgGlobals;
    dcgmDiagPluginGpuList_t gpuInfo = {};
    gpuInfo.numGpus                 = 1;
    const unsigned int ourGpuId     = 2;
    gpuInfo.gpus[0].gpuId           = ourGpuId;

    bgGlobals.busGrind        = new BusGrind(handle, &gpuInfo);
    dcgmDiagResults_t results = {};
    pcie_check_nvlink_status(&bgGlobals, gpuInfo, handle);
    bgGlobals.busGrind->GetResults(&results);
    CHECK(results.numErrors == 0);

    // Make sure failing a GPU outside of our run doesn't trigger a failure
    memset(&results, 0, sizeof(results));
    failingGpuId = 3;
    delete bgGlobals.busGrind;
    bgGlobals.busGrind = new BusGrind(handle, &gpuInfo);
    pcie_check_nvlink_status(&bgGlobals, gpuInfo, handle);
    bgGlobals.busGrind->GetResults(&results);
    CHECK(results.numErrors == 0);

    // Verify we do trigger a failure with our GPU
    memset(&results, 0, sizeof(results));
    failingGpuId = ourGpuId;
    delete bgGlobals.busGrind;
    bgGlobals.busGrind = new BusGrind(handle, &gpuInfo);
    pcie_check_nvlink_status(&bgGlobals, gpuInfo, handle);
    bgGlobals.busGrind->GetResults(&results);
    CHECK(results.numErrors == 1);

    // Make sure we can fail with switches
    failingGpuId    = DCGM_MAX_NUM_DEVICES;
    failingSwitchId = 1; // id doesn't matter
    memset(&results, 0, sizeof(results));
    delete bgGlobals.busGrind;
    bgGlobals.busGrind = new BusGrind(handle, &gpuInfo);
    pcie_check_nvlink_status(&bgGlobals, gpuInfo, handle);
    bgGlobals.busGrind->GetResults(&results);
    std::stringstream buf;
    for (unsigned int i = 0; i < results.numErrors; i++)
    {
        buf << "Error " << i << ": " << results.errors[i].msg << "\n";
    }

    INFO(buf.str());
    CHECK(results.numErrors == 1);
}
