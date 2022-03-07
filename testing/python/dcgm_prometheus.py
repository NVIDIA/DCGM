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
from DcgmReader import DcgmReader
from common import dcgm_client_cli_parser as cli
import dcgm_fields
import time
import logging
import os
import argparse
import sys
import signal

try:
    from prometheus_client import start_http_server, Gauge
except ImportError:
    pass
    logging.critical("prometheus_client not installed, please run: \"pip install prometheus_client\"")
    sys.exit(3)

DEFAULT_FIELDS = [
    dcgm_fields.DCGM_FI_DEV_PCI_BUSID, #Needed for plugin_instance
    dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
    dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
    dcgm_fields.DCGM_FI_DEV_SM_CLOCK,
    dcgm_fields.DCGM_FI_DEV_GPU_UTIL,
    dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING,
    dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
    dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
    dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_TOTAL,
    dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_TOTAL,
    dcgm_fields.DCGM_FI_DEV_FB_TOTAL,
    dcgm_fields.DCGM_FI_DEV_FB_FREE,
    dcgm_fields.DCGM_FI_DEV_FB_USED,
    dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
    dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TOTAL,
    dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
    dcgm_fields.DCGM_FI_DEV_POWER_VIOLATION,
    dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION,
    dcgm_fields.DCGM_FI_DEV_XID_ERRORS,
    dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL,
]

class DcgmPrometheus(DcgmReader):
    ###########################################################################
    def __init__(self):
        #Have DCGM update its watches twice as fast as our update interval so we don't get out of phase by our update interval
        updateIntervalUsec = (1000000 * g_settings['prometheusPublishInterval']) / 2
        #Add our PID to our field group name so we can have multiple instances running
        fieldGroupName = 'dcgm_prometheus_' + str(os.getpid())

        DcgmReader.__init__(self, ignoreList=g_settings['ignoreList'], fieldIds=g_settings['publishFieldIds'],
                updateFrequency=updateIntervalUsec,
                fieldGroupName=fieldGroupName, hostname=g_settings['dcgmHostName'])
        self.m_existingGauge = {}

    ###########################################################################
    '''
    This function is implemented from the base class : DcgmReader. It converts each
    field / value from the fvs dictionary to a gauge and publishes the gauge to the
    prometheus client server.

    @params:
    fvs : The fieldvalue dictionary that contains info about the values of field Ids for each gpuId.
    '''
    def CustomDataHandler(self,fvs):
        if not self.m_existingGauge:
            self.SetupGauges()

        for fieldId in self.m_publishFieldIds:

            if fieldId in self.m_dcgmIgnoreFields:
                continue

            g = self.m_existingGauge[fieldId]

            for gpuId in fvs.keys():
                gpuFv = fvs[gpuId]
                val = gpuFv[fieldId][-1]

                #Skip blank values. Otherwise, we'd have to insert a placeholder blank value based on the fieldId
                if val.isBlank:
                    continue

                gpuUuid = self.m_gpuIdToUUId[gpuId]
                gpuBusId = self.m_gpuIdToBusId[gpuId]
                gpuUniqueId = gpuUuid if g_settings['sendUuid'] else gpuBusId

                # pylint doesn't find the labels member for Gauge, but it exists. Ignore the warning
                g.labels(gpuId, gpuUniqueId).set(val.value) # pylint: disable=no-member

                logging.debug('Sent GPU %d %s %s = %s' % (gpuId, gpuUniqueId, self.m_fieldIdToInfo[fieldId].tag, str(val.value)))

    ###############################################################################
    '''
    NOTE: even though some fields are monotonically increasing and therefore fit the mold to be
    counters, all are published as gauges so that DCGM is the sole authority on the state of the
    system, preventing problems around down times, driver reboots, and the unlikely event of
    flashing the inforom.
    For specific information about which fields monotonically increase, see the API guide or
    dcgm_fields.h
    '''
    def SetupGauges(self):
        for fieldId in self.m_publishFieldIds:
            if fieldId in self.m_dcgmIgnoreFields:
                continue

            uniqueIdName = 'GpuUuid' if g_settings['sendUuid'] else 'GpuBusID'

            fieldTag = self.m_fieldIdToInfo[fieldId].tag
            self.m_existingGauge[fieldId] = Gauge("dcgm_"+fieldTag,'DCGM_PROMETHEUS',['GpuID', uniqueIdName])

    ###############################################################################
    '''
    Scrape the fieldvalue data and publish. This function calls the process function of
    the base class DcgmReader.
    '''
    def Scrape(self, data=None):
        return self.Process()

    ###############################################################################
    def LogBasicInformation(self):
        # Reconnect causes everything to get initialized
        self.Reconnect()

        logging.info('Started prometheus client')

        fieldTagList = ''
        for fieldId in self.m_publishFieldIds:
            if fieldId in self.m_dcgmIgnoreFields:
                continue

            if fieldTagList == '':
                fieldTagList = self.m_fieldIdToInfo[fieldId].tag
            else:
                fieldTagList = fieldTagList + ", %s" % (self.m_fieldIdToInfo[fieldId].tag)

        logging.info("Publishing fields: '%s'" % (fieldTagList))

    ###############################################################################
    def LogError(self, msg):
        logging.error(msg)

    ###############################################################################
    def LogInfo(self, msg):
        logging.info(msg)

###############################################################################
def exit_handler(signum, frame):
    g_settings['shouldExit'] = True

###############################################################################
def main_loop(prometheus_obj, publish_interval):
    try:
        while True:
            prometheus_obj.Scrape(prometheus_obj)
            time.sleep(publish_interval)

            if g_settings['shouldExit'] == True:
                prometheus_obj.LogInfo('Received a signal...shutting down')
                break
    except KeyboardInterrupt:
        print("Caught CTRL-C. Exiting")

###############################################################################
def initialize_globals():
    '''
    Name of the host.
    '''
    global g_settings
    g_settings = {}

    g_settings['shouldExit'] = False

    '''
    List of the ids that are present in g_settings['publishFieldIds'] but ignored for watch.
    '''
    g_settings['ignoreList'] = [dcgm_fields.DCGM_FI_DEV_PCI_BUSID, ]

    '''
    Those are initialized by the CLI parser. We only list them here for clarity.
    '''
    for key in [
            'dcgmHostName',
            'prometheusPort',
            'prometheusPublishInterval',
            'publishFieldIds',
    ]:
        g_settings[key] = None

###############################################################################
def parse_command_line():
    parser = cli.create_parser(
        name='Prometheus',
        field_ids=DEFAULT_FIELDS,
    )

    cli.add_custom_argument(parser, '--send-uuid', dest='send_uuid', default=False,
                            action='store_true', help='Send GPU UUID instead of bus id')

    args = cli.run_parser(parser)
    field_ids = cli.get_field_ids(args)
    numeric_log_level = cli.get_log_level(args)

    # Defaults to localhost, so we need to set it to None
    if args.embedded:
        g_settings['dcgmHostName'] = None
    else:
        g_settings['dcgmHostName'] = args.hostname

    g_settings['prometheusPort'] = args.publish_port

    g_settings['prometheusPublishInterval'] = args.interval

    logfile = args.logfile

    g_settings['publishFieldIds'] = field_ids

    g_settings['sendUuid'] = args.send_uuid

    if logfile != None:
        logging.basicConfig(level=numeric_log_level, filename=logfile, filemode='w+', format='%(asctime)s %(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=numeric_log_level, filemode='w+', format='%(asctime)s %(levelname)s: %(message)s')

###############################################################################
def initialize_signal_handlers():
    signal.signal(signal.SIGINT, exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)

###############################################################################
def main():
    initialize_globals()

    initialize_signal_handlers()

    parse_command_line()

    prometheus_obj = DcgmPrometheus()

    #start prometheus client server.
    start_http_server(g_settings['prometheusPort'])

    prometheus_obj.LogBasicInformation()

    main_loop(prometheus_obj, g_settings['prometheusPublishInterval'])

    prometheus_obj.Shutdown()

if __name__ == '__main__':
    main()
