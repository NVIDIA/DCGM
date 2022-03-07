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
# test the performance of DCGM
import time
import datetime
import json
import os
import sys
import pkgutil
import operator
import math
from collections import defaultdict
from distutils.version import LooseVersion # pylint: disable=import-error,no-name-in-module

import dcgm_structs
import dcgm_agent_internal
import dcgm_fields
import pydcgm
import logger
import test_utils
import utils
import stats
import option_parser

REQ_MATPLOTLIB_VER = '1.5.1'
def isReqMatplotlibVersion():
    return 'matplotlib' in sys.modules and \
        LooseVersion(matplotlib.__version__) >= LooseVersion(REQ_MATPLOTLIB_VER)

try:
    import matplotlib
except ImportError:
    logger.info('Graphs for performance tests will be missing since "matplotlib" is not installed')
else:
    if not isReqMatplotlibVersion():
        logger.info('Graphs for performance tests will be missing since "matplotlib" version '
                  + '%s is less than the required version %s' % (matplotlib.__version__, REQ_MATPLOTLIB_VER))
    else:
        # must do this before importing matplotlib.pyplot 
        # to use backend that does not require X11 display server running
        matplotlib.use('AGG')
        from matplotlib import pyplot as plt
        plt.style.use('ggplot')                 # pylint: disable=no-member

# duration to gather data for when we limit the record count for DCGM to store  
# This time needs to be long enough for memory usage to level off.
BOUNDED_TEST_DURATION = 40

class MetadataTimeseries(object):
    
    def __init__(self):
        self.timestamps = []
        self.fieldVals = defaultdict(list)
        self.fieldGroupVals = defaultdict(list)
        self.allFieldsVals = []
        self.processVals = []
        
class CpuTimeseries(object):
    
    def __init__(self):
        self.timestamps = []
        self.cpuInfo = []
        
def _plotFinalValueOrderedBarChart(points, title, ylabel, filenameBase, topValCount=20):
    '''points are (x, y) pairs where x is the xlabel and y is the height of the var'''
    if not isReqMatplotlibVersion():
        logger.info('not generating ordered bar chart since "matplotlib" is not the required version')
        return
    if logger.log_dir is None:
        logger.info('not generating ordered bar chart since logging is disabled')
        return
    
    width = 4
     
    topSortedPts = sorted(points, key=operator.itemgetter(1), reverse=True)[:topValCount]
    xTickLabels = [point[0] for point in topSortedPts]
    x = list(range(0, len(topSortedPts)*width, width))
    y = [point[1] for point in topSortedPts]
     
    ax = plt.subplot(1, 1, 1)
    ax.set_title(title)
    plt.xlabel('ID')
    plt.ylabel(ylabel)
     
    ax.set_xticklabels(xTickLabels)
    ax.set_xticks([tick + width/2. for tick in x])
     
    ax.bar(left=x, height=y, width=4)
     
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    plt.gcf().set_size_inches(9, 6)
     
    filename = 'OrderedBar-%s-%s.png' % ('-'.join(title.split()), filenameBase)
    figName = os.path.join(logger.log_dir, filename)
    plt.savefig(figName)
    plt.close()
    logger.info('ordered bar chart for %s saved in %s' % (title, utils.shorten_path(figName)))
        
def _plot_metadata(x, yLists, title, ylabel, plotNum):
    ax = plt.subplot(2, 2, plotNum)
    ax.set_title(title)
     
    for y in yLists:
        ax.plot(x, y)
         
    plt.xlabel('seconds since start')
    plt.ylabel(ylabel)

def _generate_metadata_line_charts(metadataTSeries, ylabel, title):
    if not isReqMatplotlibVersion():
        logger.info('Not generating memory usage plots since "matplotlib" is not the required version')
        return
    if logger.log_dir is None:
        logger.info('Not generating memory usage plots since logging is disabled')
        return
    
    if metadataTSeries.allFieldsVals:
        _plot_metadata(x=metadataTSeries.timestamps, 
                       yLists=[metadataTSeries.allFieldsVals], 
                       title='%s for all fields' % title, 
                       ylabel=ylabel,
                       plotNum=1)
    
    if metadataTSeries.fieldVals:
        _plot_metadata(x=metadataTSeries.timestamps, 
                       yLists=list(metadataTSeries.fieldVals.values()), 
                       title='%s for fields' % title, 
                       ylabel=ylabel,
                       plotNum=2)
    
    if metadataTSeries.processVals:
        _plot_metadata(x=metadataTSeries.timestamps, 
                       yLists=[metadataTSeries.processVals], 
                       title='%s for process' % title, 
                       ylabel=ylabel,
                       plotNum=3)

    if metadataTSeries.fieldGroupVals:
        _plot_metadata(x=metadataTSeries.timestamps, 
                       yLists=list(metadataTSeries.fieldGroupVals.values()),
                       title='%s for field groups' % title,
                       ylabel=ylabel,
                       plotNum=4)
    
    plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0)
    plt.gcf().set_size_inches(11, 7)
    
    filename = '%s-%s.png' % ('-'.join(title.split()), test_dcgm_standalone_perf_bounded.__name__)
    figName = os.path.join(logger.log_dir, filename)
    plt.savefig(figName)
    plt.close()
    logger.info('%s figure saved in %s' % (title, utils.shorten_path(figName)))

def _gather_perf_timeseries(handle, watchedFieldIds):
    '''
    Gathers metadata over time and returns a tuple of 
    4 MetadataTimeseries (mem usage, exec time, avg exec time, cpu utilization)
    '''
    
    system = pydcgm.DcgmSystem(handle)
    
    memUsageTS = MetadataTimeseries()
    execTimeTS= MetadataTimeseries()
    execTimeAvgTS = MetadataTimeseries()
    cpuUtilTS = CpuTimeseries()

    numFields = min(len(watchedFieldIds), 50)
    fieldGroups = []
    for i in range(1,6):
        fieldGroups.append(pydcgm.DcgmFieldGroup(handle, "my_field_group_%d" % i, list(watchedFieldIds)[0:numFields]))

    startTime = datetime.datetime.now()
    
    while (datetime.datetime.now() - startTime).total_seconds() < BOUNDED_TEST_DURATION:
        
        # poll memory usage
        memUsageTS.timestamps.append((datetime.datetime.now() - startTime).total_seconds())
            
        memUsageTS.processVals.append(system.introspect.memory.GetForHostengine().bytesUsed)
        memUsageTS.allFieldsVals.append(system.introspect.memory.GetForAllFields().aggregateInfo.bytesUsed)
            
        for id in watchedFieldIds:
            memUsageTS.fieldVals[id].append(
                dcgm_agent_internal.dcgmIntrospectGetFieldMemoryUsage(handle.handle, id).aggregateInfo.bytesUsed)
                
        for fieldGroup in fieldGroups:
            memUsageTS.fieldGroupVals[int(fieldGroup.fieldGroupId.value)].append(system.introspect.memory.GetForFieldGroup(fieldGroup).aggregateInfo.bytesUsed)
            
        # poll execution time
        execTimeTS.timestamps.append((datetime.datetime.now() - startTime).total_seconds())
            
        execTimeTS.allFieldsVals.append(system.introspect.execTime.GetForAllFields().aggregateInfo.totalEverUpdateUsec)
            
        for id in watchedFieldIds:
            execTimeTS.fieldVals[id].append(
                dcgm_agent_internal.dcgmIntrospectGetFieldExecTime(handle.handle, id).aggregateInfo.totalEverUpdateUsec)
            #logger.info("fieldId %d: %s" % (id, str(execTimeTS.fieldVals[id][-1])))
                
        for fieldGroup in fieldGroups:
            execTimeTS.fieldGroupVals[int(fieldGroup.fieldGroupId.value)].append(system.introspect.execTime.GetForFieldGroup(fieldGroup).aggregateInfo.totalEverUpdateUsec)
           
        # poll average execution time
        execTimeAvgTS.timestamps.append((datetime.datetime.now() - startTime).total_seconds())
           
        execTimeAvgTS.allFieldsVals.append(system.introspect.execTime.GetForAllFields().aggregateInfo.recentUpdateUsec)
           
        for id in watchedFieldIds:
            execTimeAvgTS.fieldVals[id].append(
                dcgm_agent_internal.dcgmIntrospectGetFieldExecTime(handle.handle, id).aggregateInfo.recentUpdateUsec)
                
        for fieldGroup in fieldGroups:
            execTimeAvgTS.fieldGroupVals[int(fieldGroup.fieldGroupId.value)].append(system.introspect.execTime.GetForFieldGroup(fieldGroup).aggregateInfo.recentUpdateUsec)
         
        # poll cpu utilization
        cpuUtilTS.timestamps.append((datetime.datetime.now() - startTime).total_seconds())
        cpuUtilTS.cpuInfo.append(system.introspect.cpuUtil.GetForHostengine())
        
        time.sleep(0.050)
        
    return memUsageTS, execTimeTS, execTimeAvgTS, cpuUtilTS

# generating graphs may cause hostengine to timeout so make timeout an extra 20 sec
@test_utils.run_with_standalone_host_engine(timeout=BOUNDED_TEST_DURATION + 20)
@test_utils.run_with_initialized_client()
@test_utils.run_with_introspection_enabled(runIntervalMs=50)
def test_dcgm_standalone_perf_bounded(handle):
    '''
    Test that runs some subtests.  When we bound the number of samples to keep for each field: 
      - DCGM memory usage eventually flatlines on a field, field group, all fields, and process level.
      - DCGM memory usage is at a value that we expect (golden value).  If what we 
         expect changes over time the we must update what these values are (the tests will fail if we don't).
         
    Plots of the memory usage and execution time generated during this test are saved and the 
    filename of the figure is output on the terminal.
    
    Multiple tests are included in this test in order to save time by only gathering data once.
    '''
    if not option_parser.options.developer_mode:
        test_utils.skip_test("Skipping developer test.")
    
    handle = pydcgm.DcgmHandle(handle)
    group = pydcgm.DcgmGroup(handle, groupName="metadata-test", groupType=dcgm_structs.DCGM_GROUP_DEFAULT)
    updateFreq = 1000000 # 1 second. Needs to be long enough for all fields on all GPUs to update, or the record density will vary based on CPU consumption
     
    watchedFieldIds = test_utils.watch_all_fields(handle.handle,
                                                  group.GetGpuIds(),
                                                  updateFreq,
                                                  maxKeepAge=0.0, #Use maxKeepEntries only to enforce the quota
                                                  maxKeepEntries=10)
    
    memUsageTS, execTimeTS, execTimeAvgTS, cpuUtilTS = _gather_perf_timeseries(handle, watchedFieldIds)
    activeGpuCount = test_utils.get_live_gpu_count(handle.handle)
    
    # run the actual tests on the gathered data
    
    # test that memory usage flatlines
    test_utils.run_subtest(_test_mem_bounded_flatlines_fields, memUsageTS)
    test_utils.run_subtest(_test_mem_bounded_flatlines_fieldgroups, memUsageTS)
    test_utils.run_subtest(_test_mem_bounded_flatlines_allfields, memUsageTS)
    test_utils.run_subtest(_test_mem_bounded_flatlines_process, memUsageTS)
      
    # test that memory usage is at an expected level (golden value)
    # the tail end of the series should be VERY close to the end since we compare the mean
    # of the tail to the golden value
    tailStart = int(0.8 * len(memUsageTS.timestamps))
    test_utils.run_subtest(_test_mem_bounded_golden_values_fields, activeGpuCount, memUsageTS, tailStart)
    test_utils.run_subtest(_test_mem_bounded_golden_values_allfields, activeGpuCount, memUsageTS, tailStart, len(watchedFieldIds))
    test_utils.run_subtest(_test_mem_bounded_golden_values_process, memUsageTS, tailStart, len(watchedFieldIds))
      
    # tests for CPU utilization (see functions for descriptions)
    test_utils.run_subtest(_test_cpuutil_bounded_flatlines_hostengine, cpuUtilTS)
      
    # test that execution time grows at a linear rate
    #test_utils.run_subtest(_test_exectime_bounded_linear_growth, execTimeTS)
    
    # make some pretty graphs to look at for insight or to help debug failures
    _generate_metadata_line_charts(memUsageTS, ylabel='bytes', title='Bytes Used')
    _generate_metadata_line_charts(execTimeTS, ylabel='usec', title='Execution Time')
    _generate_metadata_line_charts(execTimeAvgTS, ylabel='usec', title='Recent Exec Time')
    _generate_cpu_line_charts(cpuUtilTS)
    
    barPlotPoints = [(id, execTimeAvgTS.fieldVals[id][-1]) for id in execTimeAvgTS.fieldVals]
    _plotFinalValueOrderedBarChart(barPlotPoints, 
                                  title='Top 20 Field Recent Exec Time', 
                                  ylabel='usec', 
                                  filenameBase='test-perf')
    
def _generate_cpu_line_charts(cpuUtilTS):
    if not isReqMatplotlibVersion():
        logger.info('Not generating CPU utilization graphs since "matplotlib" is not the required version')
        return
    if logger.log_dir is None:
        logger.info('Not generating CPU utilization graphs since logging is disabled')
        return
    
    x = cpuUtilTS.timestamps
    
    totalCpuUtil    = [100*data.total for data in cpuUtilTS.cpuInfo]
    kernelCpuUtil   = [100*data.kernel for data in cpuUtilTS.cpuInfo]
    userCpuUtil     = [100*data.user for data in cpuUtilTS.cpuInfo]
    
    fig, ax = plt.subplots()
    ax.set_title('CPU Utilization')
    
    # hacky way of generating legend colors to match graph colors
    polys = ax.stackplot(x, kernelCpuUtil, userCpuUtil)
    legendProxies = []
    for poly in polys:
        legendProxies.append(plt.Rectangle((0,0), 1, 1, fc=poly.get_facecolor()[0]))
    
    ax.legend(legendProxies, ['kernel', 'user'], loc='upper right')
    
    plt.xlabel('seconds since start')
    plt.ylabel('% of device CPU resources')
    
    plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0)
    plt.gcf().set_size_inches(11, 7)
    
    filename = '%s-%s.png' % ('CPU-Util', test_dcgm_standalone_perf_bounded.__name__)
    figName = os.path.join(logger.log_dir, filename)
    plt.savefig(figName)
    plt.close()
    logger.info('%s figure saved in %s' % ('CPU-Util', utils.shorten_path(figName)))
    
def _test_exectime_bounded_linear_growth(execTimeTS):
    '''
    Test that when the number of samples that DCGM collects is limited there is linear growth 
    in the total amount of time used to retrieve that each field.  
    '''
    tolerance = 0.60
    
    for fieldId, series in execTimeTS.fieldVals.items():
        tailStart = int(0.4*len(series))
        tailLen = len(series) - tailStart
        
        # take a linear regression of the execution timeseries 
        # if its corr. coeff. is not high (1.0 is highest)
        # OR
        # if its slope is much different from the actual start -> end slope 
        # THEN something is wrong.
        
        # calc the lin. regr. slope 
        # taken from https://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line
        x = execTimeTS.timestamps[tailStart:]
        y = series[tailStart:]
        if y[-1] == 0:
            logger.info("Skipping fieldId %d with exec times of 0" % fieldId)
            continue

        #logger.info("x %s, y %s" % (str(x), str(y)))
        rxy = stats.correlation_coefficient(x, y)
        sx = stats.standard_deviation(x)
        sy = stats.standard_deviation(y)
        
        assert(rxy >= 0.90), (
            'execution time for field %s did not have a strong linear correlation. ' % fieldId + 
            'Its correlation coefficient was %.4f' % rxy)
        logger.debug('corr. coeff. for field %s: %s' % (fieldId, rxy))
        
        linRegSlope = rxy * (sy / sx)
        slope = (y[-1] - y[0]) / float(x[-1] - x[0])
        
        minSlope = (1-tolerance)*linRegSlope
        maxSlope = (1+tolerance)*linRegSlope
        assert(minSlope <= slope <= maxSlope), (
            'execution time growth for field %s was not linear. ' % fieldId +
            'It had an overall slope of %s but the linear regression slope was %s. '
            % (slope, linRegSlope) +
            'Tolerated min: %s, tolerated max: %s' % (minSlope, maxSlope))
    
def _assert_flatlines(seriesType, seriesId, series):
    if sum(series) == 0:
        return
    
    tailStart = int(0.4 * len(series))
    seriesTail = series[tailStart:]
    
    # assert that the each point on the tail is no more than 5% away from the mean
    # this indicates that the series leveled-off
    flatlineVal = stats.mean(point for point in seriesTail)
    
    for point in seriesTail:
        dFlatlinePercent = (point - flatlineVal) / flatlineVal
        assert (abs(dFlatlinePercent) < 0.05), ('memory usage did not flatline.  ' 
            + 'A point of type "%s" with ID "%s" was %.4f%% away from indicating a flat line. \nTail Points: %s\nPoints: %s' 
            % (seriesType, seriesId, 100*dFlatlinePercent, str(seriesTail), str(series))
            + 'See the the memory usage plot ".png" file outputted on the terminal above for further details')
        
def _test_mem_bounded_flatlines_allfields(memUsageTS):
    _assert_flatlines('all-fields', '', memUsageTS.allFieldsVals)
    
def _test_mem_bounded_flatlines_process(memUsageTS):
    _assert_flatlines('process', '', memUsageTS.processVals)
    
def _test_mem_bounded_flatlines_fields(memUsageTS):
    for id, series in memUsageTS.fieldVals.items():
        _assert_flatlines('field', id, series)
        
def _test_mem_bounded_flatlines_fieldgroups(memUsageTS):
    for id, series in memUsageTS.fieldGroupVals.items():
        _assert_flatlines('field-group', id, series)

def helper_field_has_variable_size(fieldId):
    '''
    Returns True if a field has a variable memory size per record. False if it doesn't.
    '''
    if fieldId == dcgm_fields.DCGM_FI_DEV_GPU_UTIL_SAMPLES or \
       fieldId == dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES or \
       fieldId == dcgm_fields.DCGM_FI_DEV_GRAPHICS_PIDS or \
       fieldId == dcgm_fields.DCGM_FI_DEV_COMPUTE_PIDS:
       return True

    fieldMeta = dcgm_fields.DcgmFieldGetById(fieldId)
    if fieldMeta.fieldType == dcgm_fields.DCGM_FT_BINARY:
        return True
    else:
        return False

def _test_mem_bounded_golden_values_fields(activeGpuCount, memUsageTS, tailStart):
    goldenVal = 1148      # 1 KB plus some swag per field instance (Global, GPU). This is based off of the keyed vector block size and default number of blocks
    tolerance = 0.10      # low tolerance, amount of records stored is bounded
    
    for fieldId, series in memUsageTS.fieldVals.items():
        
        seriesTail = series[tailStart:]
        
        # skip fields that are not implemented
        if sum(seriesTail) == 0:
            continue

        #Don't check the size of binary fields since it's arbitrary per fieldId
        if helper_field_has_variable_size(fieldId):
            logger.info("Skipping variable-sized fieldId %d" % fieldId)
            continue
        
        mean = stats.mean(seriesTail)
            
        lowLimit = (1-tolerance)*goldenVal
        highLimit = (1+tolerance)*goldenVal*activeGpuCount
        assert lowLimit < mean < highLimit, \
            'Expected field "%d" memory usage to be between %s and %s but got %s' % \
            (fieldId, lowLimit, highLimit, mean) \
            + 'If this new value is expected, change the golden value used for comparison.'
            
def _test_mem_bounded_golden_values_allfields(activeGpuCount, memUsageTS, tailStart, numFieldIds):
    goldenVal = 2000 * numFieldIds * activeGpuCount # 2 KiB per fieldId per GPU. This gives some swag for the binary fields that are larger
    tolerance = 0.15    # low tolerance, amount of records stored is bounded
    
    mean = stats.mean(memUsageTS.allFieldsVals[tailStart:])
    logger.info("Mean total field value memory usage: %f" % mean)

    assert mean < (1+tolerance)*goldenVal, \
        'Expected all fields bytes used to be within %.2f%% of %d but it was %d.  ' % \
        (100*tolerance, goldenVal, mean) \
        + 'If this new value is expected, change the golden value used for comparison.'
        
def _test_mem_bounded_golden_values_process(memUsageTS, tailStart, numFieldIds):
    highWaterMark = 29000000 #Setting a canary in the coal mine value. This comes from the /proc filesystem and can report anywhere from 15 to 28 MiB. 
    
    mean = stats.mean(memUsageTS.processVals[tailStart:])
    assert (mean < highWaterMark), \
        'Expected bytes used of the process to be under %d but it was %.2f.  ' % \
        (highWaterMark, mean) \
        + 'If this new value is expected, change the high water mark.'
        
def _test_cpuutil_bounded_flatlines_hostengine(cpuUtilTS):
    '''
    Test that the CPU utilization flatlines when record storage is bounded
    '''
    tailStart = int(0.4 * len(cpuUtilTS.timestamps))
    tail = [data.total for data in cpuUtilTS.cpuInfo[tailStart:]]
    
    tolerance = 0.20 # points more than this much relative distance from the mean are outliers
    relativeOutliersAllowed = 0.02 # 2% outliers allowed
    outlierCount = 0
    mean = stats.mean(tail)
    
    for cpuUtil in tail:
        if not ((1-tolerance)*mean < cpuUtil < (1+tolerance)*mean):
            outlierCount += 1
           
    relativeOutliers = outlierCount / float(len(tail)) 
    assert(relativeOutliers < relativeOutliersAllowed), (
        'CPU utilization did not stay consistent.  It varied for %.2f%% of the time out of %d points '
        % (100*relativeOutliers, len(tail))
        + 'but it is only allowed to vary %.2f%% of the time' % (100*relativeOutliersAllowed))
