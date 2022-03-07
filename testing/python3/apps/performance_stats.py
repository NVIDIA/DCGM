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
import re
import math
import sys
from operator import itemgetter

## Temp added to track down if log length issue happens again, Remove when nightly is fixed
import option_parser
import os

epsilon = sys.float_info.epsilon # add epsilon to all times to prevent division by 0
import test_utils
import logger

__all__ = ['PerformanceStats']

# TODO programmable switching
def verbose_debug( text ):
    # print text
    # logger.debug(text)
    pass

def average(num_list):
    return sum(num_list) / float(len(num_list))

def stdev(num_list):
    avg = average(num_list)
    return math.sqrt(sum((x - avg) ** 2 for x in num_list) / len(num_list))

def _time_str(num_list):
    if len(num_list) > 1:
        return "%.3fms\t%.3fms\t%.3fms" % (average(num_list) * 1000, stdev(num_list) * 1000, max(num_list) * 1000)
    return "%.3fms" % (num_list[0] * 1000)

class DebugLine:
    """ Class that matches DEBUG lines in trace log """
    #                                             tid        timestamp     path         fname  line         content
    regexpDebugLine = re.compile("^DEBUG:\s*\[tid \d*\]\s*\[(\d+\.\d+)s - (\w?:?[^:]+):?(\w+)?:([0-9]+)\]\s*(.*)")

    @staticmethod
    def construct(text):
        try:
            return DebugLine(text)
        except ValueError:
            return None

    def __init__(self, text):
        text = text.strip()
        self.match = DebugLine.regexpDebugLine.match(text);
        if not self.match:
            raise ValueError
        self.timestamp = float(self.match.group(1))
        self.srcfilename = self.match.group(2).replace("\\", "/")
        self.srcfunctionname = self.match.group(3)
        self.srcline = int(self.match.group(4))
        self.message = self.match.group(5)

    def __str__(self):
        return "(%s, %s, %s)" % (self.timestamp, self.srcStr(), self.message)

    def srcStr(self):
        if self.srcfunctionname:
            return "%s:%s:%d" % (self.srcfilename, self.srcfunctionname, self.srcline)
        else:
            return "%s:%d" % (self.srcfilename, self.srcline)

    def isInTheSamePlace(self, line):
        return self.srcfilename == line.srcfilename and self.srcfunctionname == line.srcfunctionname and self.srcline == line.srcline
    
    def isInTheSameFunction(self, line):
        return self.srcfilename == line.srcfilename and self.srcfunctionname == line.srcfunctionname

class RmCall:
    """ Class that matches Rm Calls """
    regexpRmCallRelease        = re.compile("^([0-9a-f]*) ([0-9a-f]*)$")
    regexpRmCallReleaseReturn  = re.compile("^([0-9a-f]*) ([0-9a-f]*) ## 0x([0-9a-f]*)$")
    regexpRmCallDebug          = re.compile("^dcgmRmCall\(([a-zA-Z0-9_.]* [0-9a-f]*), (\w*), \.\.\.\)$")
    regexpRmCallDebugReturn    = re.compile("^dcgmRmCall\(([a-zA-Z0-9_.]* [0-9a-f]*), (\w*), \.\.\.\) returned 0x([0-9a-f]*)$")
    regexpFilePath             = re.compile(".*dmal/rm/.*\.c$")
    regexpRmCallSrc            = re.compile(".*dcgmRmCall.*(NV\d{4}_CTRL_CMD_[A-Z0-9_]*).*")
    
    @staticmethod
    def construct(debugLines, i, dcgmParent):
        try:
            return RmCall(debugLines, i, dcgmParent)
        except ValueError:
            return None

    def __init__(self, debugLines, i, dcgmParent):
        if i + 1 >= len(debugLines):
            raise ValueError
        
        line1 = debugLines[i];
        line2 = debugLines[i + 1];
        verbose_debug("RmCall: Matching line1 %d %s" % (i, str(line1)))
        verbose_debug("RmCall: Matching line2 %d %s" % (i+1, str(line2)))
        if not line1.isInTheSamePlace(line2):
            verbose_debug("RmCall: Failed because they are not in the same line")
            raise ValueError
        if not RmCall.regexpFilePath.match(line1.srcfilename):
            verbose_debug("RmCall: Failed because they are not in the correct file in dmal/rm/rm_*")
            raise ValueError

        self.releaseLog = 1
        self.rmCallName = 0
        match1 = RmCall.regexpRmCallRelease.match(line1.message)
        match2 = RmCall.regexpRmCallReleaseReturn.match(line2.message)
        if not match1 or not match2:
            self.releaseLog = 0
            self.rmCallName = 1
            verbose_debug("RmCall: Failed match regexpRmCallRelease* but still in trying other regexps")
            match1 = RmCall.regexpRmCallDebug.match(line1.message)
            match2 = RmCall.regexpRmCallDebugReturn.match(line2.message)
        if not match1 or not match2:
            verbose_debug("RmCall: Failed match regexpRmCallDebug*")
            raise ValueError

        if match1.group(1) != match2.group(1) or match1.group(2) != match2.group(2):
            verbose_debug("RmCall: Failed check where device and function strings should be the same*")
            raise ValueError

        self.src = line1;
        self.device = match1.group(1)
        self.function = match1.group(2)
        self.returnCode = match2.group(3)
        self.time = line2.timestamp - line1.timestamp + epsilon
        self.times = [self.time] # so that stats from multiple runs could be appended
        self.dcgmParent = dcgmParent if dcgmParent and dcgmParent.isParentOfRmCall(self) else None

    def is_simillar(self, b):
        return isinstance(b, RmCall) and self.src.isInTheSamePlace(b.src) and self.function == b.function and self.returnCode == b.returnCode

    def __str__(self):
        dcgmParentStr = "(%.2f%% of %s)" % (self.time / self.dcgmParent.time * 100, self.dcgmParent.shortString()) if self.dcgmParent else ""
        name = self.function if self.rmCallName else "RM CALL " + self.src.srcStr()
        return "%s\t  %s\t%s" % (_time_str(self.times), name, dcgmParentStr)

class NvmlCall:
    """ Class that matches Nvml Calls """
    # TODO doesn't handle apiEnter failures!
    regexpNvmlCall             = re.compile("^Entering (dcgm[a-zA-Z0-9_]*)(\(.*\)) *(\(.*\))$")
    regexpNvmlIntRelCall       = re.compile("^()()(\(.*\))$")
    regexpNvmlCallReturn       = re.compile("^Returning (\d*) \(([a-zA-Z ]*)\)$")
    regexpNvmlIntRelCallReturn = re.compile("^(\d*) ([a-zA-Z ]*)$")
    regexpFilePath             = re.compile(".*entry_points.h$")
    regexpFilePathNonTsapi     = re.compile(".*dcgm.c$")
    regexpNvmlCallSrc          = re.compile("^ *NVML_INT_ENTRY_POINT\((dcgm[A-Z][a-zA-Z0-9_]*) *,.*")
    
    @staticmethod
    def construct(debugLines, i):
        try:
            return NvmlCall(debugLines, i)
        except ValueError:
            return None

    def __init__(self, debugLines, i):
        line1 = debugLines[i];
        
        verbose_debug("NvmlCall: Matching line %d %s" % (i, str(line1)))
        self.istsapi = 1
        if not NvmlCall.regexpFilePath.match(line1.srcfilename):
            self.istsapi = 0
            verbose_debug("NvmlCall: Wrong file name also matching non tsapi regexp")
            if not NvmlCall.regexpFilePathNonTsapi.match(line1.srcfilename):
                verbose_debug("NvmlCall: Wrong file also doesn't match non tsapi regexp")
                raise ValueError

        self.internal = 0
        self.dcgmCallName = 1
        match1 = NvmlCall.regexpNvmlCall.match(line1.message)
        if not match1:
            self.internal = 1
            self.dcgmCallName = 0
            verbose_debug("NvmlCall: Failed match regexpNvmlCall but need to also try regexpNvmlIntRelCall")
            match1 = NvmlCall.regexpNvmlIntRelCall.match(line1.message)
            if not match1:
                verbose_debug("NvmlCall: Failed match regexpNvmlIntRelCall")
                raise ValueError

        verbose_debug("NvmlCall: Matching the end of the dcgm call")
        for j in range(i + 1, len(debugLines)):
            line2 = debugLines[j];
            if not line1.isInTheSameFunction(line2):
                continue
            if self.istsapi and not line1.isInTheSamePlace(line2):
                continue

            match2 = NvmlCall.regexpNvmlIntRelCallReturn.match(line2.message) if self.internal else NvmlCall.regexpNvmlCallReturn.match(line2.message)
            if not match2:
                verbose_debug("NvmlCall: Sth went wrong. Found line2 \"%s\" that doesn't match the return but is in the same line" % (str(line2)))
                raise ValueError
                return
        
            verbose_debug("NvmlCall: Matched the end line %d %s" % (j, str(line2)))

            # TODO match device
            self.src = line1;
            self.srcEnd = line2;
            self.function = match1.group(1) if self.dcgmCallName else "NVML INT " + self.src.srcStr()
            self.argsType = match1.group(2)
            self.args = match1.group(3)
            self.errcode = match2.group(1)
            self.errcodeStr = match2.group(2)
            self.time = line2.timestamp - line1.timestamp + epsilon
            self.times = [self.time] # so that stats from multiple runs could be appended
            return
        verbose_debug("NvmlCall: End of dcgm call wasn't found")
        raise ValueError
    
    def isParentOfRmCall(self, rmCall):
        return self.src.timestamp <= rmCall.src.timestamp and rmCall.src.timestamp <= self.srcEnd.timestamp

    def shortString(self):
        return "%s" % (self.function)
    
    def is_simillar(self, b):
        return isinstance(b, NvmlCall) and self.src.isInTheSamePlace(b.src) and self.function == b.function and self.errcode == b.errcode

    def __str__(self):
        return "%s\t%s\t%s" % (_time_str(self.times), self.function, self.args)

class PerformanceStats(object):
    def __init__(self, input_fname):
        verbose_debug("Decoding " + input_fname + " file")
       
        # read from file
        with open(input_fname, encoding='utf-8', errors='ignore') as fin:
            rawlines = fin.readlines()

        # Parse only DEBUG level trace lines
        # only these contain start/stop function entry information
        lines = [x for x in [DebugLine.construct(y) for y in rawlines] if x]
        
        # look for dcgm function calls and RM function calls inside of trace lines
        i = 0
        lastNvmlCall = None
        self.time_in_rm = 0.0 + epsilon
        self.time_in_dcgm = 0.0 + epsilon
        self.stats = []
        for i in range(len(lines)):
            line = lines[i]
            verbose_debug(line.match.groups())
        
            dcgmCall = NvmlCall.construct(lines, i)
            if dcgmCall:
                lastNvmlCall = dcgmCall
                self.time_in_dcgm += dcgmCall.time
                self.stats.append(dcgmCall)
                continue
        
            rmCall = RmCall.construct(lines, i, lastNvmlCall)
            if rmCall:
                self.time_in_rm += rmCall.time
                self.stats.append(rmCall)
                continue
        
        if len(lines) > 0:
            self.time_total = lines[-1].timestamp - lines[0].timestamp + epsilon
        else:
            self.time_total = -1
        self.times_in_dcgm = [self.time_in_dcgm]
        self.times_in_rm = [self.time_in_rm]
        self.times_total = [self.time_total]
        self._combined_stats_count = 1

    def write_to_file(self, fname, dcgm_stats=True, rm_stats=True):
        with open(fname, "w") as fout:
            fout.write("Called functions (in order):\n")
            
            if self._combined_stats_count > 1:
                fout.write("    avg\t  stdev\t    max\t name\n")
            else:
                fout.write("   time\t name\n")
            calls = dict() # for per function stats:
            for stat in self.stats:
                if not dcgm_stats and isinstance(stat, NvmlCall):
                    continue
                if not rm_stats and isinstance(stat, RmCall):
                    continue
                fout.write(str(stat))
                fout.write("\n")
                calls.setdefault(stat.function, []).extend(stat.times)

            fout.write("%s\t%s\n" % (_time_str(self.times_total), "Total time"))
            fout.write("%s\t%s (%.2f%% of total time)\n" % (_time_str(self.times_in_dcgm), "Time spent in NVML", average(self.times_in_dcgm) / average(self.times_total) * 100))
            fout.write("%s\t%s (%.2f%% of total time)\n" % (_time_str(self.times_in_rm), "Time spent in RM", average(self.times_in_rm) / average(self.times_total) * 100))
            fout.write("\n")
            
            # Print per function stats
            avgsum = "sum"
            if self._combined_stats_count > 1:
                avgsum = "avg sum" # if stats are combined then we return avg sum for all runs

            fout.write("Per function stats (sorted by avg):\n")
            fout.write("    avg\t  stdev\t    max\t%7s\t name\n" % avgsum)
            per_function = [(average(calls[x]) * 1000, stdev(calls[x]) * 1000, max(calls[x]) * 1000, sum(calls[x]) * 1000 / self._combined_stats_count, x) for x in calls]
            per_function.sort(reverse=True)
            for function in per_function:
                fout.write("%.3fms\t%.3fms\t%.3fms\t%.3fms\t%s\n" % function)
            fout.write("\n")
            
            fout.write("Per function stats (sorted by sum):\n")
            fout.write("    avg\t  stdev\t    max\t%7s\t name\n" % avgsum)
            per_function.sort(key=itemgetter(3), reverse=True)
            for function in per_function:
                fout.write("%.3fms\t%.3fms\t%.3fms\t%.3fms\t%s\n" % function)
            fout.write("\n")

    def write_to_file_dvs(self, fname, dcgm_stats=True, rm_stats=True):
        with open(fname, "w") as fout:
            def format_stats(name, num_list):
                if len(num_list) > 1:
                    return "%s_avg, %.3f\n%s_stdev, %.3f\n%s_max,%.3f\n" % (name, average(num_list) * 1000, name, stdev(num_list) * 1000, name, max(num_list) * 1000)
                return "%s, %.3f" % (name, num_list[0] * 1000)

            calls = dict()
            for stat in self.stats:
                if not dcgm_stats and isinstance(stat, NvmlCall):
                    continue
                if not rm_stats and isinstance(stat, RmCall):
                    continue
                calls.setdefault(stat.function, []).extend(stat.times)

            fout.write(format_stats("total_time", self.times_total))
            fout.write(format_stats("total_dcgm_time", self.times_in_dcgm))
            fout.write(format_stats("total_rm_time", self.times_in_rm))
            for (name, times) in list(calls.items()):
                fout.write(format_stats(name, times))

    def combine_stat(self, perf_stat):
        """
        Merges into self additional stats so that average and stdev for each entry could be calculated.
        perf_stat must contain the same NVML/RM calls in the same order.

        """

        ## Temp added to track down if log length issue happens again, Remove when nightly is fixed
        if (len(perf_stat.stats) != len(self.stats)):
            log_perf_stat = ""
            log_self_stat = ""
            for i in range(len(perf_stat.stats)):
                log_perf_stat += str(perf_stat.stats[i]) + "\n"
            for i in range(len(self.stats)):
                log_self_stat += str(self.stats[i]) + "\n"
            if not test_utils.noLogging:
                fname_log_perf_stat = os.path.relpath(os.path.join(logger.log_dir, "log_perf_stat.txt"))
                fname_log_self_stat = os.path.relpath(os.path.join(logger.log_dir, "log_self_stat.txt"))
                f1 = open(fname_log_perf_stat, "w")
                f2 = open(fname_log_self_stat, "w")
                f1.write(log_perf_stat)
                f2.write(log_self_stat)
                f1.close()
                f2.close()

        # TODO get rid of this requirement by merging the logs with difflib
        # Some dcgm calls (e.g. dcgmDeviceGetCurrentClocksThrottleReasons) can take different RM calls depending on the
        # state of the GPU (e.g. clock changes that can happen at any point).
        
        ## Comment strict matching of log length. The perf data will be collected for atleast 1 run anyways
        ##  assert len(perf_stat.stats) == len(self.stats), "One of the logs is of different length"

        if (len(perf_stat.stats) != len(self.stats)):
            logger.warning("Perf logs mismatch. nvsmi perf data collected for %s run(s)" % str(self._combined_stats_count))
            return

        for i in range(len(self.stats)):
            stat1 = self.stats[i]
            stat2 = perf_stat.stats[i]
            assert stat1.is_simillar(stat2), "stat %d: %s doesn't match %s and can't be combined" % (i, stat1, stat2)
            stat1.times.extend(stat2.times)

        self.times_total.extend(perf_stat.times_total)
        self.times_in_rm.extend(perf_stat.times_in_rm)
        self.times_in_dcgm.extend(perf_stat.times_in_dcgm)
        self._combined_stats_count += 1

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("Usage <app> <input_decoded_log> <output_stats_file>")
        sys.exit(1)

    stats = PerformanceStats(sys.argv[1])
    stats.write_to_file(sys.argv[2])
    print("Done")
        
