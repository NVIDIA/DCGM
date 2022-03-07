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
import argparse
import collections
import csv

class ParseDcgmSingleMetric:
    "class for parsing a single metric"

    def __init__(self):
        self.d_cpunum_th = collections.OrderedDict()
        self.d_cpu_count = {}

        self.metric_label_list = []
        self.dcgm_val_gpu = []
        self.gpuCount = 1

    def createFieldsFromMetricLabel(self, metricLabelString, gpu_list):
        print(("GPU Count", gpuCount))
        for i in range(0, gpuCount):
            #self.metric_label_list.append(str(gpu_list[i]) + '_' + str(metricLabelString))
            self.metric_label_list.append(str(metricLabelString) + '_' + str(gpu_list[i]))


    def writeToCsv(self, sample_num):
        dict_row = {}
        if gpuCount == 8:
            dict_row = {'Sample Number':sample_num, self.metric_label_list[0]:self.dcgm_val_gpu[0], self.metric_label_list[1]:self.dcgm_val_gpu[1], self.metric_label_list[2]:self.dcgm_val_gpu[2], self.metric_label_list[3]:self.dcgm_val_gpu[3], self.metric_label_list[4]:self.dcgm_val_gpu[4], self.metric_label_list[5]:self.dcgm_val_gpu[5], self.metric_label_list[6]:self.dcgm_val_gpu[6], self.metric_label_list[7]:self.dcgm_val_gpu[7],}
        elif gpuCount == 7:
            dict_row = {'Sample Number':sample_num, self.metric_label_list[0]:self.dcgm_val_gpu[0], self.metric_label_list[1]:self.dcgm_val_gpu[1], self.metric_label_list[2]:self.dcgm_val_gpu[2], self.metric_label_list[3]:self.dcgm_val_gpu[3], self.metric_label_list[4]:self.dcgm_val_gpu[4], self.metric_label_list[5]:self.dcgm_val_gpu[5], self.metric_label_list[6]:self.dcgm_val_gpu[6]}
        elif gpuCount == 6:
            dict_row = {'Sample Number':sample_num, self.metric_label_list[0]:self.dcgm_val_gpu[0], self.metric_label_list[1]:self.dcgm_val_gpu[1], self.metric_label_list[2]:self.dcgm_val_gpu[2], self.metric_label_list[3]:self.dcgm_val_gpu[3], self.metric_label_list[4]:self.dcgm_val_gpu[4], self.metric_label_list[5]:self.dcgm_val_gpu[5]}
        elif gpuCount == 5:
            dict_row = {'Sample Number':sample_num, self.metric_label_list[0]:self.dcgm_val_gpu[0], self.metric_label_list[1]:self.dcgm_val_gpu[1], self.metric_label_list[2]:self.dcgm_val_gpu[2], self.metric_label_list[3]:self.dcgm_val_gpu[3], self.metric_label_list[4]:self.dcgm_val_gpu[4]}
        elif gpuCount == 4:
            dict_row = {'Sample Number':sample_num, self.metric_label_list[0]:self.dcgm_val_gpu[0], self.metric_label_list[1]:self.dcgm_val_gpu[1], self.metric_label_list[2]:self.dcgm_val_gpu[2], self.metric_label_list[3]:self.dcgm_val_gpu[3]}
        elif gpuCount == 3:
            dict_row = {'Sample Number':sample_num, self.metric_label_list[0]:self.dcgm_val_gpu[0], self.metric_label_list[1]:self.dcgm_val_gpu[1], self.metric_label_list[2]:self.dcgm_val_gpu[2]}
        elif gpuCount == 2:
            dict_row = {'Sample Number':sample_num, self.metric_label_list[0]:self.dcgm_val_gpu[0], self.metric_label_list[1]:self.dcgm_val_gpu[1]}
        elif gpuCount == 1:
            dict_row = {'Sample Number':sample_num, self.metric_label_list[0]:self.dcgm_val_gpu[0]}

        return dict_row

    def parseAndWriteToCsv(self, fName, metric, gpu_list):
        global gpuCount
        csvFileName = 'dcgm_' + str(metric) + '.csv'
        sample_num = 0
        f = open(fName, 'r+')
        lines = f.readlines() # read all lines at once
        metric_label = lines[0].split()[2]
        gpuCount = len((gpu_list).split(","))

        self.createFieldsFromMetricLabel(metric_label, gpu_list.split(","))
        with open(csvFileName, 'wb') as csvFile:
            fieldnames = ['Sample Number']
            for i in range(0, gpuCount):
                fieldnames.append(self.metric_label_list[i])
            writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
            writer.writeheader()

            i = 2
            while i < len(lines):
                try:
                    line = lines[i]
                    row = line.split()
                    if row == [] or row[0] == 'Id' or row[1] == 'GPU/Sw':
                        #print ("Skipping non data row: " + str(row))
                        i = i+1
                    elif row[0].isdigit():
                        for k in range(0, gpuCount):
                            val = float(lines[i+k].split()[1])
                            #print (val, i)
                            self.dcgm_val_gpu.append(val)

                        sample_num = sample_num + 1
                        dict_row = self.writeToCsv(sample_num)
                        writer.writerow(dict_row)
                        i += gpuCount
                        self.dcgm_val_gpu[:] = []
                except IndexError:
                    i = i+1
                except StopIteration:
                    pass
        print("Done")

def main(cmdArgs):
    fName = cmdArgs.fileName
    metric = cmdArgs.metric
    gpu_list = cmdArgs.gpu_list
    #parse and output the data
    po = ParseDcgmSingleMetric()
    po.parseAndWriteToCsv(fName, metric, gpu_list)


def parseCommandLine():
    parser = argparse.ArgumentParser(description="Parse logs from dcgmLogs into a csv")
    parser.add_argument("-f", "--fileName", required=True, help="fielName of the file to be parsed and outputted to csv")
    parser.add_argument("-m", "--metric", required=True, help="metric for which the data is being analyzed")
    parser.add_argument("-i", "--gpu_list", required=False, default='0,1,2,3,4,5,6,7', help="metric for which the data is being analyzed")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    cmdArgs = parseCommandLine()
    main(cmdArgs)
