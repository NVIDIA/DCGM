import datetime
import time
import json
from collections import OrderedDict

class TestData:
    # -------------------------------------------------------------
    def __init__(self):
        self.dataMapFinal = OrderedDict()
        self.dataMap = OrderedDict()
        self.summary = {
            "startTime" : "",
            "endTime" : "",
            "timeOfRun" : "",
            "numberOfTestsRun" : 0,
            "testsPassed" : 0,
            "testsFailed" : 0,
            "testsSkipped" : 0
        }

        self.dataMapStorage = {
            "startTime":"",
            "endTime":"",
            "timeOfRun":"",
            "name":"",
            "status": "",
            "message":""
        }

        self.curModuleName = ""
        self.curFuncName = ""
        self.curCustomFuncName = ""
        self.jsonFilePath = "test_data.json"
        self.jsonFilePathCompiled = "test_data_compiled.json"

    # addModuleName
    #
    # This adds a module with name moduleName. It has no effect if it already
    # exists except to cache the module name.
    #
    def addModuleName(self, moduleName):
        self.refreshClassVars()
        self.curModuleName = moduleName
        
        if moduleName not in self.dataMap: 
            self.dataMap[moduleName] = {}

    # addModule
    #
    # Add module and module functions. This sets the dataMap for the module
    # identified by moduleName. It overrides any existing dataMap.
    #
    # Arguments:
    #
    #    moduleName    - module name
    #    functionNames - list of functions
    #
    def addModule(self, moduleName, functionNames):
        self.addModuleName(moduleName)
        
        temp = {}
        for name in functionNames:
            funcDic = {
                "runData":list([]),
                "isMultiRun": 0,
                "allTests": []
                }
            temp[name] = dict(funcDic)
        
        self.dataMap[moduleName] = temp

    # refreshClassVars
    #
    # Reset the cached module and funcion name.
    def refreshClassVars(self):
        self.curModuleName = ""
        self.curFuncName = ""
        
    # initFuncDict
    #
    # initialize dictionary for currently cached function.
    #
    def initFuncDic(self):
        dictionary = dict(self.dataMapStorage)
        dictionary["name"] = self.curFuncName
        self.dataMap[self.curModuleName][self.curFuncName]["runData"].append(dictionary)

    # addName
    #
    # Arguments:
    #
    #     functionName - function to add under curModuleName
    #
    def addName(self, functionName):
        self.curFuncName = functionName
        
        if self.curFuncName not in self.dataMap[self.curModuleName]:
            funcDic = {
                "runData":list([]),
                "isMultiRun": 0,
                "allTests": []
            }

            self.dataMap[self.curModuleName][self.curFuncName] = funcDic
            
        self.initFuncDic()

    # addModuleStartTime
    #
    # Add current time to cached module start time.
    #
    def addModuleStarttime(self):
        startTimeSeconds = time.time()
        startTimeMicroseconds = datetime.datetime.fromtimestamp(startTimeSeconds)
        startTime = startTimeMicroseconds.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        self.dataMap[self.curModuleName]["startTime"] = startTime

    # -------------------------------------------------------------
    # addModuleEndTime
    #
    # Add current time to cached module end time and compute run time.
    #
    def addModuleEndTime(self):
        endTimeSeconds = time.time()
        endTimeMicroseconds = datetime.datetime.fromtimestamp(endTimeSeconds)
        endTime = endTimeMicroseconds.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        self.dataMap[self.curModuleName]["endTime"] = endTime

        start_time = datetime.datetime.strptime(self.dataMap[self.curModuleName]["startTime"], "%Y-%m-%d %H:%M:%S.%f")
        end_time = datetime.datetime.strptime(endTime, "%Y-%m-%d %H:%M:%S.%f")
        time_diff = end_time - start_time
        self.dataMap[self.curModuleName]["timeOfRun"] = str(time_diff)

    # addStartTime
    #
    # Add the current time to the start time of the current run of the current
    # cached module and function.
    #
    # Arguments:
    #
    #     runNumber - test run mumber (default 0)
    #
    def addStartTime(self, runNumber = 0):
        startTimeSeconds = time.time()
        startTimeMicroseconds = datetime.datetime.fromtimestamp(startTimeSeconds)
        startTime = startTimeMicroseconds.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        if runNumber not in self.dataMap[self.curModuleName][self.curFuncName]["runData"]:
            self.dataMap[self.curModuleName][self.curFuncName]["runData"][runNumber] = {}
        
        self.dataMap[self.curModuleName][self.curFuncName]["runData"][runNumber]["startTime"] = startTime


    # addEndTime
    #
    # Add the current time to the end time of the current run of the current
    # cached module and function and compute the run time.
    #
    # Arguments:
    #
    #     runNumber - test run mumber (default 0)
    #
    def addEndTime(self, runNumber = 0):
        endTimeSeconds = time.time()
        endTimeMicroseconds = datetime.datetime.fromtimestamp(endTimeSeconds)
        endTime = endTimeMicroseconds.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        self.dataMap[self.curModuleName][self.curFuncName]["runData"][runNumber]["endTime"]= endTime

        start_time = datetime.datetime.strptime(self.dataMap[self.curModuleName][self.curFuncName]["runData"][runNumber]["startTime"], "%Y-%m-%d %H:%M:%S.%f")
        end_time = datetime.datetime.strptime(endTime, "%Y-%m-%d %H:%M:%S.%f")
        time_diff = end_time - start_time
        self.dataMap[self.curModuleName][self.curFuncName]["runData"][runNumber]["timeOfRun"] = str(time_diff)

    # addTestSuiteStartTime
    #
    # Add the current time as the start time of the current test suite.
    #
    def addTestSuiteStartTime(self):
        startTimeSeconds = time.time()
        startTimeMicroseconds = datetime.datetime.fromtimestamp(startTimeSeconds)
        startTime = startTimeMicroseconds.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        self.summary["startTime"] = startTime

    # addTestSuiteEndTime
    #
    # Add the current time as the end time of the current test suite and compute
    # the run time.
    #
    def addTestSuiteEndTime(self):
        endTimeSeconds = time.time()
        endTimeMicroseconds = datetime.datetime.fromtimestamp(endTimeSeconds)
        endTime = endTimeMicroseconds.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        self.summary["endTime"] = endTime

        start_time = datetime.datetime.strptime(self.summary["startTime"], "%Y-%m-%d %H:%M:%S.%f")
        end_time = datetime.datetime.strptime(endTime, "%Y-%m-%d %H:%M:%S.%f")
        time_diff = end_time - start_time
        self.summary["timeOfRun"] = str(time_diff)

    # addTestStatus
    #
    # Add the status of the test to the last run of the cached current function
    # of the current module. Update the total number of tests run in the test
    # suite and the number of tests of the given status
    # (SUCCESS/FAILED/SKIPPED).
    #
    # Arguments:
    #     status - status of test.
    #
    def addTestStatus(self, status):
        # adding test status
        self.dataMap[self.curModuleName][self.curFuncName]["runData"][-1]["status"] = status

        # incrememnt test count
        self.summary["numberOfTestsRun"] += 1
        if status == "SUCCESS":
            self.summary["testsPassed"] +=1
        elif status == "FAILED":
            self.summary["testsFailed"] +=1
        else:
            self.summary["testsSkipped"] +=1

    # addMessage
    #
    # Arguments:
    #
    #     message - message to add the the last run of the cached module and
    # function.
    #
    def addMessage(self, message):
        self.dataMap[self.curModuleName][self.curFuncName]["runData"][-1]["message"] = message

    # deleteEntry
    #
    # Delete the run entry for the cached module and function.
    #
    # Arguments:
    #
    #     runNumber - run data to delete
    def deleteEntry(self, runNumber):
        self.dataMap[self.curModuleName][self.curFuncName]["runData"][runNumber]["startTime"] = ""
        self.dataMap[self.curModuleName][self.curFuncName]["runData"][runNumber]["endTime"] = ""
        
    # updateMultiRun
    #
    # Set the multirun flag on the cached module and function.
    def updateMultiRun(self):
        self.dataMap[self.curModuleName][self.curFuncName]["isMultiRun"] = 1

    # getFuncDic
    #
    # Get the function dictionaru of the cached module and function.
    def getFuncDic(self):
        return self.dataMap[self.curModuleName][self.curFuncName]

    # addSummary
    #
    # Add summarydata to dataMapFinal["testSuite"] from dataMap.
    #
    def addSummary(self):
        self.dataMapFinal["testSuite"] = dict(self.dataMap)
        self.dataMapFinal = OrderedDict([("summary", self.summary)] + list(self.dataMapFinal.items()))

    # sortInDescendingOrderOfTime
    #
    # Sort items in dataMapFinal
    #
    def sortInDescendingOrderOfTime(self):

        # local functions
        def convertToDatetime(timeStr):
            try:
                return datetime.datetime.strptime(timeStr, "%H:%M:%S.%f")
            except:
                return datetime.datetime.strptime(timeStr, "%H:%M:%S")

        def maxTimeOfRun(tData):
            return max(convertToDatetime(run["timeOfRun"]) for run in tData["runData"])

        # logic
        data = self.dataMapFinal["testSuite"]
        mapOfSortedTests = OrderedDict()

        # sorting tests within each module based on timeofrun
        for module in data:
            sortedTestKeys = sorted(data[module].keys(), key=lambda x: maxTimeOfRun(data[module][x]), reverse=True)
            mapOfSortedTests[module] = {t: data[module][t] for t in sortedTestKeys}
            
        # sorting each module based on longest running test
        temp = sorted(mapOfSortedTests.keys(), key=lambda x: convertToDatetime(mapOfSortedTests[x][next(iter(mapOfSortedTests[x]))]["runData"][0]["timeOfRun"]), reverse=True)
        mapOfSortedModules = {module: mapOfSortedTests[module] for module in temp}

        return mapOfSortedModules

    # saveMapToJson
    #
    # Save map to JSON
    #
    # Arguments:
    #
    #     intermediate - flag to write non-final or final data map to regular
    #                    path
    #     compiled     - flag to write to compiled data to compiled path.
    #
    def saveMapToJson(self, intermediate=0, compiled=0):
        if compiled:
            # saving compiled version
            with open(self.jsonFilePathCompiled, 'w') as json_file:
                json.dump(self.dataMapFinal, json_file, indent=4) 
        else:
            # saving full version
            if intermediate == 1:
                with open(self.jsonFilePath, 'w') as json_file:
                    json.dump(self.dataMap, json_file, indent=4) 
            else:
                with open(self.jsonFilePath, 'w') as json_file:
                    json.dump(self.dataMapFinal, json_file, indent=4) 

    # sortDataMap
    #
    # sort Final data map and save it to JSON on the regular path.
    #
    def sortDataMap(self):
        # sort the dictionary
        self.dataMapFinal["testSuite"] = self.sortInDescendingOrderOfTime()
        self.saveMapToJson()


    


