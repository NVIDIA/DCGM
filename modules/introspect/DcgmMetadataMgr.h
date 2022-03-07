/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef _NVCM_METADATA_H
#define _NVCM_METADATA_H

#include "DcgmCoreProxy.h"
#include "DcgmMutex.h"
#include "DcgmStatCollection.h"
#include <condition_variable>
#include <map>
#include <shared_mutex>
#include <sstream>
#include <string>


/******************************************************************
 * Class to implement the introspection module
 ******************************************************************/
class DcgmMetadataManager
{
public:
    enum StatContext
    {
        STAT_CONTEXT_INVALID = 0,
        STAT_CONTEXT_FIELD,
        STAT_CONTEXT_FIELD_GROUP,
        STAT_CONTEXT_ALL_FIELDS,
        STAT_CONTEXT_PROCESS,

        STAT_CONTEXT_COUNT,
    };

    /**
     * This struct identifies the context of a metadata stat.
     */
    class ContextKey
    {
    public:
        StatContext context;

        // field ID or field group ID if needed
        unsigned long long contextId;

        // whether this represents an aggregate of all context locations (gpus + global)
        bool aggregate;

        // one of DCGM_FS_? from dcgm_fields.h
        // is only needed if aggregate == false
        int fieldScope;
        unsigned int gpuId; // only needed if fieldScope == DCGM_FS_DEVICE

        // empty constructor for null object
        ContextKey();

        // constructor for Process context
        explicit ContextKey(StatContext context);

        // constructor for All Fields context
        ContextKey(StatContext context, bool aggregate, int fieldScope = -1, unsigned int gpuId = 0);

        // constructor for Field and Field group context
        ContextKey(StatContext context,
                   unsigned long long contextId,
                   bool aggregate,
                   int fieldScope     = -1,
                   unsigned int gpuId = 0);

        bool operator==(const ContextKey &other) const;

        // Used internally to tell if the storage location of this context
        bool isGlobalStat() const;
        bool isGpuStat() const;

        std::string str() const;
    };

    typedef struct
    {
        double total; // kernel + user
        double kernel;
        double user;
    } CpuUtil;

    /**
     * This struct contains stats that only make sense when paired with
     * a unit measure of time (meanFrequencyUsec)
     */
    typedef struct
    {
        // the total amount of time, in usec, that has been spent updating the specified fields
        long long totalEverUpdateUsec;

        // the mean update frequency of all specified fields (all field watches)
        long long meanFrequencyUsec;

        // the sum of every specified field's most recent execution time after it
        // has been normalized to "frequencyUsec".
        // This is roughly how long it takes to update all specified fields every "meanFrequencyUsec"
        double recentUpdateUsec;
    } ExecTimeInfo;

    explicit DcgmMetadataManager(DcgmCoreProxy *dcc);
    virtual ~DcgmMetadataManager();

    /*************************************************************************/
    /*
     * Get the total amount of memory, in bytes, that is currently being used
     * by the given context.
     *
     * context           IN: specify the context to retrieve memory info for (see below)
     *     Valid StatContext in the given context:
     *
     *     STAT_CONTEXT_FIELD,
     *     STAT_CONTEXT_FIELD_GROUP,
     *     STAT_CONTEXT_ALL_FIELDS:
     *         - gives the memory used to store all the specified fields under the given context
     *
     *     STAT_CONTEXT_PROCESS:
     *         - gives the memory used by the hostengine process
     *
     * pTotalBytesUsed  OUT: Total amount of memory used in bytes.  Only valid if return is 0.
     *                       This may be 0 for a watched field if the hostengine has not retrieved that field yet.
     * waitIfNoData      IN: if no metadata is gathered wait till this occurs (!0)
     *                       or return DCGM_ST_NO_DATA (0)
     *
     * Returns: 0 on success
     *         <0 on error. See DCGM_ST_? enums
     *
     */
    dcgmReturn_t GetBytesUsed(ContextKey context, long long *pTotalBytesUsed, bool waitIfNoData = true);

    /*************************************************************************/
    /*
     * Get introspection info about the execution time taken while updating the fields in
     * the given context.
     *
     * context           IN: specify the context to retrieve execution time for (see below)
     *     Valid StatContext in the given context:
     *
     *     STAT_CONTEXT_FIELD,
     *     STAT_CONTEXT_FIELD_GROUP,
     *     STAT_CONTEXT_ALL_FIELDS:
     *         - gives execution time info about all the field instances contained in the given context
     *
     * ExecTimeInfo     OUT: See \ref ExecTimeInfo struct for explanation of the return value.
     *                       Only valid if return is 0.
     * waitIfNoData      IN: if no metadata is gathered wait till this occurs (!0)
     *                       or return DCGM_ST_NO_DATA (0)
     *
     * Returns: 0 on success
     *         <0 on error. See DCGM_ST_? enums
     *
     */
    dcgmReturn_t GetExecTime(ContextKey context, ExecTimeInfo *execTime, bool waitIfNoData = true);

    /*************************************************************************/
    /**
     * Get the current CPU utilization of the DCGM host engine process.
     *
     * cpuUtil                      OUT: See \ref CpuUtil struct for explanation of all the return values
     * waitIfNoData                  IN: if no metadata is gathered wait till this occurs (!0)
     *                                   or return DCGM_ST_NO_DATA (0)
     * Returns: 0 on success
     *         <0 on error. See DCGM_ST_? enums
     */
    dcgmReturn_t GetCpuUtilization(CpuUtil *cpuUtil, bool waitIfNoData = true);

    /**
     * This method is used to manually tell the the metadata manager to update all DCGM metadata.
     * This is normally performed automatically on an interval that can be set with \ref SetRunInterval.
     *
     * @param waitForUpdate         IN: Whether or not to wait for the update loop to
     *                                  complete before returning to the caller
     *
     * @return
     *        - \ref DCGM_ST_OK     Unconditionally
     */
    dcgmReturn_t UpdateAll(int waitForUpdate);

private:
    // forward declare some internal classes
    class AggregateFunctor;

    static const std::string STAT_CONTEXT_STRINGS[STAT_CONTEXT_COUNT];

    enum FieldMetadataType
    {
        FIELD_MT_CUR_BYTES_USED = 0,
        FIELD_MT_TOTAL_EXEC_TIME_USEC,
        FIELD_MT_TOTAL_FETCH_COUNT,
        FIELD_MT_MEAN_UPDATE_FREQ_USEC,
        FIELD_MT_RECENT_EXEC_TIME_USEC,

        // the number of field instances that are represented by the context,
        // ex: with a field watched on 3 gpus this count would be 3
        // ex: with a field group with 2 fields, each watched on 2 gpus, this count would be 4
        FIELD_MT_AGGR_INSTANCE_COUNT,

        FIELD_MT_COUNT,
    };
    static const std::string FIELD_METADATA_TYPE_STRINGS[FIELD_MT_COUNT];

    enum ProcessMetadataType
    {
        PROCESS_MT_VM_RSS_KB = 0,
        PROCESS_MT_VM_SWAP_KB,
        PROCESS_MT_REAL_RAM_KB,
        PROCESS_MT_TICKS_UTIME,
        PROCESS_MT_TICKS_STIME,
        PROCESS_MT_DEVICE_TICKS_TOTAL,
        PROCESS_MT_CPU_UTIL_UTIME,
        PROCESS_MT_CPU_UTIL_STIME,

        PROCESS_MT_COUNT,
    };
    static const std::string PROCESS_METADATA_TYPE_STRINGS[PROCESS_MT_COUNT];

    /*************************************************************************/
    /*
     * Execute function (usually a functor) for each GPU field that is currently watched.
     *
     * callbackFn  IN: function to be called. Must have signature
     *                 "dcgmReturn_t fn(unsigned short fieldId, unsigned int gpuId)"
     * fieldIds    IN: (optional) a subset of fields to iterate over instead of all fields
     *
     * Returns 0 on success
     *        <0 on error for the last error that occurred when calling the callbackFn. See DCGM_ST_? #defines
     */
    template <typename Fn>
    dcgmReturn_t ForEachWatchedGpuField(Fn *callbackFn,
                                        std::vector<unsigned short> *fieldIds = nullptr,
                                        unsigned int *const onlyOnThisGpuId   = nullptr);

    /*************************************************************************/
    /*
     * Execute functions (usually functors) for each GPU field that is currently watched.
     *
     * callbackFns  IN: functions to be called. Must have signature
     *                  "dcgmReturn_t fn(unsigned short fieldId, unsigned int gpuId)"
     * fieldIds     IN: (optional) a subset of fields to iterate over instead of all fields
     *
     * Returns 0 on success
     *        <0 on error for the last error that occurred when calling the callbackFn's. See DCGM_ST_? #defines
     */
    template <typename Fn>
    dcgmReturn_t ForEachWatchedGpuField(std::vector<Fn *> const &callbackFns,
                                        std::vector<unsigned short> *fieldIds = nullptr,
                                        unsigned int *const onlyOnThisGpuId   = nullptr);

    /*************************************************************************/
    /*
     * Execute function (usually a functor) for each global field that is currently watched.
     *
     * callbackFn  IN: function to be called. Must have signature
     *                 "dcgmReturn_t fn(unsigned short fieldId)"
     * fieldIds    IN: (optional) a subset of fields to iterate over instead of all fields
     *
     * Returns 0 on success
     *        <0 on error for the last error that occurred when calling the callbackFn. See DCGM_ST_? #defines
     */
    template <typename Fn>
    dcgmReturn_t ForEachWatchedGlobalField(Fn *callbackFn, std::vector<unsigned short> *fieldIds = nullptr);

    /*************************************************************************/
    /*
     * Execute functions (usually functors) for each global field that is currently watched.
     *
     * callbackFns  IN: functions to be called. Must have signature
     *                  "dcgmReturn_t fn(unsigned short fieldId)"
     * fieldIds     IN: (optional) a subset of fields to iterate over instead of all fields
     *
     * Returns 0 on success
     *        <0 on error for the last error that occurred when calling the callbackFn's. See DCGM_ST_? #defines
     */
    template <typename Fn>
    dcgmReturn_t ForEachWatchedGlobalField(std::vector<Fn *> const &callbackFns,
                                           std::vector<unsigned short> *fieldIds = nullptr);

    /*************************************************************************/
    /*
     * Execute function (usually a functor) for each field that is currently watched globally
     * or on at least 1 GPU.
     *
     * callbackFns  IN: function to be called. Must have signature
     *                  "dcgmReturn_t fn(unsigned short fieldId)"
     * fieldIds     IN: (optional) a subset of fields to iterate over instead of all fields
     *
     * Returns 0 on success
     *        <0 on error for the last error that occurred when calling the callbackFn. See DCGM_ST_? #defines
     */
    template <typename Fn>
    dcgmReturn_t ForEachWatchedField(Fn *callbackFn, std::vector<unsigned short> *fieldIds = nullptr);

    /*************************************************************************/
    /*
     * Execute functions (usually functors) for each field that is currently watched globally
     * or on at least 1 GPU.
     *
     * callbackFns  IN: functions to be called. Must have signature
     *                  "dcgmReturn_t fn(unsigned short fieldId)"
     * fieldIds     IN: (optional) a subset of fields to iterate over instead of all fields
     *
     * Returns 0 on success
     *        <0 on error for the last error that occurred when calling the callbackFn's. See DCGM_ST_? #defines
     */
    template <typename Fn>
    dcgmReturn_t ForEachWatchedField(std::vector<Fn *> const &callbackFns,
                                     std::vector<unsigned short> *fieldIds = nullptr);

    // A key that uniquely identifies a metadata stat in a stat collection.
    // it can be translated into the string key that the stat collection uses.
    class StatKey
    {
    public:
        ContextKey cKey;
        union _mType
        {
            FieldMetadataType field;
            ProcessMetadataType process;

            _mType(FieldMetadataType field)
                : field(field) {};
            _mType(ProcessMetadataType process)
                : process(process) {};
        } mType;

        template <typename MetadataType>
        StatKey(ContextKey cKey, MetadataType mType)
            : cKey(cKey)
            , mType(mType)
        {}
        std::string str() const;
    };

    /* constants */
    static const timelib64_t CPU_AVG_INTERVAL_USEC = 1000000; // 1 second, interval to average CPU utilization

    // how many pieces of a single metadata type we should store.
    // 1 = only present value, 2 = 1 past, 1 present, etc
    static const unsigned int DEFAULT_MAX_KEEP_ENTRIES = 2;

    // by default, don't limit metadata storage by timestamp.  Rely on count instead.
    static const unsigned int DEFAULT_OLDEST_KEEP_TIMESTAMP = 0;

    /* variables */
    unsigned int m_runIntervalMs; // how often to wait between cycles of metadata gathering
    bool m_currentlyUpdating;     // set to true when an update loop is occuring and false when the
                                  // update thread goes to sleep.
    bool m_startUpdateTriggered;  // A variable to check during the m_startUpdateCondition condVar waiting loops
    std::uint64_t m_updateLoopId; // identifier for an update loop iteration that changes the moment that a loop
                                  // finishes. This changes in no guaranteed order.

    timelib64_t m_startOfCurUpdateLoop;

    // If a metadata type should store more than the default entries we store it here.
    // storing more metadata is needed for some fields that are used to calculate averages
    std::map<ProcessMetadataType, timelib64_t> m_processTypeToOldestKeepTimestamp;
    std::map<ProcessMetadataType, int> m_processTypeToMaxKeepEntries;
    std::map<FieldMetadataType, timelib64_t> m_fieldTypeToOldestKeepTimestamp;
    std::map<FieldMetadataType, int> m_fieldTypeToMaxKeepEntries;

    DcgmCoreProxy *m_coreProxy; /* Cached pointer to the core proxy manager. Not owned by this class */
    DcgmStatCollection *m_statCollection;

    // the functors that are used for performing aggregations
    // they will be executed in the same order that they are in this vector so
    // if a functor is dependent on an earlier aggregated FieldMetadataType it should
    // be inserted after that aggregator
    std::vector<AggregateFunctor *> m_aggregationFunctors;

    /* functions */

    dcgmReturn_t getCpuUtilizationForHostengine(CpuUtil *cpuUtil);
    dcgmReturn_t getHostengineBytesUsed(ContextKey context, long long *bytesUsed, bool waitIfNoData);
    dcgmReturn_t getFieldStatBytesUsed(ContextKey context, long long *bytesUsed, bool waitIfNoData);

    mcollect_value_p getStatMeasurement(StatKey sKey);

    // same as "getStat" but returns DCGM_ST_STALE_DATA if the stat trying to be retrieved
    // hasn't been updated since the start of the last update loop.  This is useful in making
    // sure that the stat being relied on is being updated regularly
    template <typename T>
    dcgmReturn_t getRecentStat(StatKey sKey, T *metadata);

    template <typename T>
    dcgmReturn_t getStat(StatKey sKey, T *metadata);

    template <typename T>
    dcgmReturn_t recordStat(StatKey sKey, const T &val);

    timelib64_t getOldestKeepTimestamp(ProcessMetadataType mType);
    timelib64_t getOldestKeepTimestamp(FieldMetadataType mType);

    int getMaxKeepEntries(ProcessMetadataType mType);
    int getMaxKeepEntries(FieldMetadataType mType);

    /**
     * returns true if the context can be used to identify a stat
     */
    bool isValidContext(ContextKey c);
    bool isValidMultiFieldContext(ContextKey c);

    /**
     * implementation for getting metadata but waiting if no data is present
     * getMetadataFn is generally a functor that tries to retrieve metadata and returns a status
     * If the returned status indicates no data then we wait a couple times before returning.
     * getMetadataFn must be callable with no args and return a dcgmReturn_t
     */
    template <typename Fn>
    dcgmReturn_t getMetadataWithWait(Fn getMetadataFn, bool waitIfNoData);

    /**
     * Methods used for updating the metadata stat collection that are run from the main update loop
     */
    void retrieveFieldInstanceData();
    void postProcessFieldInstanceData();

    void aggregateFieldData();
    void postProcessFieldData();

    void aggregateFieldGroupInstanceData();
    void postProcessFieldGroupInstanceData();

    void aggregateFieldGroupData();
    void postProcessFieldGroupData();

    void aggregateAllFieldsInstanceData();
    void postProcessAllFieldsInstanceData();

    void aggregateAllFieldsData();
    void postProcessAllFieldsData();

    void retrieveProcessData();
    void postProcessProcessData();

    template <typename Fn>
    dcgmReturn_t forEachInstanceContext(Fn *fn, std::vector<unsigned short> *fieldIds = nullptr);

    /**
     * Roughly equivalent to the following pseudo-code:
     *
     * for aggregator in functors:
     *     for fieldInstanceType in allFieldInstanceTypes: # "global" or a GPU#
     *         aggregator.startAggregation()
     *
     *         for fieldId in fieldIds (or all fields if NULL):
     *             aggregator(makeFieldInstance(fieldId, fieldInstanceType))
     *
     *         aggregator.recordAggregation()
     *
     * partialAggregateContext: A context key with the "context" and "contextId" fields filled out.
     *                          This identifies where the aggregators will save their aggregations
     */
    dcgmReturn_t aggregateOverFieldInstances(ContextKey partialAggregateContext,
                                             std::vector<AggregateFunctor *> functors,
                                             std::vector<unsigned short> *fieldIds = nullptr);
    /**
     * Roughly equivalent to the following pseudo-code:
     *
     * for aggregator in functors:
     *     aggregator.startAggregation()
     *
     *     for fieldId in fieldIds (or all fields if NULL):
     *         for fieldInstanceType in allFieldInstanceTypes: # "global" or a GPU#
     *            aggregator(makeFieldInstance(fieldId, fieldInstanceType))
     *
     *     aggregator.recordAggregation()
     *
     * partialAggregateContext: A context key with the "context" and "contextId" fields filled out.
     *                          This identifies where the aggregators will save their aggregations
     *
     * This sort of aggregation ends up duplicating a lot of work
     * with the "aggregateOverFieldInstances" function but makes programming simpler.
     *
     * An aggregation system could also be made to aggregate, for example, from field group
     * instances up to a field group instead of from all field instances in the
     * field group but then the aggregators would have to take into account the number
     * of field instances that a field group instance has (for "avg" calculations or similar)
     */
    dcgmReturn_t aggregateOverFields(ContextKey partialAggregateContext,
                                     std::vector<AggregateFunctor *> functors,
                                     std::vector<unsigned short> *fieldIds = nullptr);

    /**
     * Roughly equivalent to the following pseudo-code:
     *
     * for aggregator in functors:
     *     for fieldGroup in allFieldGroups:
     *         for fieldInstanceType in all fieldInstanceTypes: # "global" or a GPU number
     *             aggregator.startAggregation()
     *             for fieldInstance in fieldGroup that has fieldInstanceType:
     *                 aggregator(fieldInstance)
     *             aggregator.recordAggregation()
     */
    dcgmReturn_t aggregateOverFieldGroupInstances(std::vector<AggregateFunctor *> functors);

    /**
     * Roughly equivalent to the following pseudo-code:
     *
     * for each aggregator in functors:
     *     for fieldGroup in allFieldGroups:
     *         aggregator.startAggregation()
     *         for fieldInstance in fieldGroup:  # field + "global" or a field + GPU number
     *             aggregator(fieldInstance)
     *         aggregator.recordAggregation()
     */
    dcgmReturn_t aggregateOverFieldGroups(std::vector<AggregateFunctor *> functors);

    // methods that actually perform aggregation
    template <typename T>
    dcgmReturn_t aggregateBySumming(ContextKey cKey, FieldMetadataType mType);
    dcgmReturn_t aggregateFieldMeanUpdateFreq(unsigned short fieldId);
    dcgmReturn_t aggregateFieldRecentExecTime(unsigned short fieldId);

    // is introspection info being retrieved for this context?
    bool isContextWatched(ContextKey context);

    // get the time of the last stat inserted to this measurement
    dcgmReturn_t lastUpdateTime(StatKey sKey, timelib64_t *updateTimeUsec);

    /*
     * should be called by every public function that retrieves data from this manager.
     * This validates the context and checks that it is being watched.
     */
    dcgmReturn_t validateRetrievalContext(ContextKey context);

    // high level functions that are called from the main polling loop
    void retrieveFieldMemInfo();
    void retrieveFieldTotalExecTimeInfo();
    void retrieveFieldFetchCountInfo();

    // all private methods with prefix "retrieve" populate metadata without using existing metadata
    void retrieveOSInfo();
    void retrieveOSInfoFromProcSelfStatus();
    void retrieveOSInfoFromProcSelfStat();
    void retrieveOSInfoFromProcStat();

    // uses the total execution time of a field
    void generateFieldAvgExecTimeInfo();
    dcgmReturn_t generateAvgExecTimeForGlobalField(dcgm_field_meta_p fieldMeta, long long *fieldAvgExecTime);
    dcgmReturn_t generateAvgExecTimeForGpuField(unsigned int gpuId,
                                                dcgm_field_meta_p fieldMeta,
                                                long long *fieldAvgExecTime);

    dcgmReturn_t generateFieldRecentUpdateTime(unsigned short fieldId, unsigned int gpuId, int scope);

    void generateCpuUtilization();

    /**
     * Return the type that all measurement collections for the given metadata type should have
     */
    int mcollectTypeForMetadataType(FieldMetadataType mType) const;
    int mcollectTypeForMetadataType(ProcessMetadataType mType) const;

    /**
     * mType is something that maps to a MC_TYPE_? define via \ref mcollectTypeForMetadataType
     */
    template <typename MetadataType>
    dcgmReturn_t validateMcollectType(mcollect_value_p measurement, MetadataType mType);

    /*
     * Given a timeseries, get the diff from the second last value to the last value.
     * Returns DCGM_ST_NO_DATA when there is not at least two values in the timeseries.
     */
    static dcgmReturn_t getRecentStatDiff(timeseries_p ts, long long *diff);

    /*
     * Get the diff from the first value to the last value of a timeseries
     * Returns DCGM_ST_NO_DATA when there is not at least two values in the timeseries.
     */
    static dcgmReturn_t getFullStatDiff(timeseries_p ts, long long *diff);

    template <typename T>
    dcgmReturn_t getFullStatDiffForType(StatKey sKey, T *dStat);


    template <typename StatT, typename NormT, typename RetT>
    dcgmReturn_t getNormalizedLatestStat(const StatKey &statKey,
                                         const StatKey &normFromStatKey,
                                         const StatKey &normToStatKey,
                                         RetT *normalizedStat);


    template <typename MetadataType>
    void extractTimeseriesVal(void *val, timeseries_entry_p tsVal, MetadataType mType);

    /******************************************************************
     * Functors for aggregating
     ******************************************************************/

    // Interface for functors that aggregate stats up a level by being called
    // with a series of ContextKeys
    class AggregateFunctor
    {
    public:
        AggregateFunctor()
            : m_storageContext() {};
        virtual dcgmReturn_t operator()(ContextKey cKey) = 0;
        virtual void reset()                             = 0;

        /**
         * Must call this before starting aggregation.
         * storageContext is the context for where to store the aggregation
         * when it is done.  If this returns anything other than DCGM_ST_OK then
         * aggregation will be skipped for the storageContext
         */
        virtual dcgmReturn_t initAggregation(ContextKey storageContext)
        {
            reset();
            this->m_storageContext = storageContext;
            return DCGM_ST_OK;
        }

        virtual dcgmReturn_t recordIfOkay() = 0;

        virtual ~AggregateFunctor()
        {}

    protected:
        ContextKey m_storageContext;
    };

    template <typename ValType>
    class AggregateSumFunctor : public AggregateFunctor
    {
    public:
        AggregateSumFunctor(DcgmMetadataManager *mm, FieldMetadataType mType)
            : mm(mm)
            , mType(mType)
            , total(0)
        {
            reset();
        }

        virtual dcgmReturn_t operator()(ContextKey cKey)
        {
            // fail early if any previous iteration failed
            if (wasCalled && DCGM_ST_OK != status)
                return status;

            wasCalled = true;

            StatKey sKey(cKey, mType);

            ValType metadata;
            status = mm->getRecentStat(sKey, &metadata);
            if (DCGM_ST_OK != status)
            {
                return status;
            }

            total += metadata;
            return status;
        }

        virtual void reset()
        {
            status    = DCGM_ST_NO_DATA;
            total     = 0;
            wasCalled = false;
        }

        virtual dcgmReturn_t recordIfOkay()
        {
            if (this->m_storageContext == ContextKey())
            {
                PRINT_ERROR("", "aggregation was never initialized");
                return DCGM_ST_UNINITIALIZED;
            }

            // ok to not aggregate anything since sometimes we do an iteration of global fields on
            // a field group that only has device fields, for example
            if (!wasCalled)
            {
                PRINT_DEBUG("%s %d",
                            "nothing was summed for (%s) with MetadataType \"%d\" because no records were present",
                            this->m_storageContext.str().c_str(),
                            mType);
                return DCGM_ST_OK;
            }

            if (DCGM_ST_OK != status)
            {
                return status;
            }

            mm->recordStat(StatKey(this->m_storageContext, mType), total);
            return DCGM_ST_OK;
        }

    protected:
        DcgmMetadataManager *mm;

        // the metadata type that is going to be summed
        FieldMetadataType mType;

        // variable for accumulating the sum
        ValType total;

        // identifies the first error that happened when calling this aggregator
        dcgmReturn_t status;

        // flag to ensure that we return DCGM_ST_OK when told to record after aggregating over nothing
        // since this is different from receiving an error during aggregation like DCGM_ST_NO_DATA
        bool wasCalled;
    };

    template <typename StatT>
    class AggregateMeanFunctor : public AggregateSumFunctor<StatT>
    {
    public:
        AggregateMeanFunctor(DcgmMetadataManager *mm, FieldMetadataType mType)
            : AggregateSumFunctor<StatT>(mm, mType)
        {
            reset();
        }

        dcgmReturn_t operator()(ContextKey cKey)
        {
            AggregateSumFunctor<StatT>::operator()(cKey);
            count++;

            return this->status;
        }

        void reset()
        {
            AggregateSumFunctor<StatT>::reset();
            count = 0;
        }

        dcgmReturn_t recordIfOkay()
        {
            if (this->m_storageContext == ContextKey())
            {
                PRINT_ERROR("", "aggregation was never initialized");
                return DCGM_ST_UNINITIALIZED;
            }

            // ok to not aggregate anything since sometimes we do an iteration of global fields on
            // a field group that only has device fields, for example
            if (count == 0)
            {
                PRINT_DEBUG("", "never iterated over anything so cannot calculate mean (count == 0)");
                return DCGM_ST_OK;
            }

            if (this->status != DCGM_ST_OK)
            {
                return this->status;
            }

            StatT mean = (StatT)((double)this->total / (double)count);
            this->mm->recordStat(StatKey(this->m_storageContext, this->mType), mean);
            return DCGM_ST_OK;
        }

    private:
        unsigned int count;
    };

    /**
     * Calculate a normalized sum of all "mType" of a field where each stored instance of "mType"
     * is first normalized to "normalizeTo" based on that instance's "normalizeWithMType".
     * The sum is accumulated as a double to avoid truncation errors.
     * "StatT" is the data type of the stat being summed.
     * "NormalizeStatT" is the data type of the field used for normalizing.
     */
    template <typename StatT, typename NormalizeStatT>
    class AggregateNormalizedSumFunctor : public AggregateFunctor
    {
    public:
        AggregateNormalizedSumFunctor(DcgmMetadataManager *mm,
                                      FieldMetadataType mType,
                                      FieldMetadataType normalizeWithMType)
            : mm(mm)
            , mType(mType)
            , normalizeWithMType(normalizeWithMType)
            , normalizeTo(0)
        {
            reset();
        }

        virtual dcgmReturn_t initAggregation(ContextKey storageContext)
        {
            dcgmReturn_t st = AggregateFunctor::initAggregation(storageContext);
            if (DCGM_ST_OK != st)
            {
                DCGM_LOG_ERROR << "initAggregation returned " << errorString(st);
                return st;
            }

            StatKey sKey(this->m_storageContext, normalizeWithMType);

            st = mm->getRecentStat(sKey, &normalizeTo);
            if (DCGM_ST_OK != st)
            {
                PRINT_ERROR("%s",
                            "could not init aggregation functor because the stat \"%s\" was not retrieved recently. "
                            "It must be retrieved before this aggregation is done",
                            sKey.str().c_str());
                return st;
            }
            return DCGM_ST_OK;
        }

        dcgmReturn_t operator()(ContextKey cKey)
        {
            // fail early if previous iteration has failed
            if (wasCalled && DCGM_ST_OK != status)
                return status;

            wasCalled = true;

            StatKey fieldStatKey(cKey, mType);
            StatKey normalizeStatKey(cKey, normalizeWithMType);

            StatT instanceStat;
            NormalizeStatT normalizeFrom;

            status = mm->getRecentStat(fieldStatKey, &instanceStat);
            if (DCGM_ST_STALE_DATA == status)
            {
                PRINT_DEBUG("%s",
                            "skipping norm sum of record because it wasn't recorded recently: %s",
                            fieldStatKey.str().c_str());
                status = DCGM_ST_OK;
                return status;
            }
            if (DCGM_ST_OK != status)
                return status;

            status = mm->getStat(normalizeStatKey, &normalizeFrom);
            if (DCGM_ST_OK != status)
                return status;

            if (normalizeFrom == 0)
            {
                PRINT_ERROR("%s %d",
                            "context %s cannot be normalized because its \"normalize from\" type %d is 0.",
                            cKey.str().c_str(),
                            normalizeWithMType);
                status = DCGM_ST_BADPARAM;
                return status;
            }

            normalizedSum += instanceStat * (double)normalizeTo / (double)normalizeFrom;

            return DCGM_ST_OK;
        }

        dcgmReturn_t recordIfOkay()
        {
            if (this->m_storageContext == ContextKey())
            {
                PRINT_ERROR("", "aggregation was never initialized");
                return DCGM_ST_UNINITIALIZED;
            }

            // ok to not aggregate anything since sometimes we do an iteration of global fields on
            // a field group that only has device fields, for example
            if (!wasCalled)
            {
                PRINT_DEBUG(
                    "%s %d",
                    "nothing was \"normal sum\"ed for (%s) with MetadataType \"%d\" because no records were present",
                    this->m_storageContext.str().c_str(),
                    mType);
                return DCGM_ST_OK;
            }

            if (status != DCGM_ST_OK)
                return status;

            mm->recordStat(StatKey(this->m_storageContext, mType), normalizedSum);
            return status;
        }

        void reset()
        {
            normalizedSum = 0;
            status        = DCGM_ST_NO_DATA;
            wasCalled     = false;
        }

    private:
        DcgmMetadataManager *mm;

        /**
         * The following 3 member variables are used to calculate a "normalized sum" as the following:
         *
         * stat             = mm->getStat(StatKey(<ContextKey that was given>, mType))
         * normalizeFrom    = mm->getStat(StatKey(<ContextKey that was given>, normailizeWithMType))
         * normalizedSum    += stat * (normalizeTo / normalizeFrom)
         */

        // the metadata type that is being summed
        FieldMetadataType mType;

        // the metadata type that identifies the denominator for normalizing
        // before the stat is summed.
        FieldMetadataType normalizeWithMType;

        // the numerator value for normalizing
        NormalizeStatT normalizeTo;

        // the normalized sum that is eventually recorded.  Increases incrementally
        // as new ContextKeys are iterated over
        double normalizedSum;

        // identifies the first error that happened when this aggregator was called
        dcgmReturn_t status;

        // flag to ensure that we return DCGM_ST_OK when told to record after aggregating over nothing
        // since this is different from receiving an error during aggregation like DCGM_ST_NO_DATA
        bool wasCalled;
    };

    /******************************************************************
     * Functors for getting metadata out of this manager
     ******************************************************************/
    template <typename T>
    class GetStatFunctor
    {
    public:
        GetStatFunctor(DcgmMetadataManager *mm, StatKey sKey, T *metadata)
            : mm(mm)
            , sKey(sKey)
            , metadata(metadata)
        {}

        dcgmReturn_t operator()()
        {
            return mm->getStat(sKey, metadata);
        }

    private:
        DcgmMetadataManager *mm;
        StatKey sKey;
        T *metadata;
    };

    class GetProcessCpuUtilFunctor
    {
    public:
        GetProcessCpuUtilFunctor(DcgmMetadataManager *mm, CpuUtil *cpuUtil)
            : mm(mm)
            , cpuUtil(cpuUtil)
        {}

        dcgmReturn_t operator()()
        {
            return mm->getCpuUtilizationForHostengine(cpuUtil);
        }

    private:
        DcgmMetadataManager *mm;
        CpuUtil *cpuUtil;
    };


    /****************************************************************
     * Functors used when aggregating
     ***************************************************************/

    /**
     * For each field Id that this functor is called with it will
     * initialize, call, and record an aggregation of the field with each
     * of the given aggregation functors.
     */
    class FieldAggregator
    {
    public:
        FieldAggregator(DcgmMetadataManager *mm, std::vector<AggregateFunctor *> &functors)
            : mm(mm)
            , functors(functors)
        {}


        dcgmReturn_t operator()(unsigned short fieldId)
        {
            dcgmReturn_t st    = DCGM_ST_OK;
            dcgmReturn_t retSt = st;
            std::vector<unsigned short> fields;
            fields.push_back(fieldId);

            for (size_t i = 0; i < functors.size(); ++i)
            {
                AggregateFunctor *functor = functors.at(i);

                st = functor->initAggregation(ContextKey(STAT_CONTEXT_FIELD, fieldId, true));
                if (DCGM_ST_OK != st)
                {
                    retSt = st;
                    continue;
                }

                st = mm->forEachInstanceContext(functor, &fields);
                if (DCGM_ST_OK != st)
                {
                    retSt = st;
                    continue;
                }

                st = functor->recordIfOkay();
                if (DCGM_ST_OK != st)
                {
                    retSt = st;
                    continue;
                }
            }
            return retSt;
        };

    private:
        DcgmMetadataManager *mm;
        std::vector<AggregateFunctor *> &functors;
    };

    /**
     * When called with a field instance (field ID + maybe a gpu ID) it will translate
     * these args into a Context which will be passed to the given function
     */
    template <typename Fn>
    class FieldInstanceToContextAdaptor
    {
    public:
        FieldInstanceToContextAdaptor(Fn *fn)
            : fn(fn) {};

        dcgmReturn_t operator()(unsigned short fieldId, unsigned int gpuId = 0)
        {
            dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);
            if (!fieldMeta)
            {
                PRINT_ERROR("%u", "%u is an invallid field", fieldId);
                return DCGM_ST_BADPARAM;
            }

            ContextKey cKey(STAT_CONTEXT_FIELD, fieldId, false, fieldMeta->scope, gpuId);
            return (*fn)(cKey);
        };

    private:
        Fn *fn;
    };
};

template <typename Fn>
dcgmReturn_t DcgmMetadataManager::ForEachWatchedGpuField(Fn *callbackFn,
                                                         std::vector<unsigned short> *fieldIds,
                                                         unsigned int *const onlyOnThisGpuId)
{
    std::vector<Fn *> callbackFns;
    callbackFns.push_back(callbackFn);
    return ForEachWatchedGpuField(callbackFns, fieldIds, onlyOnThisGpuId);
}

template <typename Fn>
dcgmReturn_t DcgmMetadataManager::ForEachWatchedGpuField(std::vector<Fn *> const &callbackFns,
                                                         std::vector<unsigned short> *fieldIds,
                                                         unsigned int *const onlyOnThisGpuId)
{
    dcgmReturn_t st;
    dcgmReturn_t retSt = DCGM_ST_OK;

    std::vector<dcgmCoreWatchInfo_t> watchInfo;
    std::vector<unsigned int> gpuIds;

    if (onlyOnThisGpuId)
    {
        gpuIds.push_back(*onlyOnThisGpuId);
    }
    else
    {
        st = m_coreProxy->GetGpuIds(1, gpuIds);
        if (DCGM_ST_OK != st)
        {
            return st;
        }
    }

    for (auto gpuId = gpuIds.begin(); gpuId != gpuIds.end(); ++gpuId)
    {
        watchInfo.clear();
        st = m_coreProxy->PopulateGpuWatchInfo(watchInfo, *gpuId, fieldIds);
        if (st != DCGM_ST_OK)
        {
            continue;
        }

        for (auto field = watchInfo.begin(); field != watchInfo.end(); ++field)
        {
            for (auto &&func : callbackFns)
            {
                st = std::invoke(*func, (*field).fieldId, *gpuId);
                if (DCGM_ST_OK != st)
                {
                    retSt = st;
                }
            }
        }
    }

    return retSt;
}

// must be in header because of template
template <typename Fn>
dcgmReturn_t DcgmMetadataManager::ForEachWatchedGlobalField(Fn *callbackFn, std::vector<unsigned short> *fieldIds)
{
    std::vector<Fn *> callbackFns;
    callbackFns.push_back(callbackFn);
    return ForEachWatchedGlobalField(callbackFns, fieldIds);
}

// must be in header because of template
template <typename Fn>
dcgmReturn_t DcgmMetadataManager::ForEachWatchedGlobalField(std::vector<Fn *> const &callbackFns,
                                                            std::vector<unsigned short> *fieldIds)
{
    dcgmReturn_t st;
    dcgmReturn_t retSt = DCGM_ST_OK;

    std::vector<dcgmCoreWatchInfo_t> watchInfo;
    st = m_coreProxy->PopulateGlobalWatchInfo(watchInfo, fieldIds);
    if (st != DCGM_ST_OK)
    {
        return st;
    }

    for (auto field = watchInfo.begin(); field != watchInfo.end(); ++field)
    {
        for (auto &&func : callbackFns)
        {
            st = std::invoke(*func, ((*field).fieldId));
            if (DCGM_ST_OK != st)
            {
                retSt = st;
            }
        }
    }

    return retSt;
}

// must be in header because of template
template <typename Fn>
dcgmReturn_t DcgmMetadataManager::ForEachWatchedField(Fn *callbackFn, std::vector<unsigned short> *fieldIds)
{
    std::vector<Fn *> callbackFns;
    callbackFns.push_back(callbackFn);
    return ForEachWatchedField(callbackFns, fieldIds);
}

// must be in header because of template
template <typename Fn>
dcgmReturn_t DcgmMetadataManager::ForEachWatchedField(std::vector<Fn *> const &callbackFns,
                                                      std::vector<unsigned short> *fieldIds)
{
    dcgmReturn_t st;
    dcgmReturn_t retSt = DCGM_ST_OK;

    std::vector<dcgmCoreWatchInfo_t> watchInfo;
    st = m_coreProxy->PopulateWatchInfo(watchInfo, fieldIds);
    if (st != DCGM_ST_OK)
    {
        return st;
    }

    std::vector<unsigned int> gpuIds;
    st = m_coreProxy->GetGpuIds(1, gpuIds);
    if (DCGM_ST_OK != st)
    {
        return st;
    }

    for (auto info = watchInfo.begin(); info != watchInfo.end(); ++info)
    {
        if ((*info).scope == DCGM_FS_DEVICE)
        {
            // only run for one GPU even if watched on multiple GPUs
            auto gpuId = gpuIds.begin();
            if (gpuId != gpuIds.end())
            {
                for (auto &&func : callbackFns)
                {
                    st = std::invoke(*func, (*info).fieldId);
                    if (DCGM_ST_OK != st)
                    {
                        retSt = st;
                    }
                }
            }
        }
        else if ((*info).scope == DCGM_FS_GLOBAL)
        {
            for (auto &&func : callbackFns)
            {
                st = std::invoke(*func, (*info).fieldId);
                if (DCGM_ST_OK != st)
                {
                    retSt = st;
                }
            }
        }
    }

    return retSt;
}
#endif //_NVCM_INTROSPECTION_H
