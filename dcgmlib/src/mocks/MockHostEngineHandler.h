#ifndef MOCKHOSTENGINEHANDLER_H
#define MOCKHOSTENGINEHANDLER_H

#include "dcgm_structs.h"

#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

class MockHostEngineHandler
{
private:
    struct State
    {
        bool strictEntityLists = true;
        std::map<std::pair<dcgm_field_entity_group_t, dcgm_field_eid_t>, DcgmEntityStatus_t> entityStatuses;
        std::map<std::pair<int, dcgm_field_entity_group_t>, dcgmReturn_t> entityListReturns;
        std::map<std::pair<int, dcgm_field_entity_group_t>, std::vector<dcgmGroupEntityPair_t>> entityLists;
    };

    std::shared_ptr<State> m_state;

public:
    MockHostEngineHandler()
        : m_state(std::make_shared<State>())
        , strictEntityLists(m_state->strictEntityLists)
        , entityStatuses(m_state->entityStatuses)
        , entityListReturns(m_state->entityListReturns)
        , entityLists(m_state->entityLists)
    {}

    MockHostEngineHandler(MockHostEngineHandler const &other)
        : m_state(other.m_state)
        , strictEntityLists(m_state->strictEntityLists)
        , entityStatuses(m_state->entityStatuses)
        , entityListReturns(m_state->entityListReturns)
        , entityLists(m_state->entityLists)
    {}

    MockHostEngineHandler &operator=(MockHostEngineHandler const &) = delete;
    MockHostEngineHandler(MockHostEngineHandler &&)                 = delete;
    MockHostEngineHandler &operator=(MockHostEngineHandler &&)      = delete;

    bool &strictEntityLists;
    std::map<std::pair<dcgm_field_entity_group_t, dcgm_field_eid_t>, DcgmEntityStatus_t> &entityStatuses;
    std::map<std::pair<int, dcgm_field_entity_group_t>, dcgmReturn_t> &entityListReturns;
    std::map<std::pair<int, dcgm_field_entity_group_t>, std::vector<dcgmGroupEntityPair_t>> &entityLists;

    DcgmEntityStatus_t GetEntityStatus(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId)
    {
        auto const key = std::make_pair(entityGroupId, entityId);
        if (auto it = entityStatuses.find(key); it != entityStatuses.end())
        {
            return it->second;
        }

        return DcgmEntityStatusOk;
    }

    dcgmReturn_t GetAllEntitiesOfEntityGroup(int activeOnly,
                                             dcgm_field_entity_group_t entityGroupId,
                                             std::vector<dcgmGroupEntityPair_t> &entities)
    {
        auto const key = std::make_pair(activeOnly, entityGroupId);
        entities.clear();

        if (auto retIt = entityListReturns.find(key); retIt != entityListReturns.end())
        {
            if (retIt->second != DCGM_ST_OK)
            {
                return retIt->second;
            }
        }

        if (auto it = entityLists.find(key); it != entityLists.end())
        {
            entities = it->second;
        }
        else if (strictEntityLists)
        {
            throw std::logic_error("Unexpected GetAllEntitiesOfEntityGroup() call");
        }

        return DCGM_ST_OK;
    }
};

#endif
