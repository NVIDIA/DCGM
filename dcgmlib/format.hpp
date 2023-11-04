#ifndef DCGM_FORMAT_H
#define DCGM_FORMAT_H
#ifdef DcgmWatcherType_t
inline int format_as(DcgmWatcherType_t type) {  return static_cast<int>(type);}
#endif
inline int format_as(dcgmChipArchitecture_enum type) {  return static_cast<int>(type);}
inline int format_as(dcgmConfigType_enum type) {  return static_cast<int>(type);}
inline int format_as(dcgmDiagnosticLevel_t type) {  return static_cast<int>(type);}
//inline int format_as(dcgmEntityStatusType_enum type) {  return static_cast<int>(type);}
inline int format_as(dcgmGroupType_enum type) {  return static_cast<int>(type);}
inline int format_as(dcgmHealthSystems_enum type) {  return static_cast<int>(type);}
inline int format_as(dcgmModuleId_t type) {  return static_cast<int>(type);}
inline int format_as(dcgmModuleStatus_t type) {  return static_cast<int>(type);}
//inline int format_as(dcgmMutexSt type) {  return static_cast<int>(type);}

#ifdef dcgmNvLinkLinkState
inline int format_as(dcgmNvLinkLinkState type) {  return static_cast<int>(type);}
#endif
inline int format_as(dcgmOrder_enum type) {  return static_cast<int>(type);}
inline int format_as(dcgmPolicyCondition_enum type) {  return static_cast<int>(type);}
inline int format_as(dcgmPolicyValidation_enum type) {  return static_cast<int>(type);}
inline int format_as(dcgmPolicyAction_enum type) {  return static_cast<int>(type);}

inline int format_as(dcgmReturn_enum type) {  return static_cast<int>(type);}
inline int format_as(dcgm_field_entity_group_t type) {  return static_cast<int>(type);}

#ifdef nvmlReturn_enum
inline int format_as(nvmlReturn_enum type) {  return static_cast<int>(type);}
#endif
#endif
