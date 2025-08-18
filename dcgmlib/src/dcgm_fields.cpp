/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "dcgm_fields.h"

#include "DcgmStringHelpers.h"
#include "MurmurHash3.h"
#include "dcgm_fields_internal.hpp"
#include "hashtable.h"
#include <DcgmLogging.h>
#include <dcgm_nvml.h>

#include <fmt/format.h>
#include <malloc.h>
#include <string.h>

/* Not using an object on purpose in case this code needs to be ported to C */


/**
 * Width and unit enums for field values.
 */
enum UNIT
{
    DCGM_FIELD_UNIT_TEMP_C,      /* Temperature in Celcius */
    DCGM_FIELD_UNIT_POW_W,       /* Watts */
    DCGM_FIELD_UNIT_BW_KBPS,     /* KB/sec */
    DCGM_FIELD_UNIT_BW_MBPS,     /* MB/sec */
    DCGM_FIELD_UNIT_BW_MbPS,     /* Mb/sec */
    DCGM_FIELD_UNIT_MILLIJOULES, /* mJ (milliJoules) */
};

enum WIDTH
{
    DCGM_FIELD_WIDTH_40,
    DCGM_FIELD_WIDTH_20,
    DCGM_FIELD_WIDTH_7,
    DCGM_FIELD_WIDTH_5,
    DCGM_FIELD_WIDTH_16,
    DCGM_FIELD_WIDTH_10
};

/* Has this module been initialized? */
static int dcgmFieldsInitialized = 0;
hashtable_t dcgmFieldsKeyToIdMap = {};

/*****************************************************************************/
/* Static field information. Call DcgmFieldsPopulateOneFieldWithFormatting() to add entries to this table */
static dcgm_field_meta_t *dcgmFieldMeta[DCGM_FI_MAX_FIELDS] = { 0 };

/*****************************************************************************/

/**
 * The function returns int value of enum for width
 */
static int getWidthForEnum(enum WIDTH enumVal)
{
    switch (enumVal)
    {
        case DCGM_FIELD_WIDTH_40:
            return 40;
            break;

        case DCGM_FIELD_WIDTH_20:
            return 20;
            break;

        case DCGM_FIELD_WIDTH_7:
            return 7;
            break;

        case DCGM_FIELD_WIDTH_5:
            return 5;
            break;

        case DCGM_FIELD_WIDTH_16:
            return 16;
            break;

        case DCGM_FIELD_WIDTH_10:
            return 10;
            break;

        default:
            return 10;
            break;
    }
}

/**
 * The function returns string value of enum for Units.
 */
static const char *getTextForEnum(enum UNIT enumVal)
{
    switch (enumVal)
    {
        case DCGM_FIELD_UNIT_TEMP_C:
            return " C";
            break;

        case DCGM_FIELD_UNIT_POW_W:
            return " W";
            break;

        case DCGM_FIELD_UNIT_BW_KBPS:
            return "KB/s";
            break;

        case DCGM_FIELD_UNIT_BW_MBPS:
            return "MB/s";
            break;

        case DCGM_FIELD_UNIT_BW_MbPS:
            return "Mb/s";
            break;

        case DCGM_FIELD_UNIT_MILLIJOULES:
            return " mJ";
            break;

        default:
            return "";
            break;
    }
}

/*****************************************************************************/
static int DcgmFieldsPopulateOneFieldWithFormatting(unsigned short fieldId,
                                                    char fieldType,
                                                    unsigned char size,
                                                    const char *tag,
                                                    int scope,
                                                    int nvmlFieldId,
                                                    const char *shortName,
                                                    const char *unit,
                                                    dcgm_field_entity_group_t entityLevel,
                                                    short width)
{
    dcgm_field_meta_t *fieldMeta = 0;

    if (!fieldId || fieldId >= DCGM_FI_MAX_FIELDS)
        return -1;

    fieldMeta = (dcgm_field_meta_t *)malloc(sizeof(*fieldMeta));
    if (!fieldMeta)
        return -3; /* Out of memory */

    memset(fieldMeta, 0, sizeof(*fieldMeta));

    fieldMeta->fieldId   = fieldId;
    fieldMeta->fieldType = fieldType;
    fieldMeta->size      = size;
    SafeCopyTo(fieldMeta->tag, tag);
    fieldMeta->scope       = scope;
    fieldMeta->nvmlFieldId = nvmlFieldId;
    fieldMeta->entityLevel = entityLevel;

    fieldMeta->valueFormat = (dcgm_field_output_format_t *)malloc(sizeof(*fieldMeta->valueFormat));
    if (NULL == fieldMeta->valueFormat)
    {
        free(fieldMeta);
        return -3;
    }
    memset(fieldMeta->valueFormat, 0, sizeof(*fieldMeta->valueFormat));

    SafeCopyTo(fieldMeta->valueFormat->shortName, shortName);
    SafeCopyTo(fieldMeta->valueFormat->unit, unit);
    fieldMeta->valueFormat->width = width;
    fieldMeta->entityLevel        = entityLevel;

    dcgmFieldMeta[fieldMeta->fieldId] = fieldMeta;
    return 0;
}

/* Do static initialization of the global field list */
static int DcgmFieldsPopulateFieldTableWithFormatting(void)
{
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DRIVER_VERSION,
                                             DCGM_FT_STRING,
                                             0,
                                             "driver_version",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "DRVER",
                                             "#",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_NVML_VERSION,
                                             DCGM_FT_STRING,
                                             0,
                                             "nvml_version",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "NVVER",
                                             "#",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROCESS_NAME,
                                             DCGM_FT_STRING,
                                             0,
                                             "process_name",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "PRNAM",
                                             "",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_CUDA_DRIVER_VERSION,
                                             DCGM_FT_INT64,
                                             0,
                                             "cuda_driver_version",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "CDVER",
                                             "#",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_COUNT,
                                             DCGM_FT_INT64,
                                             8,
                                             "device_count",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "DVCNT",
                                             "",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NAME,
                                             DCGM_FT_STRING,
                                             0,
                                             "name",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "DVNAM",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_BRAND,
                                             DCGM_FT_STRING,
                                             0,
                                             "brand",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "DVBRN",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVML_INDEX,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvml_index",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVIDX",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5)); //
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SERIAL,
                                             DCGM_FT_STRING,
                                             0,
                                             "serial_number",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "SRNUM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_AFFINITY_0,
                                             DCGM_FT_INT64,
                                             8,
                                             "cpu_affinity_0",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "CAFF0",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_AFFINITY_1,
                                             DCGM_FT_INT64,
                                             8,
                                             "cpu_affinity_1",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "CAFF1",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_AFFINITY_2,
                                             DCGM_FT_INT64,
                                             8,
                                             "cpu_affinity_2",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "CAFF2",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_AFFINITY_3,
                                             DCGM_FT_INT64,
                                             8,
                                             "cpu_affinity_3",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "CAFF3",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_UUID,
                                             DCGM_FT_STRING,
                                             0,
                                             "uuid",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "UUID#",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_40));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MINOR_NUMBER,
                                             DCGM_FT_INT64,
                                             8,
                                             "minor_number",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MNNUM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_OEM_INFOROM_VER,
                                             DCGM_FT_STRING,
                                             0,
                                             "oem_inforom_version",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "OEMVR",
                                             "#",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_INFOROM_VER,
                                             DCGM_FT_STRING,
                                             0,
                                             "ecc_inforom_version",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "EIVER",
                                             "#",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_POWER_INFOROM_VER,
                                             DCGM_FT_STRING,
                                             0,
                                             "power_inforom_version",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PIVER",
                                             "#",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_INFOROM_IMAGE_VER,
                                             DCGM_FT_STRING,
                                             0,
                                             "inforom_image_version",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "IIVER",
                                             "#",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_INFOROM_CONFIG_CHECK,
                                             DCGM_FT_INT64,
                                             8,
                                             "inforom_config_checksum",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "CCSUM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCI_BUSID,
                                             DCGM_FT_STRING,
                                             0,
                                             "pci_busid",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PCBID",
                                             "#",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCI_COMBINED_ID,
                                             DCGM_FT_INT64,
                                             8,
                                             "pci_combined_id",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PCCID",
                                             "#",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCI_SUBSYS_ID,
                                             DCGM_FT_INT64,
                                             8,
                                             "pci_subsys_id",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PCSID",
                                             "#",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCIE_TX_THROUGHPUT,
                                             DCGM_FT_INT64,
                                             8,
                                             "pcie_tx_throughput",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "TXTPT",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_KBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCIE_RX_THROUGHPUT,
                                             DCGM_FT_INT64,
                                             8,
                                             "pcie_rx_throughput",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "RXTPT",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_KBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
                                             DCGM_FT_INT64,
                                             8,
                                             "pcie_replay_counter",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "RPCTR",
                                             "#",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SM_CLOCK,
                                             DCGM_FT_INT64,
                                             8,
                                             "sm_clock",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "SMCLK",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MEM_CLOCK,
                                             DCGM_FT_INT64,
                                             8,
                                             "memory_clock",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MMCLK",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VIDEO_CLOCK,
                                             DCGM_FT_INT64,
                                             8,
                                             "video_clock",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VICLK",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_APP_SM_CLOCK,
                                             DCGM_FT_INT64,
                                             8,
                                             "sm_app_clock",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "SACLK",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_APP_MEM_CLOCK,
                                             DCGM_FT_INT64,
                                             8,
                                             "mem_app_clock",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MACLK",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CLOCKS_EVENT_REASONS,
                                             DCGM_FT_INT64,
                                             0,
                                             "current_clocks_event_reasons",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "DVCCTR",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MAX_SM_CLOCK,
                                             DCGM_FT_INT64,
                                             8,
                                             "sm_max_clock",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "SMMAX",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MAX_MEM_CLOCK,
                                             DCGM_FT_INT64,
                                             8,
                                             "memory_max_clock",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MMMAX",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MAX_VIDEO_CLOCK,
                                             DCGM_FT_INT64,
                                             8,
                                             "video_max_clock",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VIMAX",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_AUTOBOOST,
                                             DCGM_FT_INT64,
                                             8,
                                             "autoboost",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "ATBST",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_GPU_TEMP,
                                             DCGM_FT_INT64,
                                             8,
                                             "gpu_temp",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "TMPTR",
                                             (char *)getTextForEnum(DCGM_FIELD_UNIT_TEMP_C),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MEM_MAX_OP_TEMP,
                                             DCGM_FT_INT64,
                                             8,
                                             "gpu_mem_max_op_temp",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "GMMOT",
                                             getTextForEnum(DCGM_FIELD_UNIT_TEMP_C),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_GPU_MAX_OP_TEMP,
                                             DCGM_FT_INT64,
                                             8,
                                             "gpu_max_op_temp",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "GGMOT",
                                             getTextForEnum(DCGM_FIELD_UNIT_TEMP_C),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_GPU_TEMP_LIMIT,
                                             DCGM_FT_INT64,
                                             8,
                                             "gpu_temp_tlimit",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "GTLIMIT",
                                             getTextForEnum(DCGM_FIELD_UNIT_TEMP_C),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SLOWDOWN_TEMP,
                                             DCGM_FT_INT64,
                                             8,
                                             "slowdown_temp",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "SDTMP",
                                             getTextForEnum(DCGM_FIELD_UNIT_TEMP_C),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SHUTDOWN_TEMP,
                                             DCGM_FT_INT64,
                                             8,
                                             "shutdown_temp",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "SHTMP",
                                             getTextForEnum(DCGM_FIELD_UNIT_TEMP_C),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_bandwidth_tx",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWTX",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_bandwidth_rx",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWRX",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_PHYS_ID,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_physical_id",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWPHID",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_RESET_REQUIRED,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_reset_required",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWFRMVER",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_ID,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_id",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "LNKID",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_STATUS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_nvlink_status",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWNVLNKST",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_TYPE,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_nvlink_dev_type",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWNVLNKDT",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_BUS,
                                             DCGM_FT_INT64,
                                             8,
                                             "link_pcie_remote_bus",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "LNKBUS",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DEVICE,
                                             DCGM_FT_INT64,
                                             8,
                                             "link_pcie_remote_dev",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "LNKDEV",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN,
                                             DCGM_FT_INT64,
                                             8,
                                             "link_pcie_remote_dom",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "LNKDOM",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_FUNCTION,
                                             DCGM_FT_INT64,
                                             8,
                                             "link_pcie_remote_func",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "LNKFNC",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_DEVICE_LINK_ID,
                                             DCGM_FT_INT64,
                                             8,
                                             "link_dev_link_id",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWNVLNKID",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_DEVICE_LINK_SID,
                                             DCGM_FT_INT64,
                                             8,
                                             "link_dev_link_sid",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWNVLNSID",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_DEVICE_UUID,
                                             DCGM_FT_STRING,
                                             8,
                                             "link_dev_uuid",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWNVDVUID",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_40));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L0,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link0",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB0",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L1,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link1",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB1",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L2,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link2",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB2",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L3,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link3",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB3",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L4,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link4",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB4",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L5,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link5",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB5",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L6,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link6",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB6",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L7,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link7",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB7",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L8,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link8",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB8",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L9,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link9",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB9",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L10,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link10",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB10",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L11,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link11",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB11",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L12,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link12",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB12",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L13,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link13",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB13",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L14,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link14",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB14",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L15,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link15",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB15",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L16,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link16",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB16",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L17,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_link17",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLRXB17",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_PCIE_BUS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_pcie_bus",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWPCIEBUS",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_PCIE_DEVICE,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_pcie_dev",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWPCIEDEV",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_pcie_dom",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWPCIEDOM",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_pcie_fun",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWPCIEFUN",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_POWER_MGMT_LIMIT,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "power_management_limit",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PMLMT",
                                             getTextForEnum(DCGM_FIELD_UNIT_POW_W),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "power_management_limit_min",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PMMIN",
                                             getTextForEnum(DCGM_FIELD_UNIT_POW_W),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "power_management_limit_max",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PMMAX",
                                             getTextForEnum(DCGM_FIELD_UNIT_POW_W),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "power_management_limit_default",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PMDEF",
                                             getTextForEnum(DCGM_FIELD_UNIT_POW_W),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_POWER_USAGE,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "power_usage",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "POWER",
                                             getTextForEnum(DCGM_FIELD_UNIT_POW_W),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_POWER_USAGE_INSTANT,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "power_usage_instant",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "POWINST",
                                             getTextForEnum(DCGM_FIELD_UNIT_POW_W),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
                                             DCGM_FT_INT64,
                                             8,
                                             "total_energy_consumption",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
                                             "TOTEC",
                                             getTextForEnum(DCGM_FIELD_UNIT_MILLIJOULES),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ENFORCED_POWER_LIMIT,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "enforced_power_limit",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "EPLMT",
                                             getTextForEnum(DCGM_FIELD_UNIT_POW_W),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FABRIC_MANAGER_STATUS,
                                             DCGM_FT_INT64,
                                             8,
                                             "fabric_manager_status",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "FMSTA",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FABRIC_MANAGER_ERROR_CODE,
                                             DCGM_FT_INT64,
                                             8,
                                             "fabric_manager_failure_code",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "FMFRC",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FABRIC_CLUSTER_UUID,
                                             DCGM_FT_STRING,
                                             0,
                                             "fabric_cluster_uuid",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "FCUID",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FABRIC_CLIQUE_ID,
                                             DCGM_FT_INT64,
                                             8,
                                             "fabric_clique_id",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "FMCID",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PSTATE,
                                             DCGM_FT_INT64,
                                             8,
                                             "pstate",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PSTAT",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FAN_SPEED,
                                             DCGM_FT_INT64,
                                             8,
                                             "fan_speed",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "FANSP",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_COMPUTE_MODE,
                                             DCGM_FT_INT64,
                                             8,
                                             "compute_mode",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "CMMOD",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_REQUESTED_POWER_PROFILE_MASK,
                                             DCGM_FT_BINARY,
                                             0,
                                             "req_power_prof",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "RPPRM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VALID_POWER_PROFILE_MASK,
                                             DCGM_FT_BINARY,
                                             0,
                                             "val_power_prof",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VPPRM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ENFORCED_POWER_PROFILE_MASK,
                                             DCGM_FT_BINARY,
                                             0,
                                             "enf_power_prof",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "EPPRM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PERSISTENCE_MODE,
                                             DCGM_FT_INT64,
                                             8,
                                             "persistance_mode",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PMMOD",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MIG_MODE,
                                             DCGM_FT_INT64,
                                             8,
                                             "mig_mode",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MGMOD",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CC_MODE,
                                             DCGM_FT_INT64,
                                             8,
                                             "cc_mode",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "CCMOD",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_GPM_SUPPORT,
                                             DCGM_FT_INT64,
                                             8,
                                             "gpm_support",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "GPMSPT",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CUDA_VISIBLE_DEVICES_STR,
                                             DCGM_FT_STRING,
                                             0,
                                             "cuda_visible_devices",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "CUVID",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MIG_MAX_SLICES,
                                             DCGM_FT_INT64,
                                             8,
                                             "mig_max_slices",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MIGMS",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MIG_GI_INFO,
                                             DCGM_FT_BINARY,
                                             0,
                                             "mig_gi_info",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MIGGIINFO",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MIG_CI_INFO,
                                             DCGM_FT_BINARY,
                                             0,
                                             "mig_ci_info",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MIGCIINFO",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MIG_ATTRIBUTES,
                                             DCGM_FT_BINARY,
                                             0,
                                             "mig_attributes",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MIGATT",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_CURRENT,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_ECC_CURRENT,
                                             "ECCUR",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_PENDING,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_pending",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_ECC_PENDING,
                                             "ECPEN",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_volatile_total",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_ECC_SBE_VOL_TOTAL,
                                             "ESVTL",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_volatile_total",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_ECC_DBE_VOL_TOTAL,
                                             "EDVTL",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_AGG_TOTAL,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_aggregate_total",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_ECC_SBE_AGG_TOTAL,
                                             "ESATL",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_AGG_TOTAL,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_aggregate_total",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_ECC_DBE_AGG_TOTAL,
                                             "EDATL",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_VOL_L1,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_volatile_l1",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_SBE_VOL_L1*/
                                             "ESVL1",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_VOL_L1,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_volatile_l1",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_DBE_VOL_L1*/
                                             "EDVL1",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_VOL_L2,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_volatile_l2",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_SBE_VOL_L2*/
                                             "ESVL2",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_VOL_L2,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_volatile_l2",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_DBE_VOL_L2*/
                                             "EDVL2",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_VOL_DEV,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_volatile_device",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_SBE_VOL_DEV*/
                                             "ESVDV",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_VOL_DEV,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_volatile_device",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_DBE_VOL_DEV*/
                                             "EDVDV",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_VOL_REG,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_volatile_register",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_SBE_VOL_REG*/
                                             "ESVRG",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_VOL_REG,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_volatile_register",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_DBE_VOL_REG*/
                                             "EDVRG",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_VOL_TEX,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_volatile_texture",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_SBE_VOL_TEX*/
                                             "ESVTX",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_VOL_TEX,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_volatile_texture",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_DBE_VOL_TEX*/
                                             "EDVTX",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_AGG_L1,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_aggregate_l1",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_SBE_AGG_L1*/
                                             "ESAL1",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_AGG_L1,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_aggregate_l1",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_DBE_AGG_L1*/
                                             "EDAL1",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_AGG_L2,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_aggregate_l2",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_SBE_AGG_L2*/
                                             "ESAL2",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_AGG_L2,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_aggregate_l2",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_DBE_AGG_L2*/
                                             "EDAL2",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_AGG_DEV,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_aggregate_device",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_SBE_AGG_DEV*/
                                             "ESADV",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_AGG_DEV,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_aggregate_device",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_DBE_AGG_DEV*/
                                             "EDADV",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_AGG_REG,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_aggregate_register",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_SBE_AGG_REG*/
                                             "ESARG",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_AGG_REG,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_aggregate_register",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_DBE_AGG_REG*/
                                             "EDARG",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_AGG_TEX,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_aggregate_texture",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_SBE_AGG_TEX*/
                                             "ESATX",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_AGG_TEX,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_aggregate_texture",
                                             DCGM_FS_DEVICE,
                                             0, /*NVML_FI_DEV_ECC_DBE_AGG_TEX*/
                                             "EDATX",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_VOL_SHM,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_volatile_shared",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "ESVSHM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_VOL_SHM,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_volatile_shared",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "EDVSHM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_VOL_CBU,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_volatile_cbu",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "ESVCBU",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_VOL_CBU,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_volatile_cbu",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "EDVCBU",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_VOL_SRM,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_volatile_sram",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "ESVSRM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_VOL_SRM,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_volatile_sram",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "EDVSRM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_AGG_SHM,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_aggregate_shared",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "ESDSHM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_AGG_SHM,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_aggregate_shared",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "EDDSHM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_AGG_CBU,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_aggregate_cbu",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "ESDCBU",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_AGG_CBU,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_aggregate_cbu",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "EDDCBU",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_AGG_SRM,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_sbe_aggregate_sram",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "ESDSRM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_AGG_SRM,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_dbe_aggregate_sram",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "EDDSRM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_THRESHOLD_SRM,
                                             DCGM_FT_INT64,
                                             8,
                                             "ecc_threshold_sram",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "ECCTHRSRM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_GPU_UTIL,
                                             DCGM_FT_INT64,
                                             8,
                                             "gpu_utilization",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "GPUTL",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MEM_COPY_UTIL,
                                             DCGM_FT_INT64,
                                             8,
                                             "mem_copy_utilization",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MCUTL",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ENC_UTIL,
                                             DCGM_FT_INT64,
                                             8,
                                             "enc_utilization",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "ECUTL",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_DEC_UTIL,
                                             DCGM_FT_INT64,
                                             8,
                                             "dec_utilization",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "DCUTL",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VBIOS_VERSION,
                                             DCGM_FT_STRING,
                                             0,
                                             "vbios_version",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VBVER",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MEM_AFFINITY_0,
                                             DCGM_FT_INT64,
                                             8,
                                             "mem_affinity_0",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MAFF0",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MEM_AFFINITY_1,
                                             DCGM_FT_INT64,
                                             8,
                                             "mem_affinity_1",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MAFF1",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MEM_AFFINITY_2,
                                             DCGM_FT_INT64,
                                             8,
                                             "mem_affinity_2",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MAFF2",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MEM_AFFINITY_3,
                                             DCGM_FT_INT64,
                                             8,
                                             "mem_affinity_3",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MAFF3",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_BAR1_TOTAL,
                                             DCGM_FT_INT64,
                                             8,
                                             "bar1_total",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "B1TTL",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_BAR1_USED,
                                             DCGM_FT_INT64,
                                             8,
                                             "bar1_used",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "B1USE",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_BAR1_FREE,
                                             DCGM_FT_INT64,
                                             8,
                                             "bar1_free",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "B1FRE",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FB_TOTAL,
                                             DCGM_FT_INT64,
                                             8,
                                             "fb_total",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "FBTTL",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FB_FREE,
                                             DCGM_FT_INT64,
                                             8,
                                             "fb_free",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "FBFRE",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FB_USED,
                                             DCGM_FT_INT64,
                                             8,
                                             "fb_used",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "FBUSD",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FB_RESERVED,
                                             DCGM_FT_INT64,
                                             8,
                                             "fb_resv",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "FBRSV",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FB_USED_PERCENT,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "fb_USDP",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "FBUSP",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VIRTUAL_MODE,
                                             DCGM_FT_INT64,
                                             8,
                                             "virtualization_mode",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VMODE",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_INSTANCE_IDS,
                                             DCGM_FT_BINARY,
                                             0,
                                             "active_vgpu_instance_ids",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VGIID",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_UTILIZATIONS,
                                             DCGM_FT_BINARY,
                                             0,
                                             "vgpu_instance_utilizations",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VIUTL",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_PER_PROCESS_UTILIZATION,
                                             DCGM_FT_BINARY,
                                             0,
                                             "vgpu_instance_per_process_utilization",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VIPPU",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_VM_ID,
                                             DCGM_FT_STRING,
                                             0,
                                             "vgpu_instance_vm_id",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VVMID",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_VM_NAME,
                                             DCGM_FT_STRING,
                                             0,
                                             "vgpu_instance_vm_name",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VMNAM",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_TYPE,
                                             DCGM_FT_INT64,
                                             8,
                                             "vgpu_instance_type",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VITYP",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_UUID,
                                             DCGM_FT_STRING,
                                             0,
                                             "vgpu_instance_uuid",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VUUID",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_DRIVER_VERSION,
                                             DCGM_FT_STRING,
                                             0,
                                             "vgpu_instance_driver_version",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VDVER",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_MEMORY_USAGE,
                                             DCGM_FT_INT64,
                                             8,
                                             "vgpu_instance_memory_usage",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VMUSG",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_INSTANCE_LICENSE_STATE,
                                             DCGM_FT_STRING,
                                             0,
                                             "vgpu_instance_license_state",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VLCIST",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_40));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_LICENSE_STATUS,
                                             DCGM_FT_INT64,
                                             8,
                                             "vgpu_instance_license_status",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VLCST",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT,
                                             DCGM_FT_INT64,
                                             8,
                                             "vgpu_instance_frame_rate_limit",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VFLIM",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_PCI_ID,
                                             DCGM_FT_STRING,
                                             0,
                                             "vgpu_instance_pci_id",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VPCIID",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_ENC_STATS,
                                             DCGM_FT_BINARY,
                                             0,
                                             "vgpu_instance_enc_stats",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VSTAT",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO,
                                             DCGM_FT_BINARY,
                                             0,
                                             "vgpu_instance_enc_sessions_info",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VSINF",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_FBC_STATS,
                                             DCGM_FT_BINARY,
                                             0,
                                             "vgpu_instance_fbc_stats",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VFSTAT",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO,
                                             DCGM_FT_BINARY,
                                             0,
                                             "vgpu_instance_fbc_sessions_info",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VFINF",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_VM_GPU_INSTANCE_ID,
                                             DCGM_FT_INT64,
                                             8,
                                             "vgpu_instance_gpu_instance_id",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VGII",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SUPPORTED_TYPE_INFO,
                                             DCGM_FT_BINARY,
                                             0,
                                             "supported_type_info",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "SPINF",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SUPPORTED_VGPU_TYPE_IDS,
                                             DCGM_FT_BINARY,
                                             0,
                                             "vgpu_type_ids",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VTID",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_TYPE_INFO,
                                             DCGM_FT_BINARY,
                                             0,
                                             "vgpu_type_info",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VTPINF",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_TYPE_NAME,
                                             DCGM_FT_BINARY,
                                             0,
                                             "vgpu_type_name",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VTPNM",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_TYPE_CLASS,
                                             DCGM_FT_BINARY,
                                             0,
                                             "vgpu_type_class",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VTPCLS",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_TYPE_LICENSE,
                                             DCGM_FT_BINARY,
                                             0,
                                             "vgpu_type_license",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "VTPLC",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS,
                                             DCGM_FT_BINARY,
                                             0,
                                             "creatable_vgpu_type_ids",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "CGPID",
                                             "",
                                             DCGM_FE_VGPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ENC_STATS,
                                             DCGM_FT_BINARY,
                                             0,
                                             "enc_stats",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "ENSTA",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FBC_STATS,
                                             DCGM_FT_BINARY,
                                             0,
                                             "fbc_stats",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "FBCSTA",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FBC_SESSIONS_INFO,
                                             DCGM_FT_BINARY,
                                             0,
                                             "fbc_sessions_info",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "FBCINF",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ACCOUNTING_DATA,
                                             DCGM_FT_BINARY,
                                             0,
                                             "accounting_data",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "ACCDT",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_RETIRED_SBE,
                                             DCGM_FT_INT64,
                                             8,
                                             "retired_pages_sbe",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "RPSBE",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_RETIRED_DBE,
                                             DCGM_FT_INT64,
                                             8,
                                             "retired_pages_dbe",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_RETIRED_DBE,
                                             "RPDBE",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_GRAPHICS_PIDS,
                                             DCGM_FT_BINARY,
                                             0,
                                             "graphics_pids",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "GPIDS",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_COMPUTE_PIDS,
                                             DCGM_FT_BINARY,
                                             0,
                                             "compute_pids",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "CMPID",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SUPPORTED_CLOCKS,
                                             DCGM_FT_BINARY,
                                             0,
                                             "supported_clocks",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "SPCLK",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_SYNC_BOOST,
                                             DCGM_FT_BINARY,
                                             0,
                                             "sync_boost",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "SYBST",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_RETIRED_PENDING,
                                             DCGM_FT_INT64,
                                             8,
                                             "retired_pages_pending",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_RETIRED_PENDING,
                                             "RPPEN",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS,
                                             DCGM_FT_INT64,
                                             8,
                                             "uncorrectable_remapped_rows",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_REMAPPED_UNC,
                                             "URMPS",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_MAX,
                                             DCGM_FT_INT64,
                                             8,
                                             "remap_rows_avail_max",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "RRAM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_HIGH,
                                             DCGM_FT_INT64,
                                             8,
                                             "remap_rows_avail_high",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "RRAH",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_PARTIAL,
                                             DCGM_FT_INT64,
                                             8,
                                             "remap_rows_avail_partial",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "RRAP",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_LOW,
                                             DCGM_FT_INT64,
                                             8,
                                             "remap_rows_avail_low",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "RRAL",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_NONE,
                                             DCGM_FT_INT64,
                                             8,
                                             "remap_rows_avail_none",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "RRAN",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS,
                                             DCGM_FT_INT64,
                                             8,
                                             "correctable_remapped_rows",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_REMAPPED_COR,
                                             "CRMPS",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ROW_REMAP_FAILURE,
                                             DCGM_FT_INT64,
                                             8,
                                             "row_remap_failure",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_REMAPPED_FAILURE,
                                             "RRF",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ROW_REMAP_PENDING,
                                             DCGM_FT_INT64,
                                             8,
                                             "row_remap_pending",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_REMAPPED_PENDING,
                                             "RRP",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_INFOROM_CONFIG_VALID,
                                             DCGM_FT_INT64,
                                             8,
                                             "inforom_config_valid",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "ICVLD",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_XID_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "xid_errors",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "XIDER",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCIE_MAX_LINK_GEN,
                                             DCGM_FT_INT64,
                                             8,
                                             "pcie_max_link_gen",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PCIMG",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCIE_MAX_LINK_WIDTH,
                                             DCGM_FT_INT64,
                                             8,
                                             "pcie_max_link_width",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PCIMW",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCIE_LINK_GEN,
                                             DCGM_FT_INT64,
                                             8,
                                             "pcie_link_gen",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PCILG",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCIE_LINK_WIDTH,
                                             DCGM_FT_INT64,
                                             8,
                                             "pcie_link_width",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PCILW",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_POWER_VIOLATION,
                                             DCGM_FT_INT64,
                                             8,
                                             "power_violation",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_PERF_POLICY_POWER,
                                             "PVIOL",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_THERMAL_VIOLATION,
                                             DCGM_FT_INT64,
                                             8,
                                             "thermal_violation",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_PERF_POLICY_THERMAL,
                                             "TVIOL",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_GPU_TOPOLOGY_PCI,
                                             DCGM_FT_BINARY,
                                             0,
                                             "system_topology_pci",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "STVCI",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_GPU_TOPOLOGY_NVLINK,
                                             DCGM_FT_BINARY,
                                             0,
                                             "system_topology_nvlink",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "STNVL",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_GPU_TOPOLOGY_AFFINITY,
                                             DCGM_FT_BINARY,
                                             0,
                                             "system_affinity",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "SYSAF",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SYNC_BOOST_VIOLATION,
                                             DCGM_FT_INT64,
                                             8,
                                             "sync_boost_violation",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_PERF_POLICY_SYNC_BOOST,
                                             "SBVIO",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_BOARD_LIMIT_VIOLATION,
                                             DCGM_FT_INT64,
                                             8,
                                             "board_limit_violation",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_PERF_POLICY_BOARD_LIMIT,
                                             "BLVIO",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LOW_UTIL_VIOLATION,
                                             DCGM_FT_INT64,
                                             8,
                                             "low_util_violation",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_PERF_POLICY_LOW_UTILIZATION,
                                             "LUVIO",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_RELIABILITY_VIOLATION,
                                             DCGM_FT_INT64,
                                             8,
                                             "reliability_violation",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_PERF_POLICY_RELIABILITY,
                                             "RVIOL",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_TOTAL_APP_CLOCKS_VIOLATION,
                                             DCGM_FT_INT64,
                                             8,
                                             "app_clock_violation",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_PERF_POLICY_TOTAL_APP_CLOCKS,
                                             "TAPCV",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_TOTAL_BASE_CLOCKS_VIOLATION,
                                             DCGM_FT_INT64,
                                             8,
                                             "base_clock_violation",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_PERF_POLICY_TOTAL_BASE_CLOCKS,
                                             "TAPBC",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "mem_util_samples",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MUSAM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_GPU_UTIL_SAMPLES,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "gpu_util_samples",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "GUSAM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L0,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l0",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L0,
                                             "NFEL0",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L1,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l1",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L1,
                                             "NFEL1",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L2,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l2",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L2,
                                             "NFEL2",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L3,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l3",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L3,
                                             "NFEL3",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L4,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l4",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L4,
                                             "NFEL4",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L5,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l5",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L5,
                                             "NFEL5",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_total",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                             "NFELT",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L0,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l0",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L0,
                                             "NDEL0",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L1,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l1",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L1,
                                             "NDEL1",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L2,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l2",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L2,
                                             "NDEL2",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L3,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l3",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L3,
                                             "NDEL3",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L4,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l4",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L4,
                                             "NDEL4",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L5,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l5",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L5,
                                             "NDEL5",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_total",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL,
                                             "NDELT",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l0",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0,
                                             "NREL0",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l1",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1,
                                             "NREL1",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l2",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2,
                                             "NREL2",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l3",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3,
                                             "NREL3",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l4",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4,
                                             "NREL4",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l5",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5,
                                             "NREL5",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_total",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL,
                                             "NRELT",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l0",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0,
                                             "NRCL0",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l1",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1,
                                             "NRCL1",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l2",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2,
                                             "NRCL2",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l3",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3,
                                             "NRCL3",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l4",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4,
                                             "NRCL4",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l5",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5,
                                             "NRCL5",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_total",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL,
                                             "NRCLT",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L0,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l0",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL0",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L1,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l1",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL1",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L2,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l2",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL2",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L3,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l3",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL3",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L4,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l4",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL4",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L5,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l5",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL5",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_TOTAL,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_total",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NTXBWLT",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_TOTAL,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rx_bandwidth_total",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NRXBWLT",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_total",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWLT",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L6,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l6",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L6,
                                             "NFEL6",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L7,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l7",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L7,
                                             "NFEL7",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L8,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l8",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L8,
                                             "NFEL8",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L9,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l9",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L9,
                                             "NFEL9",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L10,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l10",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L10,
                                             "NFEL10",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L11,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l11",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L11,
                                             "NFEL11",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L12,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l12",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NFEL12",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L13,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l13",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NFEL13",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L14,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l14",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NFEL14",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L15,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l15",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NFEL15",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L16,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l16",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NFEL16",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L17,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_flit_crc_error_count_l17",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NFEL17",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L6,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l6",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L6,
                                             "NDEL6",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L7,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l7",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L7,
                                             "NDEL7",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L8,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l8",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L8,
                                             "NDEL8",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L9,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l9",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L9,
                                             "NDEL9",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L10,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l10",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L10,
                                             "NDEL10",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L11,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l11",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L11,
                                             "NDEL11",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L12,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l12",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NDEL12",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L13,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l13",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NDEL13",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L14,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l14",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NDEL14",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L15,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l15",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NDEL15",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L16,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l16",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NDEL16",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L17,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_data_crc_error_count_l17",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NDEL17",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l6",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6,
                                             "NREL6",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l7",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7,
                                             "NREL7",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l8",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8,
                                             "NREL8",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l9",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9,
                                             "NREL9",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l10",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10,
                                             "NREL10",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l11",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11,
                                             "NREL11",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L12,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l12",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NREL12",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L13,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l13",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NREL13",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L14,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l14",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NREL14",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L15,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l15",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NREL15",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L16,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l16",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NREL16",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L17,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_error_count_l17",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NREL17",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l6",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6,
                                             "NRCL6",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l7",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7,
                                             "NRCL7",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l8",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8,
                                             "NRCL8",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l9",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9,
                                             "NRCL9",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l10",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10,
                                             "NRCL10",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l11",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11,
                                             "NRCL11",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L12,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l12",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NRCL12",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L13,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l13",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NRCL13",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L14,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l14",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NRCL14",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L15,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l15",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NRCL15",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L16,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l16",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NRCL16",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L17,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_error_count_l17",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NRCL17",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L6,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l6",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL6",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L7,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l7",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL7",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L8,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l8",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL8",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L9,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l9",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL9",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L10,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l10",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL10",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L11,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l11",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL11",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L12,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l12",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL12",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L13,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l13",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL13",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L14,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l14",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL14",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L15,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l15",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL15",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L16,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l16",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL16",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_BANDWIDTH_L17,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_l17",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NBWL17",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_ERROR_DL_CRC,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_crc_err",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_ERROR_DL_CRC,
                                             "NLCRC",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_recovery_err",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_ERROR_DL_RECOVERY,
                                             "NLREC",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_replay_err",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_ERROR_DL_REPLAY,
                                             "NLREP",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MEMORY_TEMP,
                                             DCGM_FT_INT64,
                                             8,
                                             "memory_temp",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MMTMP",
                                             "C",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_GPU_NVLINK_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "gpu_nvlink_errors",
                                             DCGM_FE_GPU,
                                             0,
                                             "GNVERR",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_voltage_mvolt",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVOLT",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_current_iddq",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWCUR",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_REV,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_current_iddq_rev",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SCIDDQ",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_DVDD,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_current_iddq_dvdd",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SCDVDD",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_POWER_VDD,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_power_vdd",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWPOWV",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_POWER_DVDD,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_power_dvdd",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWPOWD",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_POWER_HVDD,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_power_hvdd",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWPOWH",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_bandwidth_tx",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLNKTX",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_bandwidth_rx",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLNKRX",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_fatal_errors",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLNKFE",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_non_fatal_errors",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLNKNF",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_RECOVERY_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_recovery_errors",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLNKRC",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_replay_errors",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLNKRP",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_FLIT_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_flit_errors",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLNKFL",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_crc_errors",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLNKCR",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_ecc_errors",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLNKEC",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_low_vc0",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLL0",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_low_vc1",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLL1",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_low_vc2",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLL2",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_low_vc",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLL3",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_medium_vc0",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLM0",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_medium_vc1",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLM1",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_medium_vc2",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLM2",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_medium_vc3",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLM3",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_high_vc0",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLH0",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_high_vc1",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLH1",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_high_vc2",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLH2",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_high_vc3",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLH3",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_panic_vc0",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLP0",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_panic_vc1",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLP1",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_panic_vc2",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLP2",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_panic_vc3",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLP3",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_count_vc0",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLC0",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_count_vc1",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLC1",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_count_vc2",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLC2",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_latency_count_vc3",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWVCLC3",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE0,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_crc_errors_lane0",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLACR0",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE1,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_crc_errors_lane1",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLACR1",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE2,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_crc_errors_lane2",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLACR2",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE3,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_crc_errors_lane3",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLACR3",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE4,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_crc_errors_lane4",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLACR4",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE5,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_crc_errors_lane5",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLACR5",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE6,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_crc_errors_lane6",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLACR6",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE7,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_crc_errors_lane7",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLACR7",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE0,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_ecc_errors_lane0",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLAEC0",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE1,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_ecc_errors_lane1",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLAEC1",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE2,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_ecc_errors_lane2",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLAEC2",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE3,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_ecc_errors_lane3",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLAEC3",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE4,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_ecc_errors_lane4",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLAEC4",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE5,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_ecc_errors_lane5",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLAEC5",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE6,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_ecc_errors_lane6",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLAEC6",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE7,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_link_ecc_errors_lane7",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SWLAEC7",
                                             "",
                                             DCGM_FE_LINK,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L0,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link0",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB0",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L1,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link1",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB1",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L2,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link2",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB2",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L3,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link3",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB3",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L4,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link4",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB4",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L5,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link5",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB5",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L6,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link6",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB6",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L7,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link7",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB7",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L8,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link8",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB8",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L9,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link9",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB9",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L10,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link10",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB10",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L11,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link11",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB11",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L12,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link12",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB12",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L13,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link13",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB13",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L14,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link14",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB14",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L15,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link15",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB15",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L16,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link16",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB16",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L17,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_tx_bandwidth_link17",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NVLTXB17",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_fatal_error",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SEN00",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_non_fatal_error",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SEN01",
                                             "",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_TEMPERATURE_CURRENT,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_current_temperature",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "TMP01",
                                             "C",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SLOWDOWN,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_slowdown_temperature",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "TMP02",
                                             "C",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SHUTDOWN,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvswitch_shutdown_temperature",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "TMP03",
                                             "C",
                                             DCGM_FE_SWITCH,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CUDA_COMPUTE_CAPABILITY,
                                             DCGM_FT_INT64,
                                             0,
                                             "cuda_compute_capability",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "DVCCC",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_P2P_NVLINK_STATUS,
                                             DCGM_FT_INT64,
                                             8,
                                             "p2p_nvlink_status",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "P2PNS",
                                             "C",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_GR_ENGINE_ACTIVE,
                                             DCGM_FT_DOUBLE,
                                             0,
                                             "gr_engine_active",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "GRACT",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_SM_ACTIVE,
                                             DCGM_FT_DOUBLE,
                                             0,
                                             "sm_active",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "SMACT",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_SM_OCCUPANCY,
                                             DCGM_FT_DOUBLE,
                                             0,
                                             "sm_occupancy",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "SMOCC",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,
                                             DCGM_FT_DOUBLE,
                                             0,
                                             "tensor_active",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "TENSO",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_DRAM_ACTIVE,
                                             DCGM_FT_DOUBLE,
                                             0,
                                             "dram_active",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "DRAMA",
                                             "",
                                             DCGM_FE_GPU_I,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PIPE_FP64_ACTIVE,
                                             DCGM_FT_DOUBLE,
                                             0,
                                             "fp64_active",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "FP64A",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PIPE_FP32_ACTIVE,
                                             DCGM_FT_DOUBLE,
                                             0,
                                             "fp32_active",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "FP32A",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PIPE_FP16_ACTIVE,
                                             DCGM_FT_DOUBLE,
                                             0,
                                             "fp16_active",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "FP16A",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PCIE_TX_BYTES,
                                             DCGM_FT_INT64,
                                             0,
                                             "pcie_tx_bytes",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PCITX",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PCIE_RX_BYTES,
                                             DCGM_FT_INT64,
                                             0,
                                             "pcie_rx_bytes",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PCIRX",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_NVLINK_TX_BYTES,
                                             DCGM_FT_INT64,
                                             0,
                                             "nvlink_tx_bytes",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVLTX",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_NVLINK_RX_BYTES,
                                             DCGM_FT_INT64,
                                             0,
                                             "nvlink_rx_bytes",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVLRX",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE,
                                             DCGM_FT_DOUBLE,
                                             0,
                                             "tensor_imma_active",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "TIMMA",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE,
                                             DCGM_FT_DOUBLE,
                                             0,
                                             "tensor_hmma_active",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "THMMA",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PIPE_TENSOR_DFMA_ACTIVE,
                                             DCGM_FT_DOUBLE,
                                             0,
                                             "tensor_dfma_active",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "TDFMA",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PIPE_INT_ACTIVE,
                                             DCGM_FT_DOUBLE,
                                             0,
                                             "integer_active",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "INTAC",
                                             "",
                                             DCGM_FE_GPU_CI,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_UTIL_TOTAL,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "cpu_utilization_total",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CPUUT",
                                             "",
                                             DCGM_FE_CPU_CORE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_UTIL_USER,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "cpu_utilization_user",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CPUUU",
                                             "",
                                             DCGM_FE_CPU_CORE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_UTIL_NICE,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "cpu_utilization_nice",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CPUUN",
                                             "",
                                             DCGM_FE_CPU_CORE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_UTIL_SYS,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "cpu_utilization_sys",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CPUUS",
                                             "",
                                             DCGM_FE_CPU_CORE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_UTIL_IRQ,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "cpu_utilization_irq",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CPUUI",
                                             "",
                                             DCGM_FE_CPU_CORE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_TEMP_CURRENT,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "cpu_temp",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CPUTP",
                                             "",
                                             DCGM_FE_CPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_TEMP_WARNING,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "cpu_temp_warn",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CPUTW",
                                             "",
                                             DCGM_FE_CPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_TEMP_CRITICAL,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "cpu_temp_crit",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CPUTC",
                                             "",
                                             DCGM_FE_CPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_CLOCK_CURRENT,
                                             DCGM_FT_INT64,
                                             8,
                                             "cpu_clock",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CPUCL",
                                             "",
                                             DCGM_FE_CPU_CORE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_POWER_LIMIT,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "cpu_power_limit",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CPUPL",
                                             "",
                                             DCGM_FE_CPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_POWER_UTIL_CURRENT,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "cpu_power_utilization",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CPUPU",
                                             "",
                                             DCGM_FE_CPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MODULE_POWER_UTIL_CURRENT,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "module_power_utilization",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "MODPU",
                                             "",
                                             DCGM_FE_CPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SYSIO_POWER_UTIL_CURRENT,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "sysio_power_utilization",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "SIOPU",
                                             "",
                                             DCGM_FE_CPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_VENDOR,
                                             DCGM_FT_STRING,
                                             0,
                                             "cpu_vendor_name",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CPUVN",
                                             "",
                                             DCGM_FE_CPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_MODEL,
                                             DCGM_FT_STRING,
                                             0,
                                             "cpu_model_name",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CPUMN",
                                             "",
                                             DCGM_FE_CPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_C2C_LINK_COUNT,
                                             DCGM_FT_INT64,
                                             0,
                                             "c2c_link_count",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_C2C_LINK_COUNT,
                                             "C2CLC",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_C2C_LINK_STATUS,
                                             DCGM_FT_INT64,
                                             0,
                                             "c2c_link_status",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_C2C_LINK_GET_STATUS,
                                             "C2CST",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_C2C_MAX_BANDWIDTH,
                                             DCGM_FT_INT64,
                                             0,
                                             "c2c_max_bandwidth",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_C2C_LINK_GET_MAX_BW,
                                             "C2CMAXBW",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    for (unsigned int fieldId = DCGM_FI_PROF_NVDEC0_ACTIVE; fieldId <= DCGM_FI_PROF_NVDEC7_ACTIVE; fieldId++)
    {
        std::string tag      = fmt::format("nvdec{}_active", fieldId - DCGM_FI_PROF_NVDEC0_ACTIVE);
        std::string shortTag = fmt::format("NVDEC{}", fieldId - DCGM_FI_PROF_NVDEC0_ACTIVE);

        DcgmFieldsPopulateOneFieldWithFormatting(fieldId,
                                                 DCGM_FT_DOUBLE,
                                                 0,
                                                 tag.c_str(),
                                                 DCGM_FS_DEVICE,
                                                 0,
                                                 shortTag.c_str(),
                                                 "",
                                                 DCGM_FE_GPU,
                                                 getWidthForEnum(DCGM_FIELD_WIDTH_5));
    }

    for (unsigned int fieldId = DCGM_FI_PROF_NVJPG0_ACTIVE; fieldId <= DCGM_FI_PROF_NVJPG7_ACTIVE; fieldId++)
    {
        std::string tag      = fmt::format("nvjpg{}_active", fieldId - DCGM_FI_PROF_NVJPG0_ACTIVE);
        std::string shortTag = fmt::format("NVJPG{}", fieldId - DCGM_FI_PROF_NVJPG0_ACTIVE);

        DcgmFieldsPopulateOneFieldWithFormatting(fieldId,
                                                 DCGM_FT_DOUBLE,
                                                 0,
                                                 tag.c_str(),
                                                 DCGM_FS_DEVICE,
                                                 0,
                                                 shortTag.c_str(),
                                                 "",
                                                 DCGM_FE_GPU,
                                                 getWidthForEnum(DCGM_FIELD_WIDTH_5));
    }

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_NVOFA0_ACTIVE,
                                             DCGM_FT_DOUBLE,
                                             0,
                                             "nvofa0_active",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVOFA0",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_NVOFA1_ACTIVE,
                                             DCGM_FT_DOUBLE,
                                             0,
                                             "nvofa1_active",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVOFA1",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_DIAG_MEMORY_RESULT,
                                             DCGM_FT_INT64,
                                             sizeof(std::int64_t),
                                             "gpu_memory_test_result",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "MEMRES",
                                             "",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_DIAG_DIAGNOSTIC_RESULT,
                                             DCGM_FT_INT64,
                                             sizeof(std::int64_t),
                                             "diagnostics_test_result",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "DIARES",
                                             "",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_DIAG_PCIE_RESULT,
                                             DCGM_FT_INT64,
                                             sizeof(std::int64_t),
                                             "pcie_test_result",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "PCIRES",
                                             "",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_DIAG_TARGETED_STRESS_RESULT,
                                             DCGM_FT_INT64,
                                             sizeof(std::int64_t),
                                             "targeted_stress_test_result",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "STRRES",
                                             "",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_DIAG_TARGETED_POWER_RESULT,
                                             DCGM_FT_INT64,
                                             sizeof(std::int64_t),
                                             "targeted_power_test_result",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "POWRES",
                                             "",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_DIAG_MEMORY_BANDWIDTH_RESULT,
                                             DCGM_FT_INT64,
                                             sizeof(std::int64_t),
                                             "memory_bandwidth_test_result",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "MBWRES",
                                             "",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_DIAG_MEMTEST_RESULT,
                                             DCGM_FT_INT64,
                                             sizeof(std::int64_t),
                                             "memory_test_result",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "MEMRES",
                                             "",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_DIAG_PULSE_TEST_RESULT,
                                             DCGM_FT_INT64,
                                             sizeof(std::int64_t),
                                             "pulse_test_result",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "PLSRES",
                                             "",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_DIAG_EUD_RESULT,
                                             DCGM_FT_INT64,
                                             sizeof(std::int64_t),
                                             "eud_test_result",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "EUDRES",
                                             "",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_DIAG_CPU_EUD_RESULT,
                                             DCGM_FT_INT64,
                                             sizeof(std::int64_t),
                                             "cpu_eud_test_result",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "CPUEUDRES",
                                             "",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_DIAG_SOFTWARE_RESULT,
                                             DCGM_FT_INT64,
                                             sizeof(std::int64_t),
                                             "software_test_result",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "SWRES",
                                             "",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_DIAG_NVBANDWIDTH_RESULT,
                                             DCGM_FT_INT64,
                                             sizeof(std::int64_t),
                                             "nvbandwidth_test_result",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "NVBRES",
                                             "",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_TX_PACKETS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_xmit_packets",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_XMIT_PACKETS,
                                             "NLXMITP",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_TX_BYTES,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_xmit_bytes",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_XMIT_BYTES,
                                             "NLXMITB",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_RX_PACKETS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rcv_packets",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_RCV_PACKETS,
                                             "NLRCVP",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_RX_BYTES,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rcv_bytes",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_RCV_BYTES,
                                             "NLRCVB",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_RX_MALFORMED_PACKET_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_malformed_packets_err",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_MALFORMED_PACKET_ERRORS,
                                             "NLMALP",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_RX_BUFFER_OVERRUN_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_buffer_overrun_err",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_BUFFER_OVERRUN_ERRORS,
                                             "NLBUFO",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_RX_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rcv_err",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_RCV_ERRORS,
                                             "NLRCVE",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_RX_REMOTE_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rcv_remote_err",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_RCV_REMOTE_ERRORS,
                                             "NLRCVRE",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_RX_GENERAL_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rcv_generate_err",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_RCV_GENERAL_ERRORS,
                                             "NLRCVGE",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_LOCAL_LINK_INTEGRITY_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_rcv_local_link_integrity_err",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_LOCAL_LINK_INTEGRITY_ERRORS,
                                             "NLLLIE",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_TX_DISCARDS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_xmit_discards",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_XMIT_DISCARDS,
                                             "NLXMITD",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_SUCCESSFUL_EVENTS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_link_recovery_successful",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_SUCCESSFUL_EVENTS,
                                             "NLLRS",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_FAILED_EVENTS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_link_recovery_failed",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_FAILED_EVENTS,
                                             "NLLRF",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_EVENTS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_link_recovery",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_EVENTS,
                                             "NLLR",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_RX_SYMBOL_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_symbol_err",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_SYMBOL_ERRORS,
                                             "NLSYME",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_symbol_ber",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_NVLINK_COUNT_SYMBOL_BER,
                                             "NLSYMB",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER_FLOAT,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "nvlink_symbol_ber_float",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NLSYMBF",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_effective_ber",
                                             DCGM_FS_ENTITY,
                                             NVML_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER,
                                             "NLEFB",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER_FLOAT,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "nvlink_effective_ber_float",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "NLEFBF",
                                             "BER",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_ERRORS,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_effective_errors",
                                             DCGM_FS_ENTITY,
                                             NVML_FI_DEV_NVLINK_COUNT_EFFECTIVE_ERRORS,
                                             "NLEFE",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_DIAG_STATUS,
                                             DCGM_FT_BINARY,
                                             0,
                                             "diag_status",
                                             DCGM_FS_GLOBAL,
                                             0,
                                             "DIAGSTATUS",
                                             "",
                                             DCGM_FE_NONE,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CONNECTX_HEALTH,
                                             DCGM_FT_INT64,
                                             8,
                                             "cx_health",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CXHEALTH",
                                             "",
                                             DCGM_FE_CONNECTX,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CONNECTX_ACTIVE_PCIE_LINK_WIDTH,
                                             DCGM_FT_INT64,
                                             8,
                                             "cx_active_pcie_link_width",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CXAPLKWD",
                                             "",
                                             DCGM_FE_CONNECTX,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CONNECTX_ACTIVE_PCIE_LINK_SPEED,
                                             DCGM_FT_INT64,
                                             8,
                                             "cx_active_pcie_link_speed",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CXAPLKSP",
                                             "",
                                             DCGM_FE_CONNECTX,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CONNECTX_EXPECT_PCIE_LINK_WIDTH,
                                             DCGM_FT_INT64,
                                             8,
                                             "connectx_expect_pcie_link_width",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CXEPLKWD",
                                             "",
                                             DCGM_FE_CONNECTX,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CONNECTX_EXPECT_PCIE_LINK_SPEED,
                                             DCGM_FT_INT64,
                                             8,
                                             "cx_expect_pcie_link_speed",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CXEPLKSP",
                                             "",
                                             DCGM_FE_CONNECTX,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CONNECTX_CORRECTABLE_ERR_STATUS,
                                             DCGM_FT_INT64,
                                             8,
                                             "cx_correctable_err_status",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CXCERRST",
                                             "",
                                             DCGM_FE_CONNECTX,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CONNECTX_CORRECTABLE_ERR_MASK,
                                             DCGM_FT_INT64,
                                             8,
                                             "cx_correctable_err_mask",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CXCERRMK",
                                             "",
                                             DCGM_FE_CONNECTX,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERR_STATUS,
                                             DCGM_FT_INT64,
                                             8,
                                             "cx_uncorrectable_err_status",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CXUCERRST",
                                             "",
                                             DCGM_FE_CONNECTX,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERR_MASK,
                                             DCGM_FT_INT64,
                                             8,
                                             "cx_uncorrectable_err_mask",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CXUCERRMK",
                                             "",
                                             DCGM_FE_CONNECTX,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERR_SEVERITY,
                                             DCGM_FT_INT64,
                                             8,
                                             "cx_uncorrectable_err_severity",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CXUCERRSE",
                                             "",
                                             DCGM_FE_CONNECTX,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CONNECTX_DEVICE_TEMPERATURE,
                                             DCGM_FT_DOUBLE,
                                             8,
                                             "cx_dev_temp",
                                             DCGM_FS_ENTITY,
                                             0,
                                             "CXTEMP",
                                             "",
                                             DCGM_FE_CONNECTX,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));

    for (unsigned int fieldId = DCGM_FI_PROF_NVLINK_L0_TX_BYTES; fieldId <= DCGM_FI_PROF_NVLINK_L17_RX_BYTES; fieldId++)
    {
        unsigned int linkId = (fieldId - DCGM_FI_PROF_NVLINK_L0_TX_BYTES) / 2;
        bool isRx           = fieldId & 1; /* Odd fields are RX */

        std::string tag      = fmt::format("nvlink_l{}_{}_bytes", linkId, (isRx ? "rx" : "tx"));
        std::string shortTag = fmt::format("NVL{}{}", linkId, (isRx ? "R" : "T"));

        DcgmFieldsPopulateOneFieldWithFormatting(fieldId,
                                                 DCGM_FT_INT64,
                                                 0,
                                                 tag.c_str(),
                                                 DCGM_FS_DEVICE,
                                                 0,
                                                 shortTag.c_str(),
                                                 "",
                                                 DCGM_FE_GPU,
                                                 getWidthForEnum(DCGM_FIELD_WIDTH_20));
    }

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_C2C_TX_ALL_BYTES,
                                             DCGM_FT_INT64,
                                             0,
                                             "c2c_tx_all_bytes",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "C2CTXAB",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_C2C_TX_DATA_BYTES,
                                             DCGM_FT_INT64,
                                             0,
                                             "c2c_tx_data_bytes",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "C2CTXDB",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_C2C_RX_ALL_BYTES,
                                             DCGM_FT_INT64,
                                             0,
                                             "c2c_rx_all_bytes",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "C2CRXAB",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_C2C_RX_DATA_BYTES,
                                             DCGM_FT_INT64,
                                             0,
                                             "c2c_rx_data_bytes",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "C2CRXDB",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PLATFORM_INFINIBAND_GUID,
                                             DCGM_FT_STRING,
                                             0,
                                             "infiniband_guid",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "IBGUID",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_40));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PLATFORM_CHASSIS_SERIAL_NUMBER,
                                             DCGM_FT_STRING,
                                             0,
                                             "chassis_serial",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "CSERIAL",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_40));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PLATFORM_CHASSIS_SLOT_NUMBER,
                                             DCGM_FT_INT64,
                                             0,
                                             "chassis_slot_number",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "CSLOTNUM",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PLATFORM_TRAY_INDEX,
                                             DCGM_FT_INT64,
                                             0,
                                             "tray_index",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "CTRAYIDX",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PLATFORM_HOST_ID,
                                             DCGM_FT_INT64,
                                             0,
                                             "host_id",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "HOSTID",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PLATFORM_PEER_TYPE,
                                             DCGM_FT_INT64,
                                             0,
                                             "peer_type",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "PEERTYPE",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PLATFORM_MODULE_ID,
                                             DCGM_FT_INT64,
                                             0,
                                             "module_id",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "MODULEID",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_C2C_LINK_ERROR_INTR,
                                             DCGM_FT_INT64,
                                             8,
                                             "c2c_link_error_intr",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "C2CLNKINTR",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_C2C_LINK_ERROR_REPLAY,
                                             DCGM_FT_INT64,
                                             8,
                                             "c2c_link_error_replay",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "C2CLNKERRRPLY",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_C2C_LINK_ERROR_REPLAY_B2B,
                                             DCGM_FT_INT64,
                                             8,
                                             "c2c_link_error_replay_b2b",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "C2CLNKERRB2B",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_C2C_LINK_POWER_STATE,
                                             DCGM_FT_INT64,
                                             8,
                                             "c2c_link_power_state",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "C2CLNKPWRST",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_0,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count0",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST0",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_1,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count1",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST1",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_2,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count2",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST2",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_3,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count3",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST3",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_4,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count4",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST4",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_5,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count5",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST5",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_6,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count6",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST6",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_7,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count7",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST7",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_8,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count8",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST8",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_9,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count9",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST9",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_10,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count10",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST10",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_11,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count11",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST11",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_12,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count12",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST12",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_13,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count13",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST13",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_14,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count14",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST14",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_15,
                                             DCGM_FT_INT64,
                                             8,
                                             "nvlink_fec_history_count15",
                                             DCGM_FS_DEVICE,
                                             0,
                                             "NVFECHST15",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CLOCKS_EVENT_REASON_SW_POWER_CAP_NS,
                                             DCGM_FT_INT64,
                                             8,
                                             "clocks_event_power_cap_ns",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_POWER_CAP,
                                             "CLKPWRCAPNS",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CLOCKS_EVENT_REASON_SYNC_BOOST_NS,
                                             DCGM_FT_INT64,
                                             8,
                                             "clocks_event_boost_ns",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_CLOCKS_EVENT_REASON_SYNC_BOOST,
                                             "CLKSYNCBSTNS",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CLOCKS_EVENT_REASON_SW_THERM_SLOWDOWN_NS,
                                             DCGM_FT_INT64,
                                             8,
                                             "clocks_event_sw_thermal_slowdown_ns",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_THERM_SLOWDOWN,
                                             "CLKSWTHNS",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CLOCKS_EVENT_REASON_HW_THERM_SLOWDOWN_NS,
                                             DCGM_FT_INT64,
                                             8,
                                             "clocks_event_hw_thermal_slowdown_ns",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_THERM_SLOWDOWN,
                                             "CLKHWTHNS",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CLOCKS_EVENT_REASON_HW_POWER_BRAKE_SLOWDOWN_NS,
                                             DCGM_FT_INT64,
                                             8,
                                             "clocks_event_power_brake_slowdown_ns",
                                             DCGM_FS_DEVICE,
                                             NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_POWER_BRAKE_SLOWDOWN,
                                             "CLKPWRBRKNS",
                                             "",
                                             DCGM_FE_GPU,
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    return 0;
}

/*****************************************************************************/
/*
 * Hash function for measurementcollection keys (char *)
 *
 */
static unsigned int DcgmFieldsKeyHashCB(const void *key)
{
    char *val = (char *)key;
    int len   = (int)strlen(val);
    unsigned int retVal;

    MurmurHash3_x86_32(val, len, 0, &retVal);
    return retVal;
}

/*****************************************************************************/
/*
 * Key compare function
 *
 * Hashtable wants a non-zero return for equal keys, so invert strcmp()'s return
 */
static int DcgmFieldsKeyCmpCB(const void *key1, const void *key2)
{
    int st = strcmp((const char *)key1, (const char *)key2); /* Case sensitive, just like the hash function */

    return (!st); /* Inverted on purpose */
}

/*****************************************************************************/
/*
 * Key free function
 *
 */
static void DcgmFieldsKeyFreeCB(void *key)
{
    /* Since this is a malloc'd string, free it */
    free(key);
}

/*****************************************************************************/
/*
 * Value free function
 *
 */
static void DcgmFieldsValueFreeCB(void *value)
{
    dcgm_field_meta_t *fieldMeta = (dcgm_field_meta_t *)value;
    if (fieldMeta->valueFormat)
    {
        free(fieldMeta->valueFormat);
    }
}

/*****************************************************************************/
static int DcgmFieldsPopulateHashTable(void)
{
    int st, i;
    dcgm_field_meta_t *fieldMeta = nullptr;
    dcgm_field_meta_p found      = nullptr;

    st = hashtable_init(
        &dcgmFieldsKeyToIdMap, DcgmFieldsKeyHashCB, DcgmFieldsKeyCmpCB, DcgmFieldsKeyFreeCB, DcgmFieldsValueFreeCB);
    if (st)
        return -1;

    for (i = 0; i < DCGM_FI_MAX_FIELDS; i++)
    {
        fieldMeta = dcgmFieldMeta[i];
        if (!fieldMeta)
            continue;

        /* Make sure the field tag doesn't exist already */
        found = (dcgm_field_meta_p)hashtable_get(&dcgmFieldsKeyToIdMap, fieldMeta->tag);
        if (found != nullptr)
        {
            /* log_error("Found duplicate tag {} with id {} while inserting id {}", */
            /*             fieldMeta->tag, found->fieldId, fieldMeta->fieldId); */
            hashtable_close(&dcgmFieldsKeyToIdMap);
            return -1;
        }

        /* Doesn't exist. Insert ours, using tag as the key */
        st = hashtable_set(&dcgmFieldsKeyToIdMap, strdup(fieldMeta->tag), fieldMeta);
        if (st)
        {
            /* log_error( "Error {} while inserting tag {}", st, fieldMeta->tag); */
            hashtable_close(&dcgmFieldsKeyToIdMap);
            return -1;
        }
    }

    return 0;
}

/*****************************************************************************/
extern "C" DCGM_PUBLIC_API int DcgmFieldsInit(void)
{
    int st;

    if (dcgmFieldsInitialized)
        return 0;

    st = DcgmFieldsPopulateFieldTableWithFormatting();
    if (st)
        return -1;

    /* Create the hash table of tags to IDs */
    st = DcgmFieldsPopulateHashTable();
    if (st)
        return -1;

    dcgmFieldsInitialized = 1;
    return 0;
}

/*****************************************************************************/
extern "C" DCGM_PUBLIC_API int DcgmFieldsTerm(void)
{
    int i;

    if (!dcgmFieldsInitialized)
        return 0; /* Nothing to do */

    hashtable_close(&dcgmFieldsKeyToIdMap);

    /* Zero the structure just to be sure */
    memset(&dcgmFieldsKeyToIdMap, 0, sizeof(dcgmFieldsKeyToIdMap));

    for (i = 0; i < DCGM_FI_MAX_FIELDS; i++)
    {
        if (!dcgmFieldMeta[i])
            continue;

        free(dcgmFieldMeta[i]);
        dcgmFieldMeta[i] = 0;
    }

    dcgmFieldsInitialized = 0;
    return 0;
}

/*****************************************************************************/
extern "C" DCGM_PUBLIC_API dcgm_field_meta_p DcgmFieldGetById(unsigned short fieldId)
{
    if (!dcgmFieldsInitialized)
        return 0;
    if (fieldId >= DCGM_FI_MAX_FIELDS)
        return 0;

    return dcgmFieldMeta[fieldId];
}

/*****************************************************************************/
extern "C" DCGM_PUBLIC_API dcgm_field_meta_p DcgmFieldGetByTag(const char *tag)
{
    dcgm_field_meta_p retVal = 0;

    if (!dcgmFieldsInitialized || !tag)
        return 0;

    retVal = (dcgm_field_meta_p)hashtable_get(&dcgmFieldsKeyToIdMap, tag);
    return retVal;
}

/*****************************************************************************/
extern "C" DCGM_PUBLIC_API const char *DcgmFieldsGetEntityGroupString(dcgm_field_entity_group_t entityGroupId)
{
    switch (entityGroupId)
    {
        case DCGM_FE_GPU:
            return "GPU";
        case DCGM_FE_VGPU:
            return "vGPU";
        case DCGM_FE_SWITCH:
            return "Switch";
        case DCGM_FE_GPU_I:
            return "GPU_I";
        case DCGM_FE_GPU_CI:
            return "GPU_CI";
        case DCGM_FE_LINK:
            return "LINK";
        case DCGM_FE_CPU:
            return "CPU";
        case DCGM_FE_CPU_CORE:
            return "CPU_CORE";
        default:
        case DCGM_FE_NONE:
            return "None";
    }
}

/*****************************************************************************/
unsigned int DcgmFieldIdToNvmlGpmMetricId(unsigned int fieldId, bool &isPercentageField)
{
    static_assert(DCGM_FI_PROF_LAST_ID == DCGM_FI_PROF_C2C_RX_DATA_BYTES);

    isPercentageField = false;

    switch (fieldId)
    {
        case DCGM_FI_PROF_GR_ENGINE_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_GRAPHICS_UTIL;

        case DCGM_FI_PROF_SM_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_SM_UTIL;

        case DCGM_FI_PROF_SM_OCCUPANCY:
            isPercentageField = true;
            return NVML_GPM_METRIC_SM_OCCUPANCY;

        case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_ANY_TENSOR_UTIL;

        case DCGM_FI_PROF_DRAM_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_DRAM_BW_UTIL;

        case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_FP64_UTIL;

        case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_FP32_UTIL;

        case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_FP16_UTIL;

        case DCGM_FI_PROF_PCIE_TX_BYTES:
            return NVML_GPM_METRIC_PCIE_TX_PER_SEC;

        case DCGM_FI_PROF_PCIE_RX_BYTES:
            return NVML_GPM_METRIC_PCIE_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC;

        case DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_IMMA_TENSOR_UTIL;

        case DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_HMMA_TENSOR_UTIL;

        case DCGM_FI_PROF_PIPE_TENSOR_DFMA_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_DFMA_TENSOR_UTIL;

        case DCGM_FI_PROF_PIPE_INT_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_INTEGER_UTIL;

        case DCGM_FI_PROF_NVDEC0_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVDEC_0_UTIL;

        case DCGM_FI_PROF_NVDEC1_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVDEC_1_UTIL;

        case DCGM_FI_PROF_NVDEC2_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVDEC_2_UTIL;

        case DCGM_FI_PROF_NVDEC3_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVDEC_3_UTIL;

        case DCGM_FI_PROF_NVDEC4_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVDEC_4_UTIL;

        case DCGM_FI_PROF_NVDEC5_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVDEC_5_UTIL;

        case DCGM_FI_PROF_NVDEC6_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVDEC_6_UTIL;

        case DCGM_FI_PROF_NVDEC7_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVDEC_7_UTIL;

        case DCGM_FI_PROF_NVJPG0_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVJPG_0_UTIL;

        case DCGM_FI_PROF_NVJPG1_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVJPG_1_UTIL;

        case DCGM_FI_PROF_NVJPG2_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVJPG_2_UTIL;

        case DCGM_FI_PROF_NVJPG3_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVJPG_3_UTIL;

        case DCGM_FI_PROF_NVJPG4_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVJPG_4_UTIL;

        case DCGM_FI_PROF_NVJPG5_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVJPG_5_UTIL;

        case DCGM_FI_PROF_NVJPG6_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVJPG_6_UTIL;

        case DCGM_FI_PROF_NVJPG7_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVJPG_7_UTIL;

        case DCGM_FI_PROF_NVOFA0_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVOFA_0_UTIL;

        case DCGM_FI_PROF_NVOFA1_ACTIVE:
            isPercentageField = true;
            return NVML_GPM_METRIC_NVOFA_1_UTIL;

        case DCGM_FI_PROF_NVLINK_L0_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L0_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L0_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L0_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L1_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L1_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L1_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L1_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L2_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L2_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L2_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L2_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L3_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L3_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L3_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L4_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L4_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L5_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L4_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L4_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L5_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L5_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L5_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L5_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L6_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L6_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L6_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L6_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L7_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L7_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L7_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L7_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L8_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L8_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L8_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L8_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L9_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L9_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L9_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L9_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L10_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L10_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L10_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L10_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L11_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L11_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L11_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L11_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L12_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L12_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L12_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L12_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L13_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L13_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L13_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L13_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L14_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L14_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L14_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L14_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L15_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L15_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L15_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L15_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L16_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L16_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L16_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L16_RX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L17_TX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L17_TX_PER_SEC;

        case DCGM_FI_PROF_NVLINK_L17_RX_BYTES:
            return NVML_GPM_METRIC_NVLINK_L17_RX_PER_SEC;

        case DCGM_FI_PROF_C2C_TX_ALL_BYTES:
            return NVML_GPM_METRIC_C2C_TOTAL_TX_PER_SEC;

        case DCGM_FI_PROF_C2C_TX_DATA_BYTES:
            return NVML_GPM_METRIC_C2C_DATA_TX_PER_SEC;

        case DCGM_FI_PROF_C2C_RX_ALL_BYTES:
            return NVML_GPM_METRIC_C2C_TOTAL_RX_PER_SEC;

        case DCGM_FI_PROF_C2C_RX_DATA_BYTES:
            return NVML_GPM_METRIC_C2C_DATA_RX_PER_SEC;

        default:
            DCGM_LOG_ERROR << "Unknown profiling fieldId " << fieldId;
            return 0;
    }

    return 0;
}

/*****************************************************************************/
int DcgmFieldGetAllFieldIds(std::vector<unsigned short> &fieldIds)
{
    if (!dcgmFieldsInitialized)
    {
        DCGM_LOG_ERROR << "Called before fields were initialized";
        return -1;
    }

    fieldIds.clear();

    for (int i = 0; i < DCGM_FI_MAX_FIELDS; i++)
    {
        if (dcgmFieldMeta[i] != nullptr)
        {
            fieldIds.push_back(dcgmFieldMeta[i]->fieldId);
        }
    }

    return 0;
}

/*****************************************************************************/
