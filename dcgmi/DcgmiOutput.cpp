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
#include "DcgmiOutput.h"
#include <dcgm_structs_internal.h>
#include <iomanip>
#include <iostream>

std::string makeHline(char c, const std::vector<unsigned int> &widths)
{
    std::ostringstream ss;
    ss << '+';
    for (std::vector<unsigned int>::const_iterator cur = widths.begin(); cur != widths.end(); cur++)
    {
        ss << std::string(*cur, c);
        ss << '+';
    }
    ss << '\n';
    return ss.str();
}

/******* DcgmiOutputBox *******/

DcgmiOutputBox::DcgmiOutputBox()
{}

DcgmiOutputBox::~DcgmiOutputBox()
{}

/******* DcgmiOutputBoxer *******/

CommandOutputController DcgmiOutputBoxer::cmdView;

DcgmiOutputBoxer::DcgmiOutputBoxer(Json::Value &json)
    : json(json)
{}

DcgmiOutputBoxer::DcgmiOutputBoxer(const DcgmiOutputBoxer &other)
    : json(other.json)
{
    if (other.pb)
    {
        pb.reset(other.pb->clone());
    }
    else
    {
        pb.reset();
    }

    childrenOrder = other.childrenOrder;

    for (auto const &cur : other.children)
    {
        children[cur.first] = std::make_unique<DcgmiOutputBoxer>(*cur.second);
    }

    for (auto const &cur : other.overflow)
    {
        overflow.emplace_back(cur->clone());
    }
}

DcgmiOutputBoxer::~DcgmiOutputBoxer() = default;

DcgmiOutputBoxer &DcgmiOutputBoxer::operator[](const std::string &childName)
{
    Json::Value &jsonChild = json["children"][childName];

    auto const cur = children.find(childName);
    if (cur == children.end())
    {
        // It does not exist. Allocate memory for it
        children[childName] = std::make_unique<DcgmiOutputBoxer>(jsonChild);
        // Also add it to order
        childrenOrder.push_back(childName);
    }
    return *children[childName];
}

std::string DcgmiOutputBoxer::str() const
{
    if (NULL != pb)
    {
        return pb->str();
    }
    else
    {
        return "";
    }
}
DcgmiOutputBoxer &DcgmiOutputBoxer::operator=(const DcgmiOutputBoxer &other)
{
    if (this != &other)
    {
        if (other.pb)
        {
            pb.reset(other.pb->clone());
        }
        else
        {
            pb.reset();
        }

        childrenOrder = other.childrenOrder;

        children.clear();
        overflow.clear();

        for (auto const &cur : other.children)
        {
            children[cur.first] = std::make_unique<DcgmiOutputBoxer>(*cur.second);
        }

        for (auto const &cur : other.overflow)
        {
            overflow.emplace_back(cur->clone());
        }
    }

    return *this;
}

/******* DcgmiOutputFieldSelector *******/

DcgmiOutputFieldSelector::DcgmiOutputFieldSelector()
{}

DcgmiOutputFieldSelector &DcgmiOutputFieldSelector::child(const std::string &childName)
{
    selectorStrings.push_back(childName);
    return *this;
}

DcgmiOutputBoxer &DcgmiOutputFieldSelector::run(DcgmiOutputBoxer &out) const
{
    DcgmiOutputBoxer *result = &out;
    std::vector<std::string>::const_iterator cur;

    for (cur = selectorStrings.begin(); cur != selectorStrings.end(); cur++)
    {
        // First dereference the boxer (*result)
        // Then get the sub-boxer with the name [*cur]
        // Then get the address of that sub-boxer
        result = &(*result)[*cur];
    }
    return *result;
}

/******* DcgmiOutputColumnClass *******/

unsigned int DcgmiOutputColumnClass::getWidth() const
{
    return width;
}

const std::string &DcgmiOutputColumnClass::getName() const
{
    return name;
}

const DcgmiOutputFieldSelector &DcgmiOutputColumnClass::getSelector() const
{
    return selector;
}

DcgmiOutputColumnClass::DcgmiOutputColumnClass(const unsigned int width,
                                               const std::string &name,
                                               const DcgmiOutputFieldSelector &selector)
    : width(width)
    , name(name)
    , selector(selector)
{}

/******* DcgmiOutput *******/

DcgmiOutput::DcgmiOutput() = default;

DcgmiOutput::~DcgmiOutput() = default;

DcgmiOutput::DcgmiOutput(const DcgmiOutput &other)
{
    json         = other.json;
    header       = other.header;
    sectionOrder = other.sectionOrder;

    for (auto const &cur : other.sections)
    {
        sections[cur.first] = std::make_unique<DcgmiOutputBoxer>(*cur.second);
    }
}

DcgmiOutputBoxer &DcgmiOutput::operator[](const std::string &sectionName)
{
    Json::Value &jsonChild = json["body"][sectionName];

    auto cur = sections.find(sectionName);
    if (cur == sections.end())
    {
        // It does not exist. Allocate memory for it
        sections[sectionName] = std::make_unique<DcgmiOutputBoxer>(jsonChild);
        // Also add it to order
        sectionOrder.push_back(sectionName);
    }

    return *sections[sectionName];
}

void DcgmiOutput::addHeader(const std::string &headerStr)
{
    Json::Value &jsonHeader = json["header"];
    header.push_back(headerStr);
    jsonHeader.append(headerStr);
}

int DcgmiOutput::setOption(const unsigned int option, const bool value)
{
    if (option >= DCGMI_OUTPUT_OPTIONS_COUNT)
    {
        return DCGM_ST_BADPARAM;
    }

    options[option] = value;

    return 0;
}
DcgmiOutput &DcgmiOutput::operator=(const DcgmiOutput &other)
{
    if (this != &other)
    {
        json         = other.json;
        header       = other.header;
        sectionOrder = other.sectionOrder;
        sections.clear();
        for (auto const &cur : other.sections)
        {
            sections[cur.first] = std::make_unique<DcgmiOutputBoxer>(*cur.second);
        }
    }

    return *this;
}

/******* DcgmiOutputTree *******/

DcgmiOutputTree::DcgmiOutputTree(unsigned int leftWidth, unsigned int rightWidth)
    : fullWidth(leftWidth + rightWidth)
    , rightWidth(rightWidth)
{}

DcgmiOutputTree::~DcgmiOutputTree()
{}

std::string DcgmiOutputTree::str()
{
    std::string result;
    std::vector<std::string>::const_iterator cur;
    const unsigned int separatorPosition = fullWidth - rightWidth;
    bool isFirstSection                  = true;

    std::vector<unsigned int> widths;
    widths.push_back(separatorPosition - 1);
    widths.push_back(rightWidth - 2);
    const std::string hline = makeHline('-', widths);

    result += hline;

    if (header.size() > 0)
    {
        for (cur = header.begin(); cur != header.end(); cur++)
        {
            result += headerStr(*cur);
        }
        result += makeHline('=', widths);
    }

    for (cur = sectionOrder.begin(); cur != sectionOrder.end(); cur++)
    {
        if (!isFirstSection && options[DCGMI_OUTPUT_OPTIONS_SEPARATE_SECTIONS])
        {
            const std::string hline = makeHline('-', widths);
            result += hline;
        }

        isFirstSection = false;

        result += levelStr(0, *cur, *sections[*cur]);
    }

    // Footer
    result += hline;
    return result;
}

std::string DcgmiOutputTree::headerStr(const std::string &line) const
{
    std::ostringstream ss;
    ss << "| ";
    ss << std::left << std::setw(fullWidth - 3) << line;
    ss << "|\n";
    return ss.str();
}

std::string DcgmiOutputTree::levelStr(int level, const std::string &label, const DcgmiOutputBoxer &node) const
{
    std::string prefix;
    std::ostringstream ss;

    auto &&childrenOrder = node.getChildrenOrder();
    auto &&overflow      = node.getOverflow();
    auto &&children      = node.getChildren();

    // 0 is a special case
    if (level != 0)
    {
        prefix = "-> ";
    }
    else
    {
        prefix = "";
    }

    ss << rowStr(level, prefix, label, node.str());

    for (auto const &cur : overflow)
    {
        // No prefix or label for overflow
        ss << rowStr(level, "", "", cur->str());
    }

    for (auto const &cur : childrenOrder)
    {
        ss << levelStr(level + 1, cur, *(children.at(cur)));
    }

    return ss.str();
}

std::string DcgmiOutputTree::rowStr(int level,
                                    const std::string &prefix,
                                    const std::string &label,
                                    const std::string &value) const
{
    const unsigned int leftWidth     = fullWidth - rightWidth;
    const unsigned int leftTextArea  = leftWidth - 3;
    const unsigned int rightTextArea = rightWidth - 4;
    std::string left;
    std::string right;
    std::ostringstream ss;

    ss << std::setw(level * 3) << prefix;
    ss << label;
    left = ss.str();

    ss.str(std::string());
    ss.clear();
    ss << std::left << std::setw(rightTextArea) << value;
    right = ss.str();

    ss.str(std::string());
    ss.clear();
    ss << "| ";
    ss << std::left << std::setw(leftTextArea) << left;
    ss << " | ";
    ss << std::left << std::setw(rightTextArea) << right;
    ss << " |\n";

    return ss.str();
}

/******* DcgmiOutputColumns *******/

DcgmiOutputColumns::DcgmiOutputColumns()
    : fullWidth(0)
{}

DcgmiOutputColumns::~DcgmiOutputColumns()
{}

void DcgmiOutputColumns::addColumn(const unsigned int width,
                                   const std::string &columnName,
                                   const DcgmiOutputFieldSelector &selector)
{
    const DcgmiOutputColumnClass column(width, columnName, selector);
    fullWidth += 1 + width;
    columns.push_back(column);
}

std::string DcgmiOutputColumns::str()
{
    std::string result;
    std::vector<std::string>::iterator cur;
    std::vector<unsigned int> widths;
    bool isFirstSection = true;

    for (std::vector<DcgmiOutputColumnClass>::const_iterator cur = columns.begin(); cur != columns.end(); cur++)
    {
        widths.push_back(cur->getWidth());
    }

    const std::string hline = makeHline('-', widths);

    result += hline;

    if (header.size() > 0)
    {
        for (cur = header.begin(); cur != header.end(); cur++)
        {
            result += headerStr(*cur);
        }
        result += makeHline('=', widths);
    }

    result += columnLabelsStr();
    result += hline;

    for (cur = sectionOrder.begin(); cur != sectionOrder.end(); cur++)
    {
        if (!isFirstSection && options[DCGMI_OUTPUT_OPTIONS_SEPARATE_SECTIONS])
        {
            const std::string hline = makeHline('-', widths);
            result += hline;
        }

        isFirstSection = false;
        result += sectionStr(*cur, *sections[*cur]);
    }

    // Footer
    result += hline;
    return result;
}

std::string DcgmiOutputColumns::columnLabelsStr() const
{
    std::vector<std::string> strs;

    for (std::vector<DcgmiOutputColumnClass>::const_iterator cur = columns.begin(); cur != columns.end(); cur++)
    {
        strs.push_back(cur->getName());
    }

    return rowStr(strs);
}

std::string DcgmiOutputColumns::headerStr(const std::string &line) const
{
    std::ostringstream ss;
    ss << "| ";
    ss << std::left << std::setw(fullWidth - 2) << line;
    ss << "|\n";
    return ss.str();
}

std::string DcgmiOutputColumns::overflowStr(DcgmiOutputBoxer &boxer) const
{
    size_t maxsize = 0;
    std::string result;
    std::vector<std::string> strs;
    std::vector<const DcgmiOutputBoxer *> subboxes;
    std::vector<const DcgmiOutputBoxer *>::const_iterator boxcur;
    std::vector<DcgmiOutputColumnClass>::const_iterator columncur;
    DcgmiOutputBoxer *pboxer;

    for (columncur = columns.begin(); columncur != columns.end(); columncur++)
    {
        pboxer = &columncur->getSelector().run(boxer);
        subboxes.push_back(pboxer);
        maxsize = std::max(maxsize, pboxer->getOverflow().size());
    }

    for (size_t overflowLine = 0; overflowLine < maxsize; overflowLine++)
    {
        strs.clear();
        for (boxcur = subboxes.begin(); boxcur != subboxes.end(); boxcur++)
        {
            if (overflowLine < (*boxcur)->getOverflow().size())
            {
                strs.push_back((*boxcur)->getOverflow()[overflowLine]->str());
            }
            else
            {
                // If no overflow, push an empty string so we have something to
                // print for this field
                strs.push_back("");
            }
        }
        result += rowStr(strs);
    }

    return result;
}

std::string DcgmiOutputColumns::rowStr(const std::vector<std::string> &strs) const
{
    std::ostringstream ss;
    std::vector<DcgmiOutputColumnClass>::const_iterator columncur;
    std::vector<std::string>::const_iterator strcur;
    bool firstCol = true;

    for (strcur = strs.begin(), columncur = columns.begin(); columncur != columns.end(); columncur++, strcur++)
    {
        ss << (firstCol ? "| " : " | ") << std::setw(columncur->getWidth() - 2) << std::left << *strcur;
        firstCol = false;
    }

    ss << " |\n";

    return ss.str();
}

std::string DcgmiOutputColumns::sectionStr(const std::string &sectionName, DcgmiOutputBoxer &boxer) const
{
    std::string result;
    std::vector<std::string> strs;

    for (std::vector<DcgmiOutputColumnClass>::const_iterator cur = columns.begin(); cur != columns.end(); cur++)
    {
        strs.push_back(cur->getSelector().run(boxer).str());
    }

    result = rowStr(strs);
    result += overflowStr(boxer);

    return result;
}

/******* DcgmiOutputJson *******/

DcgmiOutputJson::DcgmiOutputJson()
{}

DcgmiOutputJson::~DcgmiOutputJson()
{}

std::string DcgmiOutputJson::str()
{
    return json.toStyledString();
}
