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
#ifndef DCGMI_OUTPUT_H_
#define DCGMI_OUTPUT_H_

#include "CommandOutputController.h"
#include <iostream>
#include <json/json.h>
#include <sstream>

/* *****************************************************************************
 * The classes declared in this file generate human- and machine-readable output
 * for DCGMI.
 *
 * To use DcgmiOutput, create an instance of a concrete DcgmiOutput
 * implementation, add values to it, then print instance.str()
 * *****************************************************************************
 */

enum dcgmiOutputOptions
{
    DCGMI_OUTPUT_OPTIONS_SEPARATE_SECTIONS = 0,
    DCGMI_OUTPUT_OPTIONS_COUNT
};

using dcgmiOutputOptions_t = enum dcgmiOutputOptions;

class DcgmiOutput;

/* Internal class. Do not instantiate directly.
 *
 * Box can contain any type of "boxed" value
 */
class DcgmiOutputBox
{
protected:
    // Abstract class
    DcgmiOutputBox();

public:
    // Converts boxed value to string
    virtual std::string str() const = 0;
    virtual ~DcgmiOutputBox();
    virtual DcgmiOutputBox *clone() const = 0;
};

/* Internal class. Do not instantiate directly.
 *
 * Concrete implementations of DcgmiOutputBox
 */
template <typename T>
class ConcreteDcgmiOutputBox : public DcgmiOutputBox
{
private:
    T value;

public:
    ConcreteDcgmiOutputBox(const T &t);
    virtual ~ConcreteDcgmiOutputBox();
    ConcreteDcgmiOutputBox *clone() const;
    std::string str() const;
};

/* Internal class. Do not instantiate directly.
 *
 * Note: This is only a data structure. It does not control boxing the output */
class DcgmiOutputBoxer
{
private:
    static CommandOutputController cmdView;
    std::unique_ptr<DcgmiOutputBox> pb;
    Json::Value &json;
    std::vector<std::unique_ptr<DcgmiOutputBox>> overflow;
    std::map<std::string, std::unique_ptr<DcgmiOutputBoxer>> children;
    std::vector<std::string> childrenOrder;

    template <typename T>
    void set(const T &x);

public:
    DcgmiOutputBoxer(Json::Value &);
    virtual ~DcgmiOutputBoxer();
    DcgmiOutputBoxer(const DcgmiOutputBoxer &other);
    DcgmiOutputBoxer &operator=(const DcgmiOutputBoxer &other);

    template <typename T>
    DcgmiOutputBoxer &operator=(const T &x);
    DcgmiOutputBoxer &operator[](const std::string &childName);
    template <typename T>
    void addOverflow(const T &x);
    template <typename T>
    void setOrAppend(const T &x);
    std::string str() const;

    const DcgmiOutputBox *getBox() const
    {
        return pb.get();
    };
    const std::vector<std::unique_ptr<DcgmiOutputBox>> &getOverflow() const
    {
        return overflow;
    };
    const std::map<std::string, std::unique_ptr<DcgmiOutputBoxer>> &getChildren() const
    {
        return children;
    };
    const std::vector<std::string> &getChildrenOrder() const
    {
        return childrenOrder;
    };
};

class DcgmiOutputFieldSelector
{
private:
    std::vector<std::string> selectorStrings;
    DcgmiOutputBoxer &run(DcgmiOutputBoxer &, std::vector<std::string>::const_iterator) const;

public:
    DcgmiOutputFieldSelector();
    DcgmiOutputFieldSelector &child(const std::string &childName);
    DcgmiOutputBoxer &run(DcgmiOutputBoxer &out) const;
};

class DcgmiOutputColumnClass
{
private:
    unsigned int width;
    std::string name;
    DcgmiOutputFieldSelector selector;

public:
    unsigned int getWidth() const;
    const std::string &getName() const;
    const DcgmiOutputFieldSelector &getSelector() const;
    DcgmiOutputColumnClass(const unsigned int width, const std::string &name, const DcgmiOutputFieldSelector &selector);
};

class DcgmiOutput
{
protected:
    std::map<std::string, std::unique_ptr<DcgmiOutputBoxer>> sections;
    std::vector<std::string> sectionOrder;
    std::vector<std::string> header;
    Json::Value json;
    bool options[DCGMI_OUTPUT_OPTIONS_COUNT] = {};

    DcgmiOutput();

public:
    DcgmiOutput(const DcgmiOutput &other);
    DcgmiOutput &operator=(const DcgmiOutput &other);
    virtual ~DcgmiOutput();
    DcgmiOutputBoxer &operator[](const std::string &sectionName);
    void addHeader(const std::string &headerStr);
    int setOption(const unsigned int option, const bool value);
    virtual void addColumn(const unsigned int width,
                           const std::string &columnName,
                           const DcgmiOutputFieldSelector &selector)
    {}
    virtual std::string str() = 0;
};

class DcgmiOutputTree : public DcgmiOutput
{
private:
    unsigned int fullWidth;
    unsigned int rightWidth;

    std::string headerStr(const std::string &line) const;
    std::string levelStr(int level, const std::string &label, const DcgmiOutputBoxer &node) const;
    std::string rowStr(int level, const std::string &prefix, const std::string &label, const std::string &value) const;

public:
    DcgmiOutputTree(unsigned int leftWidth, unsigned int rightWidth);
    virtual ~DcgmiOutputTree();
    std::string str();
};

class DcgmiOutputJson : public DcgmiOutput
{
public:
    DcgmiOutputJson();
    virtual ~DcgmiOutputJson();
    std::string str();
};

class DcgmiOutputColumns : public DcgmiOutput
{
private:
    unsigned int fullWidth;
    std::vector<DcgmiOutputColumnClass> columns;

    std::string columnLabelsStr() const;
    std::string headerStr(const std::string &line) const;
    std::string overflowStr(DcgmiOutputBoxer &boxer) const;
    std::string rowStr(const std::vector<std::string> &strs) const;
    std::string sectionStr(const std::string &sectionName, DcgmiOutputBoxer &boxer) const;

public:
    DcgmiOutputColumns();
    virtual ~DcgmiOutputColumns();
    void addColumn(const unsigned int width, const std::string &columnName, const DcgmiOutputFieldSelector &selector);
    std::string str();
};

/* ************************************************************************** */
/* ****************** This is the end of the declarations. ****************** */
/* ******************     (Template) definitions below     ****************** */
/* ************************************************************************** */

// Templates have to be visible to the compiler

template <typename T>
void deleteNotNull(T *&obj)
{
    if (NULL != obj)
    {
        delete obj;
        obj = NULL;
    }
}

/******* ConcreteDcgmiOutputBox *******/

template <typename T>
ConcreteDcgmiOutputBox<T>::ConcreteDcgmiOutputBox(const T &t)
    : value(t)
{}

template <typename T>
ConcreteDcgmiOutputBox<T>::~ConcreteDcgmiOutputBox()
{}

template <typename T>
ConcreteDcgmiOutputBox<T> *ConcreteDcgmiOutputBox<T>::clone() const
{
    return new ConcreteDcgmiOutputBox<T>(*this);
}

template <typename T>
std::string ConcreteDcgmiOutputBox<T>::str() const
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

/******* DcgmiOutputBoxer *******/

template <typename T>
void DcgmiOutputBoxer::addOverflow(const T &x)
{
    std::string str = cmdView.HelperDisplayValue(x);
    overflow.emplace_back(std::make_unique<ConcreteDcgmiOutputBox<std::string>>(str));
    json["overflow"].append(str);
}

template <typename T>
void DcgmiOutputBoxer::set(const T &x)
{
    std::string str = cmdView.HelperDisplayValue(x);
    pb              = std::make_unique<ConcreteDcgmiOutputBox<std::string>>(str);
    json["value"]   = str;
}

template <typename T>
void DcgmiOutputBoxer::setOrAppend(const T &x)
{
    if (!pb)
    {
        set(x);
    }
    else
    {
        addOverflow(x);
    }
}

template <typename T>
DcgmiOutputBoxer &DcgmiOutputBoxer::operator=(const T &x)
{
    set(x);
    return *this;
}

#endif /* DCGMI_OUTPUT_H_ */
