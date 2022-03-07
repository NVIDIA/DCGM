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
#include "HostEngineOutput.h"

#include <tclap/ArgException.h>
#include <tclap/XorHandler.h>

#include <iostream>
#include <string_view>

namespace
{
/**
 * Repeat outputting a value to a stream given amount of times
 */
struct Repeat
{
    Repeat(int times, std::string_view value)
        : times(times)
        , value(value)
    {}

private:
    int times;
    std::string_view value;

    friend std::ostream &operator<<(std::ostream &os, Repeat const &obj);
};
inline std::ostream &operator<<(std::ostream &os, Repeat const &obj)
{
    if (obj.times > 0)
    {
        for (int i = 0; i < obj.times; ++i)
        {
            os << obj.value;
        }
    }
    return os;
}

/**
 * Check if a symbol can be used to word-wrap:
 *      Does not potentially belong to a word
 */
bool IsWordBorder(char const ch)
{
    return ch == ' ' || ch == ',' || ch == '|' || ch == '.' || ch == '\n';
}

/**
 * \brief Print a text line with potential word-wrapping.
 * This function assumes there are no new-line characters in the value already
 * \param os        [in]    Stream to write to
 * \param value     [in]    Text to output
 * \param indent    [in]    Indentation for the first line
 * \param slIndent  [in]    Additional indentation for second and following lines. This is not an absolute number.
 *                          For the second lines absolute indentation will be indent + slIndent
 * \param maxWidth  [in]    Max allowed text width. This value is used for indentation
 */
void PrintLine(std::ostream &os, std::string_view value, size_t indent, int slIndent, size_t maxWidth)
{
    int curIndent = static_cast<int>(indent);
    while (!value.empty())
    {
        int maxAllowedSubLen = static_cast<int>(maxWidth) - curIndent;
        if (maxAllowedSubLen <= 0 || value.length() <= static_cast<size_t>(maxAllowedSubLen))
        {
            os << Repeat(curIndent, " ") << value;
            return;
        }

        for (; maxAllowedSubLen >= 0 && !IsWordBorder(value[maxAllowedSubLen]); --maxAllowedSubLen)
        {
            ;
        }

        if (maxAllowedSubLen <= 0)
        {
            // We were not able to find proper word break position
            os << Repeat(curIndent, " ") << value;
            return;
        }

        bool printedDelimiter = false;
        if (value[maxAllowedSubLen] != '\n' && value[maxAllowedSubLen] != ' ')
        {
            ++maxAllowedSubLen;
            printedDelimiter = true;
        }

        os << Repeat(curIndent, " ") << value.substr(0, maxAllowedSubLen) << "\n";
        value.remove_prefix(maxAllowedSubLen + (printedDelimiter ? 0 : 1));
        curIndent += slIndent;
        slIndent = 0;
    }
}

/**
 * \brief Print a text with word-wrapping. The text may contain new-line symbols.
 * \param os            [in]    Stream to write to
 * \param value         [in]    Text to output
 * \param indent        [in]    Indentation for the first line
 * \param slIndent      [in]    Additional indentation for second and following lines. This is not an absolute number.
 *                              For the second lines absolute indentation will be indent + slIndent
 * \param maxWidth      [in]    Max allowed text width. This value is used for indentation
 * \param continuation  [in]    This argument controls if a line after a new-line symbol should be considered as a
 *                              'second line' or a 'first line'.
 *                              If TRUE, lines after a new-line symbol will have indent + slIndent indentation level.
 *                              If FALSE, indentation will be done at indent level.
 * \sa PrintLine()
 */
void Print(std::ostream &os,
           std::string_view value,
           size_t indent,
           int slIndent,
           size_t maxWidth,
           bool continuation = true)
{
    auto nlPos = std::string_view::npos;
    do
    {
        nlPos = value.find('\n');
        PrintLine(os, value.substr(0, nlPos), indent, slIndent, maxWidth);
        if (nlPos != std::string_view::npos)
        {
            os << "\n";
        }
        value.remove_prefix(nlPos + 1);

        if (continuation)
        {
            indent += slIndent;
            slIndent = 0;
        }
    } while (nlPos != std::string_view::npos);
}

/**
 * \struct WidthLimit
 * \brief A helper struct to call \sa Print() functionality using << operator
 */
struct WidthLimit
{
    WidthLimit(size_t maxWidth, size_t indent, int secondLineIndent, std::string_view text)
        : maxWidth(maxWidth)
        , indent(indent)
        , secondLineIndent(secondLineIndent)
        , text(text)
    {}

private:
    size_t maxWidth;
    size_t indent;
    int secondLineIndent;
    std::string_view text;

    friend std::ostream &operator<<(std::ostream &os, WidthLimit const &wl);
};
std::ostream &operator<<(std::ostream &os, WidthLimit const &wl)
{
    Print(os, wl.text, wl.indent, wl.secondLineIndent, wl.maxWidth);
    return os;
}


/**
 * \brief Helper macros to provide indices in range-based for-loops
 */
#define for_index(...) for_index_v(i, __VA_ARGS__)
#define for_index_v(i, ...)      \
    if (uint i##_next = 0; true) \
        for (__VA_ARGS__)        \
            if (uint i = i##_next++; true)

} // namespace

HostEngineOutput::HostEngineOutput(std::string prologue,
                                   std::string epilogue,
                                   std::string version,
                                   std::size_t maxWidth)
    : m_prologue(std::move(prologue))
    , m_epilogue(std::move(epilogue))
    , m_version(std::move(version))
    , m_maxWidth(maxWidth)
{}

void HostEngineOutput::usage(TCLAP::CmdLineInterface &c)
{
    std::cout << WidthLimit(m_maxWidth, 4, 0, m_prologue) << "\nUSAGE:\n";

    PrintShortUsage(c, std::cout);

    std::cout << "\nWhere:\n";

    PrintLongUsage(c, std::cout);

    std::cout << "\n" << WidthLimit(m_maxWidth, 4, 0, m_epilogue) << std::endl;
}

void HostEngineOutput::version([[maybe_unused]] TCLAP::CmdLineInterface &c)
{
    std::cout << WidthLimit(m_maxWidth, 0, 0, m_version) << std::endl;
}

void HostEngineOutput::failure(TCLAP::CmdLineInterface &c, TCLAP::ArgException &e)
{
    std::string progName = c.getProgramName();

    std::cerr << "PARSE ERROR: " << e.argId() << "\n"
              << "             " << e.error() << "\n\n";

    if (c.hasHelpAndVersion())
    {
        std::cerr << "Brief USAGE:\n";

        PrintShortUsage(c, std::cerr);

        std::cerr << "\n"
                  << "For complete USAGE and HELP type:\n"
                  << "   " << progName << " " << TCLAP::Arg::nameStartString() << "help" << std::endl
                  << std::endl;
    }
    else
    {
        usage(c);
    }

    throw TCLAP::ExitException(1);
}

void HostEngineOutput::PrintShortUsage(TCLAP::CmdLineInterface &cmdLine, std::ostream &os)
{
    auto const &argList  = cmdLine.getArgList();
    auto const &progName = cmdLine.getProgramName();
    auto &xorHandler     = cmdLine.getXorHandler();
    auto const &xorList  = xorHandler.getXorList();

    std::string s = progName + " ";

    // first the xor
    for (auto const &xorItem : xorList)
    {
        s += " {";
        for (auto const &arg : xorItem)
        {
            s += arg->shortID() + "|";
        }
        s[s.length() - 1] = '}';
    }

    // then the rest
    for (auto const &arg : argList)
    {
        if (!xorHandler.contains(arg))
        {
            s += " " + arg->shortID();
        }
    }

    // if the program name is too long, then adjust the second line offset
    auto secondLineOffset = std::min(progName.length() + 2, 75UL / 2);
    os << WidthLimit(m_maxWidth, 3, secondLineOffset, s);
}

void HostEngineOutput::PrintLongUsage(TCLAP::CmdLineInterface &cmdLine, std::ostream &os)
{
    auto const &argList = cmdLine.getArgList();
    auto &xorHandler    = cmdLine.getXorHandler();
    auto const &xorList = xorHandler.getXorList();

    [[maybe_unused]] auto makeLongId = [this](TCLAP::Arg const &arg) -> std::string {
        std::string result;
        result.reserve(m_maxWidth);
        result += '[';
        if (!arg.getFlag().empty())
        {
            result += TCLAP::Arg::flagStartString() + arg.getFlag() + " | ";
        }

        result += TCLAP::Arg::nameStartString() + arg.getName() + ']';

        return result;
    };

    // first the xor
    int maxNameLength = std::numeric_limits<int>::min();
    for (auto const &xorItem : xorList)
    {
        for (auto const &arg : xorItem)
        {
            maxNameLength = std::max<int>(maxNameLength, makeLongId(*arg).length());
        }
    }
    maxNameLength = std::min<int>(maxNameLength, m_maxWidth / 2);

    for (auto const &xorItem : xorList)
    {
        for_index(auto const &arg : xorItem)
        {
            auto longId = makeLongId(*arg);
            os << WidthLimit(m_maxWidth, 2, 0, longId) << ": ";
            os << WidthLimit(m_maxWidth,
                             std::max<int>(maxNameLength - longId.length(), 0),
                             longId.length() + 4,
                             arg->getDescription());
            if (i < xorItem.size()) // See for_index macro, which provides i index
            {
                os << WidthLimit(m_maxWidth, 9, 0, "-- OR --");
            }
        }
        os << "\n";
    }

    // then the rest
    maxNameLength = std::numeric_limits<int>::min();
    for (auto const &arg : argList)
    {
        maxNameLength = std::max<int>(maxNameLength, makeLongId(*arg).length());
    }
    maxNameLength = std::min<int>(maxNameLength, m_maxWidth / 2);
    for (auto const &arg : argList)
    {
        auto longId = makeLongId(*arg);
        os << WidthLimit(m_maxWidth, 2, 0, longId) << ": ";
        os << WidthLimit(
            m_maxWidth, std::max<int>(maxNameLength - longId.length(), 0), longId.length() + 4, arg->getDescription());
        os << "\n";
    }
    os << std::endl;
}