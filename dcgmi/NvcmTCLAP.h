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
#ifndef _NVCMI_TCLAP_DEFS_H
#define _NVCMI_TCLAP_DEFS_H

#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <tclap/Arg.h>
#include <tclap/CmdLine.h>
#include <tclap/CmdLineInterface.h>
#include <tclap/CmdLineOutput.h>
#include <tclap/XorHandler.h>
#include <unordered_set>

/* Extensions to the TCLAP library for very specific processing and display */

/* This extension is meant to handle the case where the subsystem name
 * should not be parsed but still needs to be shown in the usage statements.
 * We do this by taking the 2nd arg (which will always be the subsystem name)
 * and combining it with the first argument so that TCLAP treats the whole thing
 * as the program name and displays it as such.
 */
class DCGMSubsystemCmdLine : public TCLAP::CmdLine
{
public:
    DCGMSubsystemCmdLine(std::string name,
                         const std::string &message,
                         const char delimiter       = ' ',
                         const std::string &version = "none",
                         bool helpAndVersion        = false)
        : CmdLine(message, delimiter, version, helpAndVersion)
    {
        myName = name;
        if (!helpAndVersion)
        {
            TCLAP::Visitor *v = new TCLAP::HelpVisitor(this, &_output);
            TCLAP::SwitchArg *help
                = new TCLAP::SwitchArg("h", "help", "Displays usage information and exits.", false, v);
            CmdLine::add(help);
            CmdLine::deleteOnExit(help);
            CmdLine::deleteOnExit(v);
        }
    }

    void parse(int argc, const char *const *argv)
    {
        std::vector<std::string> args;
        for (int i = 0; i < argc; i++)
        {
            if (i == 1)
            {
                args[0] = args[0] + " " + myName;
            }
            else
                args.push_back(argv[i]);
        }
        CmdLine::parse(args);
    }

private:
    std::string myName;
};


class DCGMOutput : public TCLAP::StdOutput
{
public:
    DCGMOutput()
        : m_hiddenArgs()
    {
        /*
         * These arguments will be hidden from the help output. (They all come from the diag subcommand.)
         * WARNING: do not name options in for other subcommands similarly or they will be hidden as well.
         */
        m_hiddenArgs.insert("train");
        m_hiddenArgs.insert("force");
        m_hiddenArgs.insert("training-iterations");
        m_hiddenArgs.insert("training-variance");
        m_hiddenArgs.insert("training-tolerance");
        m_hiddenArgs.insert("golden-values-filename");
    }

    void addToGroup(std::string groupName, TCLAP::Arg *arg)
    {
        groups[groupName].push_back(arg);
    }
    void addDescription(std::string description)
    {
        this->description = description;
    }
    void addFooter(std::string footer)
    {
        this->footer = footer;
    }


    virtual void usage(TCLAP::CmdLineInterface &_cmd)
    {
        std::cout << std::endl;
        DCGMOutput::spacePrint(std::cout, description, 80, 1, 0);
        std::cout << std::endl;

        std::cout << "Usage: " << _cmd.getProgramName() << std::endl;
        _shortUsageGroup(_cmd, std::cout);
        std::cout << "Flags:" << std::endl;
        _longUsage(_cmd, std::cout);

        std::cout << std::endl;
    }

private:
    std::unordered_set<std::string> m_hiddenArgs;

    void _shortUsageGroup(TCLAP::CmdLineInterface &_cmd, std::ostream &os)
    {
        std::map<std::string, std::list<TCLAP::Arg *>>::iterator groupsIt;
        std::string progName = _cmd.getProgramName();

        for (groupsIt = groups.begin(); groupsIt != groups.end(); groupsIt++)
        {
            std::string s                     = progName + " ";
            std::list<TCLAP::Arg *> groupList = groupsIt->second;
            for (std::list<TCLAP::Arg *>::iterator it = groupList.begin(); it != groupList.end(); it++)
            {
                if (ArgIsHidden((*it)->getName()))
                {
                    continue;
                }

                s = s + (*it)->shortID() + " ";
            }
            spacePrint(os, s, 75, 3, 5);
        }
        os << std::endl;
    }

    bool ArgIsHidden(const std::string &name) const
    {
        return (m_hiddenArgs.find(name) != m_hiddenArgs.end());
    }

    virtual void _longUsage(TCLAP::CmdLineInterface &_cmd, std::ostream &os) const
    {
        std::list<TCLAP::Arg *> argList = _cmd.getArgList();
        std::string message             = _cmd.getMessage();
        TCLAP::XorHandler xorHandler    = _cmd.getXorHandler();
        std::stringstream ss;

        // First the host
        for (TCLAP::ArgListIterator it = argList.begin(); it != argList.end(); it++)
            if (!(*it)->getName().compare("host") || !(*it)->getName().compare("group"))
            {
                ss.str("");
                ss << std::left << std::setw(17) << std::setfill(' ')
                   << ((*it)->getFlag().empty() ? "  " : "-") + (*it)->getFlag() + "  --" + (*it)->getName();
                ss << std::left << std::setw(10) << _helperGetValueFromShortID((*it)->shortID());
                DCGMOutput::spacePrint(os, ss.str() + " " + (*it)->getDescription(), 80, 2, 29);
            }

        // Second is XOR
        std::vector<std::vector<TCLAP::Arg *>> xorList = xorHandler.getXorList();
        for (int i = 0; static_cast<unsigned int>(i) < xorList.size(); i++)
        {
            for (TCLAP::ArgVectorIterator it = xorList[i].begin(); it != xorList[i].end(); it++)
            {
                if ((*it)->getName().compare("group"))
                {
                    ss.str("");
                    ss << std::left << std::setw(17) << std::setfill(' ')
                       << ((*it)->getFlag().empty() ? "  " : "-") + (*it)->getFlag() + "  --" + (*it)->getName();
                    ss << std::left << std::setw(10) << _helperGetValueFromShortID((*it)->shortID());
                    DCGMOutput::spacePrint(os, ss.str() + " " + (*it)->getDescription(), 80, 2, 29);
                }
            }
        }

        // Then the rest
        for (TCLAP::ArgListIterator it = argList.begin(); it != argList.end(); it++)
        {
            if (ArgIsHidden((*it)->getName()))
            {
                continue;
            }

            if (!xorHandler.contains((*it)) && (*it)->getName().compare("host") && (*it)->getName().compare("group"))
            {
                ss.str("");
                ss << std::left << std::setw(17) << std::setfill(' ')
                   << ((*it)->getFlag().empty() ? "  " : "-") + (*it)->getFlag() + "  --" + (*it)->getName();
                ss << std::left << std::setw(10) << _helperGetValueFromShortID((*it)->shortID());
                DCGMOutput::spacePrint(os, ss.str() + " " + (*it)->getDescription(), 80, 2, 29);
            }
        }

        std::cout << std::endl;

        DCGMOutput::spacePrint(os, footer, 80, 1, 0);
        if (!footer.empty())
        {
            std::cout << std::endl;
        }
        DCGMOutput::spacePrint(os, message, 80, 1, 0);
    }

protected:
    std::string _helperGetValueFromShortID(const std::string shortId) const
    {
        int start                = 0;
        int end                  = 0;
        std::string returnstring = "";
        // find start
        for (int i = 0; static_cast<unsigned int>(i) < shortId.length(); i++)
        {
            if (shortId[i] == '<')
            {
                start = i;
                break;
            }
        }

        // find end
        for (int i = shortId.length(); i > 0; i--)
        {
            if (shortId[i] == '>')
            {
                end = i;
                break;
            }
        }

        if (start > 1)
        {
            returnstring = shortId.substr(start + 1, end - start - 1);
        }
        return returnstring;
    }

    void spacePrint(std::ostream &os, const std::string &s, int maxWidth, int indentSpaces, int secondLineOffset) const
    {
        std::string buffer = s;
        int len            = static_cast<int>(buffer.length());

        if ((len + indentSpaces > maxWidth) && maxWidth > 0)
        {
            int allowedLen = maxWidth - indentSpaces;
            int start      = 0;
            while (start < len)
            {
                // find the substring length
                // int stringLen = std::min<int>( len - start, allowedLen );
                // doing it this way to support a VisualC++ 2005 bug
                int stringLen = std::min<int>(len - start, allowedLen);

                // trim the length so it doesn't end in middle of a word
                if (stringLen == allowedLen)
                    while (stringLen >= 0 && buffer[stringLen + start] != ' ' && buffer[stringLen + start] != ','
                           && buffer[stringLen + start] != '|')
                        stringLen--;

                // ok, the word is longer than the line, so just split
                // wherever the line ends
                if (stringLen <= 0)
                    stringLen = allowedLen;

                // check for newlines
                for (int i = 0; i < stringLen; i++)
                    if (buffer[start + i] == '\n')
                    {
                        buffer[start + i] = ' ';
                        stringLen         = i + 1;
                    }

                // print the indent
                for (int i = 0; i < indentSpaces; i++)
                    os << " ";

                if (start == 0)
                {
                    // handle second line offsets
                    indentSpaces += secondLineOffset;

                    // adjust allowed len
                    allowedLen -= secondLineOffset;
                }

                os << buffer.substr(start, stringLen) << std::endl;

                // so we don't start a line with a space
                while (start < len && buffer[stringLen + start] == ' ')
                {
                    start++;
                }

                start += stringLen;
            }
        }
        else
        {
            for (int i = 0; i < indentSpaces; i++)
                os << " ";
            os << buffer << std::endl;
        }
    }

    virtual void failure(TCLAP::CmdLineInterface &_cmd, TCLAP::ArgException &e)
    {
        std::string progName = _cmd.getProgramName();

        std::cerr << "PARSE ERROR: " << e.argId() << std::endl
                  << "             " << e.error() << std::endl
                  << std::endl;

        std::cerr << "Usage: " << std::endl;

        _shortUsageGroup(_cmd, std::cerr);

        std::cerr << std::endl
                  << "For complete USAGE and HELP type: " << std::endl
                  << "   " << progName << " --help" << std::endl
                  << std::endl;

        throw TCLAP::ExitException(1);
    }

    std::string description;
    std::string footer;
    std::map<std::string, std::list<TCLAP::Arg *>> groups;
};

class DCGMEntryOutput : public DCGMOutput
{
    virtual void _longUsage(TCLAP::CmdLineInterface &_cmd, std::ostream &os) const
    {
        std::list<TCLAP::Arg *> argList = _cmd.getArgList();
        std::string message             = _cmd.getMessage();
        TCLAP::XorHandler xorHandler    = _cmd.getXorHandler();
        std::stringstream ss;

        // First the required
        for (TCLAP::ArgListIterator it = argList.begin(); it != argList.end(); it++)
            if (!xorHandler.contains((*it)) && (*it)->isRequired())
            {
                ss << std::left << std::setw(17) << std::setfill(' ')
                   << ((*it)->getFlag().empty() ? "  " : "-") + (*it)->getFlag() + "    " + (*it)->getName();
                DCGMOutput::spacePrint(os, ss.str() + " " + (*it)->getDescription(), 80, 2, 19);
            }

        // Second is XOR
        std::vector<std::vector<TCLAP::Arg *>> xorList = xorHandler.getXorList();
        for (int i = 0; static_cast<unsigned int>(i) < xorList.size(); i++)
        {
            for (TCLAP::ArgVectorIterator it = xorList[i].begin(); it != xorList[i].end(); it++)
            {
                ss.str("");
                ss << std::left << std::setw(17) << std::setfill(' ')
                   << ((*it)->getFlag().empty() ? "  " : "-") + (*it)->getFlag() + "  " + (*it)->getName();
                DCGMOutput::spacePrint(os, ss.str() + " " + (*it)->getDescription(), 80, 2, 19);
                // if ( it+1 != xorList[i].end() )
                // spacePrint(os, "-- OR --", 75, 9, 0);
            }
        }

        // Then the rest
        for (TCLAP::ArgListIterator it = argList.begin(); it != argList.end(); it++)
            if (!xorHandler.contains((*it)) && !(*it)->isRequired())
            {
                std::string prefix
                    = (!(*it)->getName().compare("help") || !(*it)->getName().compare("version")) ? "  --" : "    ";
                ss.str("");
                ss << std::left << std::setw(17) << std::setfill(' ')
                   << ((*it)->getFlag().empty() ? "  " : "-") + (*it)->getFlag() + prefix + (*it)->getName();
                DCGMOutput::spacePrint(os, ss.str() + " " + (*it)->getDescription(), 80, 2, 19);
            }

        std::cout << std::endl;
        DCGMOutput::spacePrint(os, footer, 80, 1, 0);
        DCGMOutput::spacePrint(os, message, 80, 1, 0);
    }
};

#endif // _NVCMI_TCLAP_DEFS_H
