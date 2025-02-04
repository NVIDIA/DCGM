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

#include "Pipe.hpp"

namespace DcgmNs::Common::Subprocess
{

Pipe::Pipe(boost::asio::io_context &ioCtx)
    : m_readEnd(ioCtx)
    , m_writeEnd(ioCtx)
{
    boost::asio::connect_pipe(m_readEnd, m_writeEnd);
}

Pipe::~Pipe()
{
    CloseReadEnd();
    CloseWriteEnd();
}

boost::asio::readable_pipe &Pipe::ReadEnd()
{
    return m_readEnd;
}

boost::asio::writable_pipe &Pipe::WriteEnd()
{
    return m_writeEnd;
}

boost::system::error_code Pipe::CloseReadEnd()
{
    return CloseEnd(m_readEnd);
}

boost::system::error_code Pipe::CloseWriteEnd()
{
    return CloseEnd(m_writeEnd);
}

} //namespace DcgmNs::Common::Subprocess