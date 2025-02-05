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

#include <boost/asio.hpp>

#include <DcgmLogging.h>

namespace DcgmNs::Common::Subprocess
{

class Pipe
{
public:
    Pipe(boost::asio::io_context &ioCtx);
    ~Pipe();

    boost::asio::readable_pipe &ReadEnd();
    boost::asio::writable_pipe &WriteEnd();

    boost::system::error_code CloseReadEnd();
    boost::system::error_code CloseWriteEnd();

private:
    template <typename T>
    static boost::system::error_code CloseEnd(T &end)
    {
        if (!end.is_open())
        {
            return {};
        }
        boost::system::error_code ec;
        end.close(ec);
        if (ec)
        {
            log_error("failed to close pipe, err: [{}][{}].", ec.value(), ec.message());
        }
        return ec;
    }

    boost::asio::readable_pipe m_readEnd;
    boost::asio::writable_pipe m_writeEnd;
};

} //namespace DcgmNs::Common::Subprocess