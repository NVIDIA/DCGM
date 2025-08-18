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

#pragma once

#include <boost/asio/io_context.hpp>

#include <memory>

class IoContext
{
public:
    IoContext();
    ~IoContext();
    IoContext(IoContext const &)            = delete;
    IoContext &operator=(IoContext const &) = delete;

    boost::asio::io_context &Get() const;

    // Boost::asio's initial global signal handling setup happens on demand during process launch
    // and is not thread safe. Grab this mutex during process creation.
    std::mutex &GetProcessCreationMutex();

    void Post(std::function<void()> func);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};