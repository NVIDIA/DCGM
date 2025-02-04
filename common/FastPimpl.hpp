// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

namespace DcgmNs::Common
{
/**
 * This class allows to avoid heap allocation for classes with private implementations idiom usage.
 * @tparam Type         The type of the private implementation.
 * @tparam Size         The size of the \a Type
 * @tparam Alignment    The alignment of the \a Type
 *
 * Example:
 * \code
 * //class.hpp file:
 * class Interface {
 * public:
 *      Interface();
 *      void Method();
 * private:
 *      struct Impl;
 *      FastPimpl<Impl, 8, 16> pimpl;
 * };
 *
 * //class.cpp file:
 * struct Interface::Impl {
 *      Impl(intptr_t ptr) : m_ptr(ptr) {}
 *      void MethodImpl() {...}
 *      intptr_t ptr;
 * };
 *
 * Interface::Interface() : pimpl(this) {}
 * void Interface::Method() { pimpl->MethodImpl(); }
 * \endcode
 *
 * In the example above, the pimpl field will be allocated on stack, which means the `Interface` class size is the size
 * of `Impl` class.
 *
 * @note The Size and Alignment are hardcoded in the parent (Interface) header and will be changed once the Impl is
 *          changed. It's not a problem if a wrong values for the Size or Alignment are specified - there will be a
 *          compile time error with expected right values in the message.
 */
template <typename Type, std::size_t Size, std::size_t Alignment>
class FastPimpl
{
public:
    template <typename... Args>
    explicit FastPimpl(Args &&...args)
    {
        new (Ptr()) Type(std::forward<Args>(args)...);
    }

    ~FastPimpl() noexcept
    {
        static_assert(sizeof(Type) > 0, "Type is incomplete");
        validate<sizeof(Type), alignof(Type)>();
        Ptr()->~Type();
    }

    FastPimpl &operator=(FastPimpl &&other) noexcept
    {
        *Ptr() = std::move(*other);
        return *this;
    }

    Type *operator->() noexcept
    {
        return Ptr();
    }

    Type const *operator->() const noexcept
    {
        return Ptr();
    }

    Type &operator*() noexcept
    {
        return *Ptr();
    }
    Type const &operator*() const noexcept
    {
        return *Ptr();
    }

private:
    Type *Ptr()
    {
        return reinterpret_cast<Type *>(&data);
    }

    Type const *Ptr() const
    {
        return reinterpret_cast<Type const *>(&data);
    }

    template <std::size_t ActualSize, std::size_t ActualAlignment>
    static void validate() noexcept
    {
        static_assert(ActualSize == Size, "Size and sizeof(Type) mismatch");
        static_assert(ActualAlignment == Alignment, "Alignment and alignof(Type) mismatch");
    }

    std::aligned_storage_t<Size, Alignment> data;
};
} //namespace DcgmNs::Common
