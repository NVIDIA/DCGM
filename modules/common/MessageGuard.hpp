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
#include "DcgmStringHelpers.h"
#include <cstdint>
#include <cxxabi.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>


namespace DcgmNs
{
namespace details
{
    /**
     * @brief Helper struct to provide ::type typename for a type.
     */
    template <typename T>
    struct Identity
    {
        using type = T;
    };

    /**
     * @brief Type trait to remove cv and pointer modifiers from a T type.
     * @tparam T The type to remove cv and pointer modifiers.
     */
    template <typename T>
    struct NakedType
        : std::conditional_t<
              std::is_pointer_v<T>,
              NakedType<std::remove_pointer_t<T>>,
              std::conditional_t<std::is_reference_v<T>,
                                 NakedType<std::remove_reference_t<T>>,
                                 std::conditional_t<std::disjunction_v<std::is_const<T>, std::is_volatile<T>>,
                                                    NakedType<std::remove_cv_t<T>>,
                                                    Identity<T>>>>
    {};

    template <typename T>
    using NakedType_t = typename NakedType<T>::type;

    namespace hidden
    {
        /**
         * @brief Helper for the DemangleType function. Reduces amount of statically allocated buffers.
         * @tparam TNaked Result of NakedType_t<T> - this should be a bare type without cv or pointer modifiers.
         * @return Demangled name or type_info::name if demangling failed.
         */
        template <typename TNaked>
        inline const char *DemangleNakedTypeOnly()
        {
            static char buffer[1024] = {};
            if (buffer[0] != '\0')
            {
                return buffer;
            }
            int status = -4;

            const char *symbolName = typeid(TNaked).name();
            char *demangled        = abi::__cxa_demangle(symbolName, nullptr, nullptr, &status);
            if (status != 0)
            {
                return symbolName;
            }
            SafeCopyTo(buffer, demangled);
            free(demangled);
            return buffer;
        }
    } // namespace hidden
    /**
     * @brief Demangle C++ type name removing cv-modifiers and pointers
     * @tparam T Type that which name should be demangled
     * @tparam TBare Naked type without pointers and cv-modifiers. Auto-deduced.
     * @return Demangled name or type_info::name if demangling failed.
     * @note This function demangles name just once for each type.
     */
    template <typename T, typename TNaked = NakedType_t<T>>
    inline const char *DemangleType()
    {
        return hidden::DemangleNakedTypeOnly<TNaked>();
    }
} // namespace details

/**
 * @brief Represents a void pointer with constness deducted from the template argument type constness.
 * @tparam T A type which is used to deduce whether the pointer is const or not.
 */
template <typename T>
struct VoidPtr
{
    using type = std::conditional_t<std::is_const_v<T>, void const *, void *>;
};
template <typename T>
using VoidPtr_t = typename VoidPtr<T>::type;

/**
 * @brief Represents a pointer which cannot have nullptr value.
 * @tparam TPtr The type of the pointer. The final pointer will have type TPtr*
 * @throws \c\b std::runtime_error if the source pointer is null.
 * @throws \c\b std::runtime_error if the alignment of the source pointer does not match the alignment of the TPtr type.
 * @note Other than nullptr check, this wrapper is not type safe as the constructor effectively accepts void* pointer
 *       and casts it to the TPtr* later.
 */
template <typename TPtr>
class NotNull
{
public:
    constexpr NotNull(VoidPtr_t<TPtr> pointer) /*implicit*/ // NOLINT
    {
        using namespace std::string_literals;

        if (pointer == nullptr)
        {
            throw std::runtime_error("Got null pointer for type "s + details::DemangleType<TPtr>());
        }

        if ((std::uintptr_t)pointer % alignof(TPtr) != 0)
        {
            throw std::runtime_error("Improper value alignment. Probably a wrong type "s
                                     + details::DemangleType<TPtr>());
        }

        m_ptr = (TPtr *)pointer;
    }

    constexpr NotNull(NotNull const &)     = default;
    constexpr NotNull(NotNull &&) noexcept = default;

    constexpr NotNull &operator=(NotNull const &) noexcept = default;
    constexpr NotNull &operator=(NotNull &&) noexcept      = default;

    constexpr explicit operator TPtr *() const noexcept
    {
        return m_ptr;
    }

    constexpr TPtr *get() const noexcept
    {
        return m_ptr;
    }

    constexpr TPtr *operator->() const noexcept
    {
        return m_ptr;
    }

private:
    TPtr *m_ptr = nullptr;
};

/**
 * @brief Exception that is thrown from the MessageGuard constructor if a message version differs from the expected one.
 */
class VersionMismatchException : public std::runtime_error
{
public:
    explicit VersionMismatchException(std::string const &what)
        : std::runtime_error(what)
    {}
};

/**
 * @brief Smart pointer wrapper for core messages that ensures the pointer to the message is not null and the version of
 *        the message matches the expected version of the type.
 * @tparam TMessage The message type. If you need a constant message, add \c const to this type.
 * @tparam TVersion The expected version of the message.
 * @throws \c\b VersionMismatchException if the ptr->header.version does not match the expected one.
 * @note The \c TMessage type should have a \c header member of a type that, in its turn, has \c version member.
 *       So this statement should be valid: <b><tt>(TMessage const*)ptr->header.version</tt></b>
 */
template <typename TMessage, size_t TVersion>
class MessageGuard
{
public:
    constexpr MessageGuard(VoidPtr_t<TMessage> ptr) /*implicit*/ // NOLINT
        : m_ptr(ptr)
    {
        using namespace std::string_literals;
        if (m_ptr->header.version != TVersion)
        {
            throw VersionMismatchException("Incompatible versions for type "s + details::DemangleType<TMessage>());
        }
    }

    constexpr TMessage *operator->() const
    {
        return m_ptr.get();
    }

    constexpr TMessage *get() const
    {
        return m_ptr.get();
    }

private:
    NotNull<TMessage> const m_ptr;
};
} // namespace DcgmNs
