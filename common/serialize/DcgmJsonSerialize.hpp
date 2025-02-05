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

#include <DcgmLogging.h>

#include <json/json.h>
#include <optional>


/**
 * @brief A namespace that provides Serialization/Deserialization functionality for Json objects.
 *
 * To be able to use this functionality, the class that you want to serialize/deserialize must have associated
 * ParseJson and ToJson functions in the same namespace (or a global namespace) as the class itself.
 *
 * @code
 *  namespace Example::Nested::Namespace {
 *    class ExampleClass {int i;};
 *    std::optional<ExampleClass> ParseJson(Json::Value const &root, DcgmNs::JsonSerialize::To<ExampleClass>) {
 *      if (!root.isMember("i") || !root["i"].isInt()) {
 *          return std::nullopt;
 *      }
 *      return ExampleClass{root["i"].asInt()};
 *    }
 *    ::Json::Value ToJson(ExampleClass const &exampleClass) {
 *      ::Json::Value root;
 *      root["i"] = exampleClass.i;
 *      return root;
 *    }
 *  }
 *
 *  // Usage:
 *  auto exampleClass = DcgmNs::JsonSerialize::Deserialize<Example::Nested::Namespace::ExampleClass>(root);
 *  ::Json::Value root = DcgmNs::JsonSerialize::Serialize(exampleClass);
 *  auto exampleClass = DcgmNs::JsonSerialize::TryDeserialize<Example::Nested::Namespace::ExampleClass>(root);
 *  if (exampleClass.has_value()) {
 *      // Do something with exampleClass.value()
 *  }
 * @endcode
 */
namespace DcgmNs::JsonSerialize
{

/**
 * @brief A "Marker" type that is used to exploit ADL to find ParseJson function for a given T type.
 * @tparam T Type that has associated ParseJson function implemented.
 * @note This marker allows compiler to use Argument Dependent Lookup mechanism for the template argument that enables
 *       compiler to find the ParseJson function in the same namespace where the T type is defined.
 */
template <class T>
struct To
{};

/**
 * @brief Concept that validates if an object can be parsed from a JSON value
 * @tparam T Type should implement ParseJson(Json::Value, To<T>) method
 */
template <class T>
concept IsJsonDeserializable = requires(T, Json::Value const &root) {
    { ParseJson(root, To<T> {}) } -> std::same_as<std::optional<T>>;
};
template <class T>
concept IsJsonSerializable = requires(T object) {
    { ToJson(object) } -> std::same_as<Json::Value>;
};

template <IsJsonSerializable T>
auto Serialize(T const &value)
{
    return ToJson(value);
}

template <IsJsonDeserializable T>
auto TryDeserialize(Json::Value const &root)
{
    return ParseJson(root, To<T>());
}

/**
 * @brief Deserializes from a string JSON value into a given type.
 *
 * The \a jsonStr will be parsed into a Json::Value object using a predefined reader builder and then passed to the
 * ParseJson function.
 *
 * @tparam T Target type that should be deserialized into.
 * @param jsonStr A string that contains a JSON value.
 * @return std::optional<T>
 * @note This method called with std::string argument will be ambiguous with the one that accepts ::Json::Value
 * argument. This is intentional. You should call this method with explicit casting to std::string_view. Example:
 * @code
 *  std::string jsonStr = R"({"i": 1})";
 *  auto exampleClass = TryDeserialize<Example::Nested::Namespace::ExampleClass>(std::string_view {jsonStr});
 * @endcode
 * This is done due to undesired behavior of the ::Json::Value(String const&) constructor that accepts std::string
 * and produces a JSON value with a length-prefixed string. This is not what we want.
 */
template <IsJsonDeserializable T>
auto TryDeserialize(std::string_view jsonStr)
{
    Json::Value root;
    ::Json::CharReaderBuilder builder;
    ::Json::String errors;
    using ReturnType = decltype(ParseJson(root, To<T>()));

    builder["collectComments"]     = false;
    builder["allowComments"]       = true;
    builder["allowTrailingCommas"] = true;
    builder["allowSingleQuotes"]   = true;
    builder["failIfExtra"]         = true;
    builder["rejectDupKeys"]       = true;
    builder["allowSpecialFloats"]  = true;
    builder["skipBom"]             = false;
    if (builder.newCharReader()->parse(jsonStr.data(), jsonStr.data() + jsonStr.size(), &root, &errors))
    {
        return ParseJson(root, To<T>());
    }
    log_error("Failed to parse JSON:\n{}\n\nRAW JSON:\n{}", errors, jsonStr);
    return ReturnType { std::nullopt };
}

template <IsJsonDeserializable T>
auto Deserialize(Json::Value const &root) -> T
{
    return TryDeserialize<T>(root).value();
}

template <IsJsonDeserializable T>
auto Deserialize(std::string_view jsonStr) -> T
{
    Json::Value root;
    ::Json::CharReaderBuilder builder;
    ::Json::String errors;
    builder["collectComments"]     = false;
    builder["allowComments"]       = true;
    builder["allowTrailingCommas"] = true;
    builder["allowSingleQuotes"]   = true;
    builder["failIfExtra"]         = true;
    builder["rejectDupKeys"]       = true;
    builder["allowSpecialFloats"]  = true;
    builder["skipBom"]             = false;
    if (builder.newCharReader()->parse(jsonStr.data(), jsonStr.data() + jsonStr.size(), &root, &errors))
    {
        return ParseJson(root, To<T>()).value();
    }

    log_error("Failed to parse JSON: {}", errors);
    log_debug("JSON: {}", jsonStr);
    throw std::runtime_error { fmt::format("Failed to parse JSON: {}", errors) };
}

} // namespace DcgmNs::JsonSerialize
