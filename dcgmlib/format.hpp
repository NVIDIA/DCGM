#ifndef DCGM_FORMAT_H
#define DCGM_FORMAT_H
//#define FMT_HEADER_ONLY
//#include <fmt/core.h>
//#include <iostream>   // std::FILE

//   template <typename FormatContext>
//   FMT_CONSTEXPR auto format(const T& val, FormatContext& ctx) const
//       -> decltype(ctx.out());

// template<>
// struct fmt::formatter<dcgmReturn_enum>
//   : formatter<string_view> {
//   constexpr auto format(dcgmReturn_enum value, auto& format_context)
//   {
//     return formatter<string_view>::format((int)value, format_context);
//   }
// };
// template<>
// struct fmt::formatter<dcgm_field_entity_group_t>
//   : formatter<string_view> {
//   constexpr auto format(dcgm_field_entity_group_t value, auto& format_context)
//   {
//     return formatter<string_view>::format((int)value, format_context);
//   }
// };

inline int format_as(dcgmReturn_enum type) {
  return static_cast<int>(type);
}
inline int format_as(dcgm_field_entity_group_t type) {
  return static_cast<int>(type);
}

inline int format_as(dcgmGroupType_enum type) {
  return static_cast<int>(type);
}

inline int format_as(dcgmModuleId_t type) {
  return static_cast<int>(type);
}

inline int format_as(dcgmDiagnosticLevel_t type) {
  return static_cast<int>(type);
}

inline int format_as(dcgmPolicyValidation_enum type) {
  return static_cast<int>(type);
}

inline int format_as(dcgmHealthSystems_enum type) {
  return static_cast<int>(type);
}

inline int format_as(dcgmPolicyCondition_enum type) {
  return static_cast<int>(type);
}

inline int format_as(dcgmConfigType_enum type) {
  return static_cast<int>(type);
}

inline int format_as(dcgmOrder_enum type) {
  return static_cast<int>(type);
}











// //dcgmModuleId_t
// // A rudimentary dcgmReturn_enum formatter.
// template <typename Char> struct fmt::formatter<dcgmReturn_enum, Char> {
//  public:
//   FMT_CONSTEXPR auto parse(basic_format_parse_context<Char>& ctx)
//       -> decltype(ctx.begin()) {
//     auto begin = ctx.begin(), end = ctx.end();
//     return begin;
//   }

//   template <typename FormatContext>
//   auto format(dcgmReturn_enum wd, FormatContext& ctx) const -> decltype(ctx.out()) {
//     int w = static_cast<int>(wd);
//     return "TODO";// What goes here?
//   }

// };

// //std::ostream &operator<<(std::ostream& os, ) {
// ///  return os << static_cast<int>(c);
// //}
// template <is_enum T>
// struct fmt::formatter<T> : fmt::formatter<std::string_view>
// {
//     constexpr auto format(T value, auto& format_context)
//     {
//         return formatter<string_view>::format(magic_enum::enum_name(value), format_context);
//     }
// };

// namespace fmt
// {

// // Specialize fmt::formatter for dcgmReturn_enum
// template <>
// class formatter<dcgmReturn_enum>  {
//  public:
//   // Parse the format specification
//   //auto parse(format_parse_context ctx) -> decltype(ctx.begin()) {
//   //    // Use the base class method to parse the format specification
//   //  return 0;//formatter<int>::parse(ctx);
//   //}

//   // Format the value of dcgmReturn_enum
//   template <typename FormatContext>
//   auto format(dcgmReturn_enum value, FormatContext& ctx) -> decltype(ctx.out()) {
//     // Use the base class method to format the value as an int
//     return formatter<int>::format(static_cast<int>(value), ctx);
//   }
// };

// template <>
// class formatter<dcgmReturn_enum&>  {
//  public:
//   // Parse the format specification
//   auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
//     // Use the base class method to parse the format specification
//     return 0;//formatter<int>::parse(ctx);
//   }

//   // Format the value of dcgmReturn_enum
//   template <typename FormatContext>
//   auto format(dcgmReturn_enum &value, FormatContext& ctx) -> decltype(ctx.out()) {
//     // Use the base class method to format the value as an int
//     return formatter<int>::format(static_cast<int>(value), ctx);
//   }
// };

// };

#endif
