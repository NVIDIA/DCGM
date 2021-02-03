# Best Practices for Writing Code in DCGM

DCGM style is checked using a pre-commit hook that you can install using install_git_hooks.sh. This guide provides best practices that are not related to style or are not covered in the pre-commit hook.

## Changesets
Under normal circumstances, each change list (CL) should have the same topic and purpose. For example, adding another policy condition to the policy manager and fixing a bug in the SM stress test should be two distinct CLs, even if one of those changes is only a few lines of code. This makes it easier to coordinate work on our team and with the QA team; it is especially useful for regression testing.

Closely related changes can and should be made in the same CL. For example, if fixing a particular bug requires an update to both the client and the server, as well as updating the Python bindings and the test cases, there is no benefit in splitting those changes into 4 different CLs.

## Functions
Functions should be short and focused on a single goal, with a name that is descriptive of what the function is trying to do. As we write new functions, they should generally be functions where unit tests are possible and meaningful; this requires the functions to have reasonable cyclomatic complexity and easily defined inputs and outputs. These standards apply to class methods as well as standalone functions.

## Exceptions
Exceptions should not be used and will almost always result in your code review being rejected prior to check-in:
 - In most cases, errors are best dealt with closer to the source of the error, and not an arbitrary level of calls higher on the stack.
 - Adding an exception without adding a crash requires a level of testing and code inspection that is not sustainable.
 - Remembering to handle exceptions adequately as you update code is exceptionally - pun accepted - difficult.

Instead of exceptions, we should generally favor:
 - Using an Initialize() method for initialization code that could experience errors.
 - Communicating error conditions through return codes and the error log.

If there are exceptions thrown from third party applications, they should be caught at the source and converted into a DCGM-acceptable style of reporting errors.

## Casting
When it is necessary to cast, please follow these guidelines:
 - Use C++ style casting instead of C-style as C++ style casts are specific and explicit in their functionality.
 - Avoid using const_cast
 - When possible, use static_cast instead reinterpret_cast.

## Includes
When including files in other directories, use '<' and '>' instead of quotations:
```
#include <HeaderInOtherDir.h>
#include "HeaderInSameDir.h
```

## Namespaces
 - Avoid ```using namespace X``` in .cpp files to embed the whole namespace in the current one.
 - Namespaces may not be embedded in header files.
 - Embedding individual members from a namespace is allowed in .cpp files (for example, ```using std::vector;```) but is prohibited in header files.

## Constructors
 - Constructors that receive a single parameter should be explicitly defined unless there is a specific reason not to explicitly define it.
 - Constructors that receive a single boolean parameter must be explicitly defined.

## Classes
 - All variables should be private by default on new classes.
 - Class variables should be named using the prefix m_ to make it obvious that it's a member of the class.

## Variable Naming
 - Default variable naming is camel case: aVariableIAmUsing
 - Class member variables are prepended with m_: m_otherVar
 - Const variables should be prepended with c_ : c_constantVar
 - Global variables should be prepended with g_ : g_globalVar
 - Global constants should be prepended with c_ : c_globalConstant

