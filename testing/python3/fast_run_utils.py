# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logger

# These are the utility classes to handle backend hostengine state under
# FastRun.
#
# They allow creation of wrapper classes with start and reset methods to be
# called before the first test under an amortized decorator sequence and
# after each test, respectively.

# This is the base class for classes with start, and/or reset methods, to be
# called at the start of all tests with a decorator sequence amortized across
# them (start), and after each test (reset). By default, these methods do
# nothing here, but are intended to be overridden in the derived class. It is
# possible that only one actually has to be implemented (typically reset).
#
# Such classes serve to manage resources that may be created by tests and
# left dangling at the end of the test, leaving behind state (for example, in
# the hostengine process) that would break further tests under the same
# amortized decorator sequence.


class StateManager:
    def __init__(self):
        pass

    # Call before any tests under a sequence of amortized decorators.
    def start(self, kwargs, itemKwargs):
        pass

    # Call after each test under a sequence of amortized decorators,
    def reset(self, kwargs, itemKwargs):
        pass


class Managers:
    items = {}

    # Create a tuple with method reference, and keyword dictionary specific
    # for the method. The method should be called with global keywords
    # passed to tests as well as the method-specific keyword list. We can't
    # combine the two because we can't add unexpected keywords to a keyword
    # argument list for the test.
    #
    #
    # The start method of all registered objects will be called before the first
    # test under a decorator amortized over multiple tests.
    #
    # The reset methods of all registered objects will be called after each test
    # under a decorator amortized over multiple tests.
    #

    class AmortizedMethod:
        def __init__(self, name, item, kwargs):
            self.item = item
            self.kwargs = kwargs
            self.refCount = 1

    # Add a state manager.
    @classmethod
    def Add(cls, name, item, kwargs):
        logger.debug(f"Adding {name} to fast_run_utils.Managers.")
        if name in cls.items:
            cls.items[name].refCount += 1
        else:
            cls.items[name] = cls.AmortizedMethod(name, item, kwargs)

    # Delete a state manager.
    @classmethod
    def Del(cls, name):
        if name in cls.items:
            logger.debug(f"Removing {name} from fast_run_utils.Managers.")
            cls.items[name].refCount -= 1

            if cls.items[name].refCount == 0:
                del cls.items[name]

    # Call start methods.
    @classmethod
    def Start(cls, kwargs):
        for name, item in cls.items.items():
            if item and item.item:
                logger.debug(f"fast_run_utils.Managers starting {name}.")
                item.item.start(kwargs, item.kwargs)

    # Call reset methods.
    @classmethod
    def Reset(cls, kwargs):
        for name, item in cls.items.items():
            if item and item.item:
                logger.debug(f"fast_run_utils.Managers resetting {name}.")
                item.item.reset(kwargs, item.kwargs)

    # This should be called if an amortized decorator throws an exception.
    # All decorators should cleanup after themselves in that case.
    @classmethod
    def Clear(cls):
        del cls.items
        cls.items = {}
