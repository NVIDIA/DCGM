 /*
  * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * limitations under the License.
  */
 
#include <DcgmMutex.h>
 
#include <catch2/catch_all.hpp>
 
TEST_CASE("DcgmLockGuard::Unlock")
{
    DcgmMutex mutex(0);
    DcgmLockGuard lockGuard(&mutex);
    REQUIRE(mutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);
    lockGuard.Unlock();
    REQUIRE(mutex.Poll() == DCGM_MUTEX_ST_NOTLOCKED);
}
