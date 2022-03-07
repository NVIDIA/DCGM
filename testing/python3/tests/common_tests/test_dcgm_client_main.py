# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import signal
from _test_helpers import maybemock

from common import dcgm_client_main as m

@maybemock.patch('builtins.exit')
def test_exit_handler(mock_exit):
    m.exit_handler(None, None)
    mock_exit.assert_called()

@maybemock.patch('signal.signal')
def test_initialize_signal_handlers(mock_signal):
    m.initialize_signal_handlers()
    assert mock_signal.mock_calls[0][1] == (signal.SIGINT, m.exit_handler)
    assert mock_signal.mock_calls[1][1] == (signal.SIGTERM, m.exit_handler)
