#!/bin/bash

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

#Note: the protocol used below must match the one specified in dcgm_wsgi_nginx.conf. proxy_pass = HTTP. uwsgi_pass = UWSGI

#Start using the UWSGI protocol
PYTHONPATH=/usr/local/dcgm/bindings  /usr/local/bin/uwsgi --enable-threads --socket :1980 --wsgi-file /usr/share/dcgm_wsgi/dcgm_wsgi.py --logger syslog:dcgm_wsgi --daemonize2

#Start using the HTTP protocol
#PYTHONPATH=/usr/local/dcgm/bindings  /usr/local/bin/uwsgi --enable-threads --http :1980 --wsgi-file /usr/share/dcgm_wsgi/dcgm_wsgi.py --logger syslog:dcgm_wsgi --daemonize2
