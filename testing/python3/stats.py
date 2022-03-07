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
import itertools
import math

def mean(series):
    '''Calculate the mean of an iterable'''
    length = 0
    total = 0.0
    for i in series:    # must iterate because generators have no len()
        total += i
        length += 1
        
    if length == 0:
        return 0
    
    return total / length

def correlation_coefficient(x, y):
    '''
    taken from: https://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line
    '''
    xBar = mean(x)
    yBar = mean(y)
    xyBar = mean(xi*yi for xi, yi in zip(x, y))
    
    xSquaredBar = mean(xi**2 for xi in x)
    ySquaredBar = mean(yi**2 for yi in y)
    
    return (xyBar - xBar*yBar) / (math.sqrt((xSquaredBar-xBar**2) * (ySquaredBar-yBar**2)))

def standard_deviation(x):
    '''
    taken from: https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation
    '''
    xBar = mean(x)
    N = len(x)
    sumTerm = sum((xi - xBar)**2 for xi in x)
    
    return math.sqrt((1./(N-1)) * sumTerm)
