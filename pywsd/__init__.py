#!/usr/bin/env python -*- coding: utf-8 -*-
#
# Python Word Sense Disambiguation (pyWSD)
#
# Copyright (C) 2014-2020 alvations
# URL:
# For license information, see LICENSE.md

from __future__ import absolute_import, print_function

import sys
import time

from wn import WordNet

__builtins__['wn'] = WordNet()

__version__ = '1.2.1'

# Warm up the library.
print('Warming up PyWSD (takes ~10 secs)...', end=' ', file=sys.stderr, flush=True)

