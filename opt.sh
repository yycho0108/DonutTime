#!/usr/bin/env bash

set -ex

PATH="${PATH}:${PWD}/Ipopt-3.7.1-linux-x86_64-gcc4.3.2/bin" python3 opt.py 
