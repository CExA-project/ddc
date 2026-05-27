#!/bin/bash

# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

. /etc/profile
. /data/spack/share/spack/setup-env.sh
spack env activate ddc

exec "$@"
