#!/usr/bin/python3.7
#  Copyright (c) 2022 by Reiform. All Rights Reserved.
import json
import sys
import impl.services.modules.utils.image_utils as image_utils

contents = json.loads(sys.stdin.read())
try:
    print(json.dumps({
        "status": 0,
        "data": image_utils.get_image_metadata(contents['path'])
    }))
except:
    print(json.dumps({
        "status": 1,
        "data": 'error getting image metadata'
    }))
