#!/usr/bin/python3.8
#  Copyright (c) 2022 by Reiform. All Rights Reserved.

from string import Template
import json
import sys

contents = json.loads(sys.stdin.read())

if len(contents['previous_results']) != 1:
    exit(1)

dataset_uuid = contents['dataset']['uuid']
t = Template('''{
"status": 0,
"data" : {
    "dataset": {
      "uuid": "${uuid}",
      "classes" : ["class1", "class2"],
      "mean": [0.3, 0.4, 0.1],
      "std_dev": [0.1, 0.12, 0.03],
      "class_files" : {
        "class1" : {
          "/tmp/fileuuid1" : {
            "uuid": "fileuuid1",
            "current_class": "class1",
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          },
          "/tmp/fileuuid2" : {
            "uuid": "fileuuid2",
            "current_class": "class2",
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          }
        },
        "class2" : {
          "/tmp/fileuuid3" : {
            "uuid": "fileuuid3",
            "current_class": "class2",
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          },
          "/tmp/fileuuid4" : {
            "uuid": "fileuuid4",
            "current_class": "class1",
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          }
        }
      }
    },
    "tasks": [ 
      {
        "type" : "ic::diagnose::mislabeled_images",
        "metadata": {
          "outliers" : ["fileuuid1", "fileuuid3"]
        }
      }
    ]
  }
}''')
print(t.substitute(uuid=dataset_uuid))

