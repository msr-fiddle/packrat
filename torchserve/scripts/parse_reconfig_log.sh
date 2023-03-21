#!/bin/bash

cat logs/access_log.log | cut -d " " -f 1,8 | cut -d "T" -f 2 | sed "s/\s/,/g" | awk 'BEGIN{print "Time,ID,Latency"}; NR > 100 { print }'