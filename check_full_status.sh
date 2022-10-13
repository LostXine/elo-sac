#!/bin/bash
url=`cat search-server-ip`
curl -H 'Content-Type: application/json' $url"get_full_status"
