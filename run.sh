#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

python3 server.py --clientnumber 3 --model retinanet & 
sleep 10s  # Sleep for 10s to give the server enough time to start

for i in `seq 0 2`; do # Pick how many clients will be run, same client count for --clientnumber
    echo "Starting client $i"
    python3 client.py --partition=${i} --clientnumber 3 --model retinanet &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
