#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

python3 serverssd.py --clientnumber 4 --model ssd & 
sleep 10s  # Sleep for 10s to give the server enough time to start

for i in `seq 0 3`; do # Pick how many clients will be run, same client count for --clientnumber
    echo "Starting client $i"
    python3 client.py --partition=${i} --clientnumber 4 --model ssd --noniid True &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
