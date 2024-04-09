#!/bin/bash

# Kill the Flask backend process running in the tmux session
flask_pid=$(lsof -t -i :5000)
node_pid=$(lsof -t -i :3000)

echo "Node.js PID: $node_pid"
echo "Flask PID: $flask_pid"

# Check if the Node.js process is running
if [ -n "$node_pid" ]; then
  # Kill the Node.js process
  kill "$node_pid"
  echo "Node.js process running on port 3000 (PID: $node_pid) has been terminated."
else
  echo "No running Node.js process found on port 3000."
fi


if [ -n "$flask_pid" ]; then
  kill "$flask_pid"
  echo "Flask process running on port 5000 (PID: $flask_pid) has been terminated."
else
  echo "No running Flask process found on port 5000."
fi

