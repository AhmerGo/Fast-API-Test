#!/bin/bash

# Change directory to the Flask backend
cd /opt/Hardin-Helper-AI/flask-backend

# Start a detached tmux session running the Flask backend
tmux new-session -d -s my_session 'source venv/bin/activate && python app.py'

# Change directory to the location of the npm project

# Run npm start as a background process
npm start &
sleep 2

# Return to the terminal
cd -
