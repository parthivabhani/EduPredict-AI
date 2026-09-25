#!/bin/bash
# EduPredict AI - Start Web Application Server
# Author: Parthiv Abhani (23070521106)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR/web_app"

PORT=5050
echo "=========================================================="
echo " Starting EduPredict AI - Student Performance Platform"
echo " Semester 7 Data Science Mini Project"
echo " Author: Parthiv Abhani (PRN: 23070521106)"
echo " URL: http://localhost:$PORT"
echo "=========================================================="

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Port $PORT is already in use. Killing old instance..."
    kill -9 $(lsof -ti:$PORT) 2>/dev/null
    sleep 1
fi

# Launch server
python3 server.py &
SERVER_PID=$!

sleep 1

# Open browser if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "http://localhost:$PORT"
fi

echo "EduPredict AI server running (PID: $SERVER_PID). Press Ctrl+C to terminate."
wait $SERVER_PID
