#!/bin/bash

# Simple script to start a local web server for the portfolio

echo "Starting local web server..."
echo "Open your browser and visit:"
echo "  http://localhost:8000/blog/ds-interview-cheatsheet.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Try Python 3 first, then Python 2, then suggest alternatives
if command -v python3 &> /dev/null; then
    python3 -m http.server 8000
elif command -v python &> /dev/null; then
    python -m SimpleHTTPServer 8000
else
    echo "Python not found. Please install Python or use one of these alternatives:"
    echo "  - Node.js: npx http-server"
    echo "  - VS Code: Install 'Live Server' extension"
    echo "  - PHP: php -S localhost:8000"
fi

