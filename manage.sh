#!/bin/bash

# Get the script's directory path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Function to start the services
start_services() {
    echo "Starting services..."

    echo "Init Database..."
    python3 "$SCRIPT_DIR/init_database.py" --recreate-vs

    echo "Starting LLM API Service..."
    nohup python3 "$SCRIPT_DIR/server/llm_api.py" > "$SCRIPT_DIR/logs/llmapi_srv.log" 2>&1 &

    echo "Starting Chatchat API Service..."
    nohup python3 "$SCRIPT_DIR/server/api.py" > "$SCRIPT_DIR/logs/api_srv.log" 2>&1 &

    echo "Starting WebUI Service..."
    nohup streamlit run "$SCRIPT_DIR/webui.py" --browser.gatherUsageStats True --theme.base "light" --theme.primaryColor "#165dff" --theme.secondaryBackgroundColor "#f5f5f5" --theme.textColor "#000000" > "$SCRIPT_DIR/logs/webui_srv.log" 2>&1 &

    echo "All services started."
}

# Function to stop the services
stop_services() {
    echo "Stopping services..."

    pkill -9 -ef llm_api.py
    pkill -9 -ef api.py
    pkill -9 -ef streamlit
    # Additional cleanup if needed

    echo "All services stopped."
}

# Function to check the services status
check_status() {
    echo "Checking services status..."
    pgrep -af llm_api.py
    pgrep -af api.py
    pgrep -af streamlit

    if pgrep -f llm_api.py >/dev/null && pgrep -f api.py >/dev/null && pgrep -f streamlit >/dev/null; then
        echo "All services are running."
    else
        echo "Some services are not running."
    fi
}

# Function to restart the services
restart_services() {
    stop_services
    start_services
}

# Main script
case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
esac

exit 0
