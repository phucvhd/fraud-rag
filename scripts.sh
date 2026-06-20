#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

case "$1" in
    "start")
        python -m services.consumer.consumer &
        python -m services.embedder.worker &
        uvicorn services.api.main:app --port 8000
        ;;
    "test")
        pytest test/ -s
        ;;
    "clean")
        find . -type d -name "__pycache__" -exec rm -rf {} +
        echo "Cleaned."
        ;;
    *)
        echo "Usage: ./scripts.sh {start|test|clean}"
        ;;
esac