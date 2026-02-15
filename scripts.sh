#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

case "$1" in
    "start")
        python -m services.consumer.main &
        python -m services.embedder.worker &
        uvicorn services.api.main:app --port 8000
        ;;
    "test")
        pytest tests/test_rag_flow.py -s
        ;;
    "clean")
        find . -type d -name "__pycache__" -exec rm -rf {} +
        echo "Cleaned."
        ;;
    *)
        echo "Usage: ./scripts.sh {start|test|clean}"
        ;;
esac