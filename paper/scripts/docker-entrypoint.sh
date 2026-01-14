#!/bin/bash
set -e

case "$1" in
    server)
        echo "Starting feature selection server..."
        echo "  TABLE_NAME: $TABLE_NAME"
        echo "  S3_BUCKET: $S3_BUCKET"
        echo "  REGION: $AWS_DEFAULT_REGION"
        exec uv run uvicorn paper.scripts.feature_selection_server:app --host 0.0.0.0 --port 8000
        ;;

    worker)
        echo "Starting feature selection worker..."
        echo "  URL: $URL"
        echo "  TABLE_NAME: $TABLE_NAME"
        echo "  S3_BUCKET: $S3_BUCKET"
        echo "  REGION: $AWS_DEFAULT_REGION"
        exec uv run python -m paper.scripts.feature_selection_worker
        ;;

    eval-server)
        echo "Starting evaluation server..."
        exec uv run uvicorn paper.scripts.eval_server:app --host 0.0.0.0 --port 8000
        ;;

    eval-worker)
        echo "Starting evaluation worker..."
        exec uv run python -m paper.scripts.eval_worker
        ;;

    shell)
        exec /bin/bash
        ;;

    help|--help|-h|"")
        echo "citrees distributed experiment container"
        echo ""
        echo "Commands:"
        echo "  server       Start feature selection server (port 8000)"
        echo "  worker       Start feature selection worker"
        echo "  eval-server  Start evaluation server (port 8000)"
        echo "  eval-worker  Start evaluation worker"
        echo "  shell        Open bash shell"
        echo ""
        echo "Environment variables:"
        echo "  TABLE_NAME           DynamoDB table name"
        echo "  S3_BUCKET            S3 bucket for results"
        echo "  AWS_DEFAULT_REGION   AWS region (default: us-east-1)"
        echo "  URL                  Server URL (workers only)"
        echo ""
        echo "Examples:"
        echo "  docker run -e TABLE_NAME=citrees-results citrees:latest server"
        echo "  docker run -e URL=http://10.0.1.5:8000 -e TABLE_NAME=citrees-results citrees:latest worker"
        ;;

    *)
        exec "$@"
        ;;
esac
