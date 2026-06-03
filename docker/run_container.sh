#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p $HOME/.hermes

# Usage: ./run_container.sh [command] [CPUS] [MEMORY]
#   command - "shell" for bash, "hermes" for hermes chat, empty defaults to hermes
#   CPUS    - number of CPU cores (e.g. 4) — empty for unlimited
#   MEMORY  - memory limit (e.g. 8g, 512m) — empty for unlimited
COMMAND="${1:-hermes}"
CPUS="${2:-}"
MEMORY="${3:-}"

CPU_FLAG=""
MEM_FLAG=""
[ -n "$CPUS" ]   && CPU_FLAG="--cpus=${CPUS}"
[ -n "$MEMORY" ] && MEM_FLAG="--memory=${MEMORY}"

docker build -t moziflux:latest $SCRIPT_DIR

CONTAINER_NAME="moziflux"

# Stop/remove existing container if present
docker stop $CONTAINER_NAME 2>/dev/null && docker rm $CONTAINER_NAME 2>/dev/null

# Run detached with s6-overlay (default entrypoint)
# Pass HERMES_UID/HERMES_GID so the entrypoint script can chown the conda env
# to match the host user's UID. This makes the image work for any host user.
# CMD is sleep infinity so the main process never exits (no restart loop).
# Use `docker exec -it <name> hermes` to start interactive Hermes.
# without HERMES_UID and HERMES_GID hermes cannot write to host volumes and user cannot read .hermes directory
docker run -d --name $CONTAINER_NAME \
    --restart unless-stopped \
    --network host \
    $CPU_FLAG \
    $MEM_FLAG \
    -e HERMES_UID=$(id -u) \
    -e HERMES_GID=$(id -g) \
    -v $HOME/.hermes:/opt/data \
    -v $SCRIPT_DIR/..:/opt/moziflux \
    moziflux:latest bash -c "sleep infinity"

# If command is "exec", just print the exec command and exit
if [ "$COMMAND" = "exec" ]; then
    echo "Container '$CONTAINER_NAME' is running."
    echo "Attach with:  docker exec -it $CONTAINER_NAME /bin/bash"
    echo "Or Hermes:    docker exec -it $CONTAINER_NAME hermes"
    exit 0
fi

# Follow logs to see startup, then exec into it
echo "Starting container '$CONTAINER_NAME'..."
sleep 3
docker exec -it $CONTAINER_NAME $COMMAND
