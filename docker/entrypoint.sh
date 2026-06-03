#!/bin/bash
#
# MoziFlux entrypoint — runs before the main container command.
# Fixes file ownership so the container works for any host UID.
#
# The base hermes-agent image creates user 'hermes' with UID 10000.
# When the host has a different UID (e.g. 1018), run_container.sh passes
# HERMES_UID/HERMES_GID so we can change the container user's UID.
# But the conda env and other /opt files were created with UID 10000,
# so we must chown them to match at startup.
#

set -e

echo "[entrypoint] Starting MoziFlux entrypoint..."

# Get the target UID/GID (default to 10000 if not set)
TARGET_UID=${HERMES_UID:-10000}
TARGET_GID=${HERMES_GID:-$TARGET_UID}

# Check if we need to remap the hermes user
if [ "$TARGET_UID" != "10000" ] || [ "$TARGET_GID" != "10000" ]; then
    echo "[entrypoint] Remapping hermes user from 10000 to ${TARGET_UID}:${TARGET_GID}"

    # Change the group if needed
    if [ "$(getent group hermes | cut -d: -f3)" != "$TARGET_GID" ]; then
        groupmod -g $TARGET_GID hermes
    fi

    # Change the user if needed
    if [ "$(id -u hermes)" != "$TARGET_UID" ]; then
        usermod -u $TARGET_UID -g $TARGET_GID hermes
    fi
else
    echo "[entrypoint] No UID remapping needed (using default 10000)"
fi

# change ownership of miniconda3 packages for the agent to be able to apply patches to packages
# These were created by root
chown -R hermes:hermes /opt/miniconda3 2>/dev/null || true

echo "[entrypoint] Switching to hermes user and executing: $@"

# Switch to hermes user and run the command
exec su - hermes -c "$@"
