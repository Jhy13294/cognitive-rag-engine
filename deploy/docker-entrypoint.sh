#!/bin/sh
set -eu

if [ "${RUN_COMPOSE_INIT:-true}" = "true" ]; then
  python /app/deploy/init_fullstack.py
fi

exec "$@"
