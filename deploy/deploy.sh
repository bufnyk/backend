#!/usr/bin/env bash
set -Eeuo pipefail

backend_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deploy_root="${DEPLOY_ROOT:-/root/n8n-docker}"
compose_file="${backend_dir}/docker-compose.yml"
env_file="${backend_dir}/.env"
project_name="vectix-backend"

if [[ ! -f "$env_file" ]]; then
  echo "Missing backend environment file: $env_file" >&2
  exit 1
fi
chmod 600 "$env_file"

if ! docker network inspect caddy_net >/dev/null 2>&1; then
  echo "Required external Docker network caddy_net does not exist" >&2
  exit 1
fi

docker compose --project-name "$project_name" -f "$compose_file" config --quiet
docker compose --project-name "$project_name" -f "$compose_file" \
  up --detach --build --wait --wait-timeout 180

bash "${backend_dir}/deploy/update-caddy.sh" "$deploy_root"

docker compose --project-name "$project_name" -f "$compose_file" ps
