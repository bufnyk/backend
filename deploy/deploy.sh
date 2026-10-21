#!/usr/bin/env bash
set -Eeuo pipefail

backend_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deploy_root="${DEPLOY_ROOT:-/root/n8n-docker}"
compose_file="${backend_dir}/docker-compose.yml"
env_file="${backend_dir}/.env"
project_name="vectix-backend"
no_cache="${NO_CACHE:-false}"

exec 9>"${TMPDIR:-/tmp}/vectix-backend.deploy.lock"
if ! flock -n 9; then
  echo "Another Vectix backend deployment is already running" >&2
  exit 1
fi

if [[ ! -f "$env_file" ]]; then
  echo "Missing backend environment file: $env_file" >&2
  exit 1
fi
chmod 600 "$env_file"

if ! docker network inspect caddy_net >/dev/null 2>&1; then
  echo "Required external Docker network caddy_net does not exist" >&2
  exit 1
fi

compose=(docker compose --project-name "$project_name" -f "$compose_file")
"${compose[@]}" config --quiet

if [[ "$no_cache" == "true" ]]; then
  "${compose[@]}" build --no-cache
  "${compose[@]}" up --detach --remove-orphans --wait --wait-timeout 180
else
  "${compose[@]}" up --detach --build --remove-orphans --wait --wait-timeout 180
fi

bash "${backend_dir}/deploy/update-caddy.sh" "$deploy_root"

curl --fail --silent --show-error --max-time 15 https://api.vectixai.com/health
bash "${backend_dir}/deploy/retire-legacy.sh" "$backend_dir"

"${compose[@]}" ps
