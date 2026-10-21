#!/usr/bin/env bash
set -Eeuo pipefail

backend_dir="${1:?Expected backend deployment directory}"
legacy_project="backend"
container_list="$(docker ps --all --quiet --filter "label=com.docker.compose.project=${legacy_project}")"
if [[ -z "$container_list" ]]; then
  echo "No legacy backend containers remain"
  exit 0
fi
mapfile -t legacy_containers <<< "$container_list"

# The generic old project name could be used by another installation. Verify
# every container before stopping any of them, and never remove data volumes.
for container_id in "${legacy_containers[@]}"; do
  metadata="$(docker inspect --format \
    '{{index .Config.Labels "com.docker.compose.project"}}|{{index .Config.Labels "com.docker.compose.service"}}|{{index .Config.Labels "com.docker.compose.project.working_dir"}}' \
    "$container_id")"
  IFS='|' read -r project service working_dir <<< "$metadata"
  if [[ "$project" != "$legacy_project" || "$working_dir" != "$backend_dir" ]]; then
    echo "Refusing to remove unrelated container: $container_id" >&2
    exit 1
  fi
  case "$service" in
    api_gateway|redis_db|chatbot|data_distributor|processor|prompt_rewrite|log-error_container) ;;
    *) echo "Refusing to remove unknown legacy service: $service" >&2; exit 1 ;;
  esac
done

docker update --restart=no "${legacy_containers[@]}" >/dev/null
docker stop --time 60 "${legacy_containers[@]}"
docker rm "${legacy_containers[@]}"
echo "Legacy backend containers removed; data volumes retained"
