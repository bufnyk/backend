#!/usr/bin/env bash
set -Eeuo pipefail

deploy_root="${1:-/root/n8n-docker}"
caddyfile="${deploy_root}/Caddyfile"
host_compose="${deploy_root}/docker-compose.yml"
snippet="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/Caddyfile.api"
begin_marker="# BEGIN VECTIX API (managed by backend deployment)"
end_marker="# END VECTIX API (managed by backend deployment)"

for required_file in "$caddyfile" "$host_compose" "$snippet"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required deployment file does not exist: $required_file" >&2
    exit 1
  fi
done

begin_count="$(grep -Fxc "$begin_marker" "$caddyfile" || true)"
end_count="$(grep -Fxc "$end_marker" "$caddyfile" || true)"
if [[ "$begin_count" != "$end_count" || "$begin_count" -gt 1 ]]; then
  echo "Caddyfile contains inconsistent Vectix API deployment markers" >&2
  exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup="$(mktemp "${caddyfile}.backup.${timestamp}.XXXXXX")"
temporary="$(mktemp "${caddyfile}.tmp.XXXXXX")"
trap 'rm -f "$temporary"' EXIT
cp "$caddyfile" "$backup"

if [[ "$begin_count" -eq 1 ]]; then
  sed "/^${begin_marker//\//\\\/}\$/,/^${end_marker//\//\\\/}\$/d" "$caddyfile" > "$temporary"
else
  cp "$caddyfile" "$temporary"
fi

if [[ "$begin_count" -eq 0 && -s "$temporary" ]]; then
  printf '\n' >> "$temporary"
fi
cat "$snippet" >> "$temporary"

# Copy over the existing bind-mounted file so the Caddy container sees the update.
cp "$temporary" "$caddyfile"

if ! docker compose -f "$host_compose" exec -T caddy \
  caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile; then
  cp "$backup" "$caddyfile"
  echo "Caddy validation failed; restored $backup" >&2
  exit 1
fi

if ! docker compose -f "$host_compose" exec -T caddy \
  caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile; then
  cp "$backup" "$caddyfile"
  docker compose -f "$host_compose" exec -T caddy \
    caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile || true
  echo "Caddy reload failed; restored $backup" >&2
  exit 1
fi

echo "Caddy route for api.vectixai.com is active (backup: $backup)"
