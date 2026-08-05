# Production deployment

The backend is deployed as a separate Compose project under
`/root/n8n-docker/backend`. It joins the existing external `caddy_net` network;
it does not recreate or restart the existing n8n, PostgreSQL, or Caddy services.

The deployment updates only the marked `api.vectixai.com` block in the host
`/root/n8n-docker/Caddyfile`. Before a graceful Caddy reload it creates a backup
and validates the complete configuration. A failed validation or reload restores
the previous file.

Configure a GitHub `production` environment with these secrets:

- `HETZNER_HOST`: server hostname or IP address.
- `HETZNER_SSH_PRIVATE_KEY`: private key accepted by the server for `root`.
- `HETZNER_KNOWN_HOSTS`: verified `known_hosts` entry for the server.
- `BACKEND_ENV_FILE`: complete production `.env` content based on `env_example`.

Optionally set the environment variable `HETZNER_SSH_PORT`; it defaults to `22`.
The workflow runs after every push to `main` and can also be started manually.
