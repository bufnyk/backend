# Production deployment

The backend is deployed as a separate Compose project under
`/root/n8n-docker/backend`. It joins the existing external `caddy_net` network;
it does not recreate or restart the existing n8n, PostgreSQL, or Caddy services.
The Compose file pins the project name to `vectix-backend`, including manual
`docker compose up` runs. Caddy uses the unique `vectix-backend-api` network
alias rather than the shared `api_gateway` service name.

The deployment updates only the marked `api.vectixai.com` block in the host
`/root/n8n-docker/Caddyfile`. Before a graceful Caddy reload it creates a backup
and validates the complete configuration. A failed validation or reload restores
the previous file.

After all new services are healthy, deployment checks the upstream from Caddy's
network, reloads the route, and verifies the public `/health` endpoint. It then
stops and removes containers from the legacy `backend` project only when their
service and working-directory labels match this installation. Persistent
volumes are retained. A deployment lock prevents concurrent manual and CI runs.

Configure a GitHub `production` environment with these secrets:

- `HETZNER_HOST`: server hostname or IP address.
- `HETZNER_SSH_PRIVATE_KEY`: private key accepted by the server for `root`.
- `HETZNER_KNOWN_HOSTS`: verified `known_hosts` entry for the server.
- `BACKEND_ENV_FILE`: complete production `.env` content based on `env_example`.
- `BACKEND_CALLBACK_SECRET`: raw callback secret, identical to the value expected
  by the Supabase `document-callback` function. Enter the value without wrapping
  quotes. Deployment replaces `CALLBACK_SECRET` from `BACKEND_ENV_FILE` with this
  value and writes it in literal single-quoted form for Compose. This secret is
  required; a missing or empty value aborts environment installation.

Single-quote literal values containing `$` in `BACKEND_ENV_FILE` and `.env` so
Compose does not substitute parts of a secret as environment variables.

Optionally set the environment variable `HETZNER_SSH_PORT`; it defaults to `22`.
The workflow runs after every push to `main` and can also be started manually.
To start it manually, open **Actions → Deploy backend to Hetzner → Run workflow**.
The manual run includes an optional `no_cache` switch for a clean Docker rebuild.

The chatbot service requires `OPENAI_API_KEY` in addition to `GEMINI_API_KEY`.
Keep both keys in the production `BACKEND_ENV_FILE` secret. `/chatbot`,
`/test_chatbot`, and `/copilot` use OpenAI `gpt-6-luna` through the Responses API,
with `gemini-3.8-flash` as fallback. OpenAI requests use `store=false` and low
reasoning effort. The primary model has a 35-second budget; the complete chat
handler has a 90-second budget, below the gateway's 100-second upstream timeout.
Selected attachments are downloaded once and reused on fallback. Tool results
are retained across providers to avoid repeating handoffs or unresolved flags.

Run chatbot regression tests from `chatbot/` with
`venv/bin/python -m unittest discover -v`. The tests mock network and database
access; no API keys are required.

The processor reports import results with a 60-second callback timeout (10 seconds
to connect), up to three attempts, and 2- and 5-second retry delays for transport
errors, HTTP 429, and HTTP 5xx. HTTP status and the callback's JSON result must both
confirm success. Callback failures are logged locally and sent to the error alert
service with the document ID, attempted status, exception type, and traceback.
They never delete successfully stored vectors or send a contradictory `failed`.
Import failures still clean up vectors, scoped by company and hash in batches of
100, and include any cleanup failure in the reported error.
Dataset GET requests use the same 60-second read/10-second connection limits and
up to three attempts for transient transport errors, HTTP 429, and HTTP 5xx.
Only the download is retried; batches already embedded and saved are not repeated.

After confirming that an import's vectors are fully stored, its completion can
be reconciled without downloading or embedding again:

```sh
docker compose exec -T processor python retry_callback.py DOCUMENT_ID
```

This reruns the Supabase completion callback, including its vector count and
replacement handling. The callback checks for a positive count, not completeness;
verify the import before using this recovery command. It exits nonzero if reporting
still fails. Retry state is not persisted automatically across container restarts;
use the document ID in the alert for manual reconciliation.

Run processor regression tests from `processor/` with
`venv/bin/python -m unittest discover -v`. They mock external services and do not
create embeddings or modify a database.

Feedback updates commit the learned instructions, feedback state and configuration
unlock in one database transaction. If one write fails, earlier writes roll back;
unlock cleanup cannot hide the original failure. Chat alerts include exception
types, including transport exceptions with empty messages. The error handler
retains full alerts locally and verifies Telegram's HTTP and JSON acknowledgements.
