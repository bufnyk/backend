import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import yaml


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.backend = self.root / "backend"
        self.host = self.root / "host"
        self.bin = self.root / "bin"
        self.host.mkdir()
        self.bin.mkdir()
        shutil.copytree(Path(__file__).parent, self.backend / "deploy", ignore=shutil.ignore_patterns("__pycache__"))
        self.backend.joinpath(".env").write_text("TEST_ONLY=1\n")
        shutil.copy(Path(__file__).parents[1] / "docker-compose.yml", self.backend / "docker-compose.yml")
        self.host.joinpath("docker-compose.yml").write_text("services: {}\n")
        self.original_caddy = 'example.com {\n    respond "Existing service"\n}\n'
        self.host.joinpath("Caddyfile").write_text(self.original_caddy)
        self.log = self.root / "calls.jsonl"
        command = '''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
args = sys.argv[1:]
name = Path(sys.argv[0]).name
with open(os.environ['TEST_COMMAND_LOG'], 'a') as stream:
    stream.write(json.dumps([name, *args]) + '\\n')
failure = os.environ.get('TEST_FAIL_AT')
if name == 'curl':
    sys.exit(1 if failure == 'public-health' else 0)
if args[0] == 'ps':
    print(os.environ.get('TEST_LEGACY_IDS', 'old-gateway\\nold-chatbot'))
elif args[0] == 'inspect':
    service = 'api_gateway' if args[-1] == 'old-gateway' else 'chatbot'
    service = os.environ.get('TEST_LEGACY_SERVICE', service)
    directory = os.environ.get('TEST_LEGACY_DIRECTORY', os.environ['TEST_BACKEND_DIR'])
    print('backend|' + service + '|' + directory)
elif args[0] == 'compose':
    for value, key in (('up', 'up'), ('wget', 'upstream'), ('validate', 'validate'), ('reload', 'reload')):
        if value in args and failure == key:
            sys.exit(1)
'''
        for name in ("docker", "curl"):
            path = self.bin / name
            path.write_text(command)
            path.chmod(0o755)
        self.environment = {
            **os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}",
            "DEPLOY_ROOT": str(self.host), "TMPDIR": str(self.root),
            "TEST_BACKEND_DIR": str(self.backend), "TEST_COMMAND_LOG": str(self.log),
            "NO_CACHE": "false",
        }

    def deploy(self, **environment):
        return subprocess.run(
            ["bash", str(self.backend / "deploy/deploy.sh")],
            env={**self.environment, **environment}, capture_output=True, text=True, timeout=10,
        )

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def assert_no_legacy_mutation(self):
        self.assertFalse(any(call[0] == "docker" and call[1] in {"update", "stop", "rm"} for call in self.calls()))

    def test_success_switches_traffic_before_removing_verified_old_containers(self):
        result = self.deploy()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        health = next(i for i, call in enumerate(calls) if call[0] == "curl")
        stop = next(i for i, call in enumerate(calls) if call[:2] == ["docker", "stop"])
        self.assertLess(health, stop)
        self.assertIn(["docker", "rm", "old-gateway", "old-chatbot"], calls)
        up = next(call for call in calls if call[:2] == ["docker", "compose"] and "up" in call)
        self.assertIn("--wait", up)
        self.assertIn("--remove-orphans", up)
        caddy = self.host.joinpath("Caddyfile").read_text()
        self.assertIn(self.original_caddy, caddy)
        self.assertIn("reverse_proxy vectix-backend-api:8000", caddy)
        self.assertFalse(any("--volumes" in call or call[:2] == ["docker", "volume"] for call in calls))

    def test_unhealthy_services_do_not_remove_old_containers(self):
        self.assertNotEqual(self.deploy(TEST_FAIL_AT="up").returncode, 0)
        self.assert_no_legacy_mutation()
        self.assertEqual(self.host.joinpath("Caddyfile").read_text(), self.original_caddy)

    def test_unreachable_unique_upstream_does_not_switch_traffic_or_remove_old(self):
        self.assertNotEqual(self.deploy(TEST_FAIL_AT="upstream").returncode, 0)
        self.assert_no_legacy_mutation()
        self.assertEqual(self.host.joinpath("Caddyfile").read_text(), self.original_caddy)

    def test_invalid_caddy_configuration_is_restored(self):
        self.assertNotEqual(self.deploy(TEST_FAIL_AT="validate").returncode, 0)
        self.assert_no_legacy_mutation()
        self.assertEqual(self.host.joinpath("Caddyfile").read_text(), self.original_caddy)

    def test_failed_public_health_check_preserves_old_containers(self):
        self.assertNotEqual(self.deploy(TEST_FAIL_AT="public-health").returncode, 0)
        self.assert_no_legacy_mutation()

    def test_unrelated_project_working_directory_blocks_cleanup(self):
        self.assertNotEqual(self.deploy(TEST_LEGACY_DIRECTORY="/unrelated/application").returncode, 0)
        self.assert_no_legacy_mutation()

    def test_unknown_legacy_service_blocks_cleanup(self):
        self.assertNotEqual(self.deploy(TEST_LEGACY_SERVICE="postgres").returncode, 0)
        self.assert_no_legacy_mutation()

    def test_repeated_deployment_with_no_legacy_containers_is_safe(self):
        result = self.deploy(TEST_LEGACY_IDS="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_no_legacy_mutation()

    def test_compose_pins_project_name_and_unique_gateway_alias(self):
        config = yaml.safe_load(self.backend.joinpath("docker-compose.yml").read_text())
        self.assertEqual(config["name"], "vectix-backend")
        gateway = config["services"]["api_gateway"]
        self.assertIn("vectix-backend-api", gateway["networks"]["caddy_net"]["aliases"])
        self.assertEqual(config["services"]["chatbot"]["networks"], ["backend_net"])


if __name__ == "__main__":
    unittest.main()
