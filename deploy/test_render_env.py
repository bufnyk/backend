import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "render_env", Path(__file__).with_name("render-env.py")
)
render_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_env)


class RenderEnvironmentTests(unittest.TestCase):
    def test_overrides_old_secret_and_preserves_other_settings(self):
        result = render_env.render_environment(
            'OTHER=value\nCALLBACK_SECRET="old$value"\n export CALLBACK_SECRET=duplicate\n',
            'test$literal#value@123',
        )
        self.assertEqual(result, "OTHER=value\nCALLBACK_SECRET='test$literal#value@123'\n")

    def test_adds_missing_secret_and_escapes_apostrophe(self):
        self.assertEqual(
            render_env.render_environment("OTHER=value", "test'quote$value"),
            "OTHER=value\nCALLBACK_SECRET='test\\'quote$value'\n",
        )

    def test_rejects_missing_or_multiline_secret(self):
        for secret in ("", "test\nOTHER=injected", "test\rvalue", "test\0value"):
            with self.subTest(secret=secret), self.assertRaises(ValueError):
                render_env.render_environment("OTHER=value", secret)

    def test_rejects_empty_base_environment(self):
        with self.assertRaises(ValueError):
            render_env.render_environment("", "test-secret")


if __name__ == "__main__":
    unittest.main()
