"""Offline tests for independent GovBA-GAR embedding configuration."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from govba.rag import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    EmbeddingProviderStatus,
    get_embedding_provider_status,
)


def make_setting_reader(
    *,
    api_key=None,
    model=None,
):
    """Return a deterministic shared-setting test reader."""

    def reader(
        name,
        default=None,
    ):
        if name == "OPENAI_API_KEY":
            return (
                api_key
                if api_key is not None
                else default
            )

        if (
            name
            == "OPENAI_EMBEDDING_MODEL"
        ):
            return (
                model
                if model is not None
                else default
            )

        return default

    return reader


class TestEmbeddingConfiguration(unittest.TestCase):

    def get_status(
        self,
        *,
        api_key=None,
        enabled=False,
        sdk_available=True,
        model=None,
    ):
        with (
            patch(
                "govba.rag.embedding_config.read_setting",
                side_effect=make_setting_reader(
                    api_key=api_key,
                    model=model,
                ),
            ),
            patch(
                "govba.rag.embedding_config.read_boolean_setting",
                return_value=enabled,
            ),
            patch(
                "govba.rag.embedding_config._sdk_available",
                return_value=sdk_available,
            ),
        ):
            return (
                get_embedding_provider_status()
            )

    def test_status_type_is_available(self):
        status = EmbeddingProviderStatus(
            provider="OpenAI",
            configured=False,
            enabled=False,
            sdk_available=True,
            ready=False,
            model=(
                DEFAULT_OPENAI_EMBEDDING_MODEL
            ),
            mode="Retrieval fallback",
            message="test",
        )

        self.assertEqual(
            status.provider,
            "OpenAI",
        )

    def test_embeddings_are_disabled_by_default(self):
        status = self.get_status(
            api_key="test-key",
            enabled=False,
        )

        self.assertFalse(
            status["enabled"]
        )

        self.assertFalse(
            status["ready"]
        )

    def test_default_model_is_used(self):
        status = self.get_status()

        self.assertEqual(
            status["model"],
            DEFAULT_OPENAI_EMBEDDING_MODEL,
        )

    def test_custom_embedding_model_is_supported(self):
        status = self.get_status(
            model="custom-embedding-model",
        )

        self.assertEqual(
            status["model"],
            "custom-embedding-model",
        )

    def test_ready_requires_key_enablement_and_sdk(self):
        status = self.get_status(
            api_key="test-key",
            enabled=True,
            sdk_available=True,
        )

        self.assertTrue(
            status["configured"]
        )

        self.assertTrue(
            status["enabled"]
        )

        self.assertTrue(
            status["sdk_available"]
        )

        self.assertTrue(
            status["ready"]
        )

    def test_enabled_without_key_is_not_ready(self):
        status = self.get_status(
            api_key=None,
            enabled=True,
            sdk_available=True,
        )

        self.assertFalse(
            status["configured"]
        )

        self.assertFalse(
            status["ready"]
        )

    def test_missing_sdk_is_not_ready(self):
        status = self.get_status(
            api_key="test-key",
            enabled=True,
            sdk_available=False,
        )

        self.assertFalse(
            status["ready"]
        )

    def test_status_never_exposes_api_key(self):
        secret_value = (
            "PRIVATE-KEY-MUST-NOT-LEAK"
        )

        status = self.get_status(
            api_key=secret_value,
            enabled=True,
        )

        self.assertNotIn(
            "api_key",
            status,
        )

        self.assertNotIn(
            secret_value,
            repr(status),
        )

    def test_embedding_switch_is_independent_from_ai_switch(self):
        with (
            patch(
                "govba.rag.embedding_config.read_setting",
                side_effect=make_setting_reader(
                    api_key="test-key",
                ),
            ),
            patch(
                "govba.rag.embedding_config.read_boolean_setting",
                return_value=False,
            ) as boolean_reader,
            patch(
                "govba.rag.embedding_config._sdk_available",
                return_value=True,
            ),
        ):
            status = (
                get_embedding_provider_status()
            )

        boolean_reader.assert_called_once_with(
            "EMBEDDINGS_ENABLED",
            default=False,
        )

        self.assertFalse(
            status["enabled"]
        )

        self.assertFalse(
            status["ready"]
        )


if __name__ == "__main__":
    unittest.main()
