"""COAIA Model Integration for Arena System
Provides utilities to register and detect COAIA fine-tuned models so they can
be surfaced alongside existing Vertex AI and Model Garden models.
"""

from typing import List, Optional
import os

from config.default import Default

config = Default()


class COAIAModelManager:
    """Utility class helping the arena discover COAIA fine-tuned models."""

    # NOTE: keep this mapping small – the objective is to expose the latest
    # models available from the COAIA pipeline.  The values below should be
    # updated by the automated model_updater script when new models are
    # produced.
    _COAIA_MODELS: dict[str, str] = {
        # human-friendly name         model id visible to the OpenAI client
        "coaia-nano": "coaia250707n1b2",
        "coaia-regular": "coaia250708r1a3",
        "coaia-latest": "ft:gpt-4.1-nano-2025-04-14:jgwill:coaia250709n1b3:BrWkpMDi",
    }

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Return list of COAIA model identifiers."""
        return list(cls._COAIA_MODELS.values())

    @classmethod
    def is_coaia_model(cls, model_name: str) -> bool:
        """Return True if *model_name* is managed by COAIA."""
        return model_name in cls._COAIA_MODELS.values()

    @classmethod
    def add_model(cls, display_name: str, model_id: str) -> None:
        """Dynamically register a new COAIA model."""
        cls._COAIA_MODELS[display_name] = model_id

    # Convenience helpers -------------------------------------------------

    @classmethod
    def endpoint_for(cls, model_name: str) -> Optional[str]:
        """Return Vertex AI style endpoint path for *model_name* if applicable.

        Today COAIA models are accessed through the OpenAI API so we simply
        return *None*.  This helper exists for future compatibility in case
        the distribution channel changes.
        """
        if not cls.is_coaia_model(model_name):
            return None
        # If at some point COAIA models are hosted on Vertex AI we could build
        # an endpoint path here using the configured project/location.
        return None 