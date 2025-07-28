# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from models.coaia_integration import COAIAModelManager
from dotenv import load_dotenv
from google import genai
import threading
from config.default import Default

load_dotenv(override=True)
config = Default()

def load_default_models() -> list[str]:
    """Load available image generation models.
    
    Using actual Imagen models available in Vertex AI Media Studio.
    """
    # Include actual Imagen models that are available
    IMAGE_GEN_MODELS = [
        config.MODEL_IMAGEN3_FAST,  # imagen-3.0-fast-generate-001
        config.MODEL_IMAGEN3,       # imagen-3.0-generate-002
    ]
    
    # Add Model Garden models if configured
    if config.MODEL_FLUX1_ENDPOINT_ID:
        IMAGE_GEN_MODELS.append(config.MODEL_FLUX1)
    if config.MODEL_STABLE_DIFFUSION_ENDPOINT_ID:
        IMAGE_GEN_MODELS.append(config.MODEL_STABLE_DIFFUSION)
    
    return IMAGE_GEN_MODELS


# ---------------------------------------------------------------------------
# Text models (Gemini + COAIA)                                               
# ---------------------------------------------------------------------------


def load_text_models() -> list[str]:
    """Return list of text-generation capable model identifiers.

    The arena's forthcoming *text_arena* page relies on this helper to pick
    two random models for every battle.  We keep Gemini as baseline and append
    every COAIA model discovered by the *COAIAModelManager* utility.
    """

    TEXT_MODELS: list[str] = [config.MODEL_GEMINI2]

    # Add every COAIA model managed by the integration layer.
    TEXT_MODELS.extend(COAIAModelManager.get_available_models())

    return TEXT_MODELS


class ModelSetup:
    """Model set up class with caching and thread safety."""

    _client_cache = {}
    _lock = threading.Lock()

    @staticmethod
    def init(
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        """Initializes common model settings with caching and thread safety."""

        if not project_id:
            project_id = config.PROJECT_ID
        if not location:
            location = config.LOCATION
        if not model_id:
            model_id = config.MODEL_ID
        if None in [project_id, location, model_id]:
            raise ValueError("All parameters must be set.")
        
        cache_key = (project_id, location, model_id)
        with ModelSetup._lock:  # Acquire lock for thread safety
            if cache_key not in ModelSetup._client_cache:
                print(f"Initiating genai client with {project_id} in {location} using model: {model_id}")
                client = genai.Client(
                    vertexai=config.INIT_VERTEX,
                    project=project_id,
                    location=location,
                )
                ModelSetup._client_cache[cache_key] = client
            else:
                print(f"Using cached genai client for {project_id} in {location} using model: {model_id}")
            return ModelSetup._client_cache[cache_key], model_id