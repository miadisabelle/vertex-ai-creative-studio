"""Text Generation Arena Page

Allow users to pit two text-generation models (Gemini baseline vs COAIA
fine-tuned models) against each other.  Users vote for the better response,
which updates the shared ELO rating store.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mesop as me

from dataclasses import field

from models.set_up import load_text_models
from models.gemini_model import generate_content as gemini_generate
from models.coaia_integration import COAIAModelManager

from common.metadata import update_elo_ratings
from components.header import header

# ---------------------------------------------------------------------------
# Helper functions                                                            
# ---------------------------------------------------------------------------


def _generate_with_openai(model: str, prompt: str) -> str:
    """Generate completion using the OpenAI Python SDK (>=1.0)."""
    import openai  # Local import to avoid hard dependency if not needed

    client = openai.OpenAI()  # type: ignore[attr-defined]

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        temperature=0.7,
    )

    return resp.choices[0].message.content.strip()


def _generate_response(model: str, prompt: str) -> str:
    """Route generation call based on model provider."""

    # COAIA models are accessed through OpenAI API
    if COAIAModelManager.is_coaia_model(model):
        return _generate_with_openai(model, prompt)

    # Gemini baseline uses Vertex AI client wrapper
    if model.startswith("gemini"):
        return gemini_generate(prompt)

    # Fallback – echo the prompt (should not happen in normal flow)
    return f"[No backend] {model} says: {prompt}"


# ---------------------------------------------------------------------------
# Mesop State                                                                 
# ---------------------------------------------------------------------------


@me.stateclass
class TextArenaState:
    """State container for the text arena page."""

    prompt: str = ""
    is_loading: bool = False
    model1: str = ""
    model2: str = ""
    response1: str = ""
    response2: str = ""
    chosen_model: str = ""
    study: str = "text_live"
    textarea_key: int = 0  # Force rerender of textarea when resetting


# ---------------------------------------------------------------------------
# Event handlers                                                              
# ---------------------------------------------------------------------------


def _run_battle(prompt: str):
    """Generate content from two models in parallel."""

    state = me.state(TextArenaState)
    state.response1 = ""
    state.response2 = ""

    with ThreadPoolExecutor() as pool:
        futures = {
            pool.submit(_generate_response, state.model1, prompt): "resp1",
            pool.submit(_generate_response, state.model2, prompt): "resp2",
        }

        for future in as_completed(futures):
            tag = futures[future]
            try:
                res = future.result()
            except Exception as exc:  # pylint: disable=broad-except
                res = f"Error generating response: {exc}"
            if tag == "resp1":
                state.response1 = res
            else:
                state.response2 = res
            yield  # Allow UI to update incrementally


def on_click_battle(e: me.ClickEvent):  # pylint: disable=unused-argument
    """Start a new text battle with the current prompt."""

    state = me.state(TextArenaState)

    # Pick two distinct models
    models = load_text_models()
    state.model1, state.model2 = random.sample(models, 2)

    state.chosen_model = ""
    state.is_loading = True
    yield

    # Run generation
    yield from _run_battle(state.prompt)

    state.is_loading = False
    yield


def on_click_vote(e: me.ClickEvent):  # pylint: disable=unused-argument
    """Handle vote button click."""

    state = me.state(TextArenaState)
    winner = state.model1 if e.key == "vote_left" else state.model2
    state.chosen_model = winner

    update_elo_ratings(
        state.model1,
        state.model2,
        winner,
        [state.response1, state.response2],
        state.prompt,
        state.study,
    )

    yield
    time.sleep(1)
    yield

    # Automatically launch another battle with the same prompt
    on_click_battle(e)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Page content builder                                                        
# ---------------------------------------------------------------------------


def text_arena_page_content():  # noqa: C901
    """Render the text arena page."""

    state = me.state(TextArenaState)

    with me.box(style=me.Style(padding=me.Padding.all(24))):
        header("Text Generation Arena", "chat_bubble")

        # Prompt input row --------------------------------------------------
        me.textarea(
            key=f"prompt-{state.textarea_key}",
            label="Enter prompt",
            value=state.prompt,
            rows=4,
            on_input=lambda ev: setattr(state, "prompt", ev.value),
            style=me.Style(width="100%"),
        )

        me.button(
            "Start Battle",
            on_click=on_click_battle,
            disabled=state.prompt.strip() == "" or state.is_loading,
            style=me.Style(margin=me.Margin(top=12)),
        )

        if state.is_loading:
            me.progress_spinner()

        # Responses ---------------------------------------------------------
        if state.response1 and state.response2:
            with me.box(style=me.Style(display="flex", gap="20px", margin=me.Margin(top=20))):
                with me.box(style=me.Style(flex="1", border="1px solid #ccc", padding="10px")):
                    me.text(f"Model: {state.model1}", style=me.Style(font_weight="bold"))
                    me.text(state.response1)

                with me.box(style=me.Style(flex="1", border="1px solid #ccc", padding="10px")):
                    me.text(f"Model: {state.model2}", style=me.Style(font_weight="bold"))
                    me.text(state.response2)

            if not state.chosen_model:
                with me.box(style=me.Style(display="flex", gap="20px", justify_content="center", margin=me.Margin(top=12))):
                    me.button("Left Better", key="vote_left", on_click=on_click_vote)
                    me.button("Right Better", key="vote_right", on_click=on_click_vote)
            else:
                me.text(f"You voted: {state.chosen_model}", style=me.Style(margin=me.Margin(top=12))) 