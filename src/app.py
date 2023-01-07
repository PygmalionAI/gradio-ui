#!/usr/bin/env python3
import argparse
import logging
import typing as t

from model import build_model_and_tokenizer_for, run_raw_inference
from gradio_ui import build_gradio_ui_for
from parsing import parse_messages_from_str

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# For UI debugging purposes.
DONT_USE_MODEL = False


def main(model_name: str, server_port: int) -> None:
    '''Script entrypoint.'''
    if DONT_USE_MODEL:
        model, tokenizer = None, None

        def inference_fn(*args: t.Any) -> str:
            return "Fake response message"
    else:
        model, tokenizer = build_model_and_tokenizer_for(model_name)

        def inference_fn(history: list[str], user_input: str,
                         generation_settings: dict[str, t.Any],
                         *char_settings: t.Any) -> str:
            # Brittle. Comes from the order defined in gradio_ui.py.
            [
                char_name,
                _user_name,
                char_persona,
                char_greeting,
                world_scenario,
                example_dialogue,
            ] = char_settings

            if len(history) == 0 and char_greeting is not None:
                return f"{char_name}: {char_greeting}"

            example_history = parse_messages_from_str(example_dialogue,
                                                      ["You", char_name])
            concatenated_history = [*example_history, *history]

            # TODO(11b): Not complete and subject to change pretty often.
            # Consider moving out into another file.
            prompt_turns = [
                # TODO(11b): Shouldn't be here on the original 350M.
                "<START>",

                # TODO(11b): Arbitrary limit. See if it's possible to vary this
                # based on available context size and VRAM instead.
                *concatenated_history[-8:],
                f"You: {user_input}",
                f"{char_name}:",
            ]

            if world_scenario:
                prompt_turns.insert(
                    0,
                    f"Scenario: {world_scenario}",
                )

            if char_persona:
                prompt_turns.insert(
                    0,
                    f"{char_name}'s Persona: {char_persona}",
                )

            logger.debug("Constructed prompt is: `%s`", prompt_turns)
            prompt_str = "\n".join(prompt_turns)
            model_output = run_raw_inference(model, tokenizer, prompt_str,
                                             user_input, **generation_settings)
            generated_messages = parse_messages_from_str(
                model_output, ["You", char_name])
            logger.debug("Parsed model response is: `%s`", generated_messages)
            bot_message = generated_messages[0]

            return bot_message

    ui = build_gradio_ui_for(inference_fn)
    ui.launch(server_port=server_port)


def _parse_args_from_argv() -> argparse.Namespace:
    '''Parses arguments coming in from the command line.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-name",
        help="HuggingFace Transformers model name.",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=3000,
        help="Port to listen on.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args_from_argv()
    main(model_name=args.model_name, server_port=args.port)
