import logging

import gradio as gr

from model import GENERATION_DEFAULTS

logger = logging.getLogger(__name__)


def build_gradio_ui_for(inference_fn):
    '''
    Builds a Gradio UI to interact with the model. Big thanks to TearGosling for
    the initial version that inspired this.
    '''
    with gr.Blocks(title="Pygmalion", analytics_enabled=False) as interface:
        history_for_gradio = gr.State([])
        history_for_model = gr.State([])
        generation_settings = gr.State(GENERATION_DEFAULTS)

        def _update_generation_settings(
            original_settings,
            param_name,
            new_value,
        ):
            '''
            Merges `{param_name: new_value}` into `original_settings` and
            returns a new dictionary.
            '''
            updated_settings = {**original_settings, param_name: new_value}
            logging.info("Generation settings updated to: `%s`",
                         updated_settings)
            return updated_settings

        def _run_inference(
            model_history,
            gradio_history,
            user_input,
            generation_settings,
            *char_setting_states,
        ):
            '''
            Runs inference on the model, and formats the returned response for
            the Gradio state and chatbot component.
            '''
            char_name = char_setting_states[0]
            user_name = char_setting_states[1]

            inference_result = inference_fn(model_history, user_input,
                                            generation_settings,
                                            *char_setting_states)

            # TODO(11b): Consider turning `inference_result_for_gradio` into
            # HTML so line breaks are preserved.
            inference_result_for_gradio = inference_result \
                .replace(f"{char_name}:", f"**{char_name}:**") \
                .replace("<USER>", user_name)

            model_history.append(f"You: {user_input}")
            model_history.append(inference_result)
            gradio_history.append((user_input, inference_result_for_gradio))

            return None, model_history, gradio_history, gradio_history

        def _regenerate(
            model_history,
            gradio_history,
            generation_settings,
            *char_setting_states,
        ):
            '''Regenerates the last response.'''
            return _run_inference(
                model_history[:-2],
                gradio_history[:-1],
                model_history[-2].replace("You: ", ""),
                generation_settings,
                *char_setting_states,
            )

        def _undo_last_exchange(model_history, gradio_history):
            '''Undoes the last exchange (message pair).'''
            return model_history[:-2], gradio_history[:-1], gradio_history[:-1]

        with gr.Tab("Character Settings"):
            char_setting_states = _build_character_settings_ui()

        with gr.Tab("Chat Window"):
            chatbot = gr.Chatbot(
                label="Your conversation will show up here").style(
                    color_map=("#326efd", "#212528"))
            message = gr.Textbox(
                label="Your message (hit Enter to send)",
                placeholder="Write a message...",
            )
            message.submit(
                fn=_run_inference,
                inputs=[
                    history_for_model, history_for_gradio, message,
                    generation_settings, *char_setting_states
                ],
                outputs=[
                    message, history_for_model, history_for_gradio, chatbot
                ],
            )

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                send_btn.click(
                    fn=_run_inference,
                    inputs=[
                        history_for_model, history_for_gradio, message,
                        generation_settings, *char_setting_states
                    ],
                    outputs=[
                        message, history_for_model, history_for_gradio, chatbot
                    ],
                )

                regenerate_btn = gr.Button("Regenerate")
                regenerate_btn.click(
                    fn=_regenerate,
                    inputs=[
                        history_for_model, history_for_gradio,
                        generation_settings, *char_setting_states
                    ],
                    outputs=[
                        message, history_for_model, history_for_gradio, chatbot
                    ],
                )

                undo_btn = gr.Button("Undo last exchange")
                undo_btn.click(
                    fn=_undo_last_exchange,
                    inputs=[history_for_model, history_for_gradio],
                    outputs=[history_for_model, history_for_gradio, chatbot],
                )

        with gr.Tab("Generation Settings"):
            _build_generation_settings_ui(
                state=generation_settings,
                fn=_update_generation_settings,
            )

    return interface


def _build_character_settings_ui():
    with gr.Column():
        with gr.Row():
            char_name = gr.Textbox(
                label="Character Name",
                placeholder="The character's name",
            )
            user_name = gr.Textbox(
                label="Your Name",
                placeholder="How the character should call you",
            )

        char_persona = gr.Textbox(
            label="Character Persona",
            placeholder=
            "Describe the character's persona here. Think of this as CharacterAI's description + definitions in one box.",
            lines=4,
        )
        char_greeting = gr.Textbox(
            label="Character Greeting",
            placeholder=
            "Write the character's greeting here. They will say this verbatim as their first response.",
            lines=3,
        )

        world_scenario = gr.Textbox(
            label="Scenario",
            placeholder=
            "Optionally, describe the starting scenario in a few short sentences.",
        )
        example_dialogue = gr.Textbox(
            label="Example Chat",
            placeholder=
            "Optionally, write in an example chat here. This is useful for showing how the character should behave, for example.",
            lines=4,
        )

    return char_name, user_name, char_persona, char_greeting, world_scenario, example_dialogue


def _build_generation_settings_ui(state, fn):
    with gr.Row():
        with gr.Column():
            max_new_tokens = gr.Slider(
                16,
                512,
                value=GENERATION_DEFAULTS["max_new_tokens"],
                step=4,
                label="max_new_tokens",
            )
            max_new_tokens.change(
                lambda state, value: fn(state, "max_new_tokens", value),
                inputs=[state, max_new_tokens],
                outputs=state,
            )

            temperature = gr.Slider(
                0.1,
                2,
                value=GENERATION_DEFAULTS["temperature"],
                step=0.01,
                label="temperature",
            )
            temperature.change(
                lambda state, value: fn(state, "temperature", value),
                inputs=[state, temperature],
                outputs=state,
            )

            top_p = gr.Slider(
                0.0,
                1.0,
                value=GENERATION_DEFAULTS["top_p"],
                step=0.01,
                label="top_p",
            )
            top_p.change(
                lambda state, value: fn(state, "top_p", value),
                inputs=[state, top_p],
                outputs=state,
            )

        with gr.Column():
            typical_p = gr.Slider(
                0.0,
                1.0,
                value=GENERATION_DEFAULTS["typical_p"],
                step=0.01,
                label="typical_p",
            )
            typical_p.change(
                lambda state, value: fn(state, "typical_p", value),
                inputs=[state, typical_p],
                outputs=state,
            )

            repetition_penalty = gr.Slider(
                1.0,
                3.0,
                value=GENERATION_DEFAULTS["repetition_penalty"],
                step=0.01,
                label="repetition_penalty",
            )
            repetition_penalty.change(
                lambda state, value: fn(state, "repetition_penalty", value),
                inputs=[state, repetition_penalty],
                outputs=state,
            )

            top_k = gr.Slider(
                0,
                100,
                value=GENERATION_DEFAULTS["top_k"],
                step=1,
                label="top_k",
            )
            top_k.change(
                lambda state, value: fn(state, "top_k", value),
                inputs=[state, top_k],
                outputs=state,
            )

    #
    # Some of these explanations are taken from Kobold:
    # https://github.com/KoboldAI/KoboldAI-Client/blob/main/gensettings.py
    #
    # They're passed directly into the `generate` call, so they should exist here:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    #
    with gr.Accordion(label="Helpful information", open=False):
        gr.Markdown("""
        Here's a basic rundown of each setting:

        - `max_new_tokens`: Number of tokens the AI should generate. Higher numbers will take longer to generate.
        - `temperature`: Randomness of sampling. High values can increase creativity but may make text less sensible. Lower values will make text more predictable but can become repetitious.
        - `top_p`: Used to discard unlikely text in the sampling process. Lower values will make text more predictable but can become repetitious. (Put this value on 1 to disable its effect)
        - `top_k`: Alternative sampling method, can be combined with top_p. The number of highest probability vocabulary tokens to keep for top-k-filtering. (Put this value on 0 to disable its effect)
        - `typical_p`: Alternative sampling method described in the paper "Typical Decoding for Natural Language Generation" (10.48550/ARXIV.2202.00666). The paper suggests 0.2 as a good value for this setting. Set this setting to 1 to disable its effect.
        - `repetition_penalty`: Used to penalize words that were already generated or belong to the context (Going over 1.2 breaks 6B models. Set to 1.0 to disable).
        """)
