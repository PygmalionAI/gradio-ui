import json
import logging

import gradio as gr


def get_generation_defaults(for_kobold):
    defaults = {
        "do_sample": True,
        "max_new_tokens": 196,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 0,
        "typical_p": 1.0,
        "repetition_penalty": 1.05,
    }

    if for_kobold:
        defaults.update({"max_context_length": 768})
    else:
        defaults.update({"penalty_alpha": 0.65})

    return defaults


logger = logging.getLogger(__name__)


def build_gradio_ui_for(inference_fn, for_kobold):
    '''
    Builds a Gradio UI to interact with the model. Big thanks to TearGosling for
    the initial version that inspired this.
    '''
    with gr.Blocks(title="Pygmalion", analytics_enabled=False) as interface:
        history_for_gradio = gr.State([])
        history_for_model = gr.State([])
        generation_settings = gr.State(
            get_generation_defaults(for_kobold=for_kobold))

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
            logging.debug("Generation settings updated to: `%s`",
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
            
        def _save_chat_history(model_history, *char_setting_states):
            '''Saves the current chat history to a .json file.'''
            char_name = char_setting_states[0]
            with open(f"{char_name}_conversation.json", "w") as f:
                f.write(json.dumps({"chat": model_history}))
            return f"{char_name}_conversation.json"
            
                
        def _load_chat_history(file_obj, char_name):
            '''Loads up a chat history from a .json file.'''
            # https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
            def pairwise(iterable):
                # "s -> (s0, s1), (s2, s3), (s4, s5), ..."
                a = iter(iterable)
                return zip(a, a)
                
            file_data = json.loads(file_obj.decode('utf-8'))
            model_history = file_data["chat"]
            # Construct a new gradio history
            new_gradio_history = []
            for human_turn, bot_turn in pairwise(model_history):
                # Handle the situation where convo history may be loaded before character defs
                if char_name == "":
                    # Grab char name from the model history
                    char_name = bot_turn.split(":")[0]
                # Format the user and bot utterances
                user_turn = human_turn.replace("You :", "")
                bot_turn = bot_turn.replace(f"{char_name}:", f"**{char_name}**:")
                
                new_gradio_history.append((user_turn, bot_turn))
                
            return model_history, new_gradio_history, new_gradio_history

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
                
            with gr.Row():
                with gr.Column():
                    chatfile = gr.File(type="binary", file_types=[".json"], interactive=True)
                    chatfile.upload(
                        fn=_load_chat_history,
                        inputs=[chatfile, char_setting_states[0]],
                        outputs=[history_for_model, history_for_gradio, chatbot]
                    )

                    save_char_btn = gr.Button(value="Save Conversation History")
                    save_char_btn.click(_save_chat_history, inputs=[history_for_model, *char_setting_states], outputs=[chatfile])
                with gr.Column():
                    gr.Markdown("""
                        ### To save a chat
                        Click "Save Conversation History". The file will appear above the button and you can click to download it.

                        ### To load a chat
                        Drag a valid .json file onto the upload box, or click the box to browse.
                        
                        **Remember to fill out/load up your character definitions before resuming a chat!**
                    """)

                

        with gr.Tab("Generation Settings"):
            _build_generation_settings_ui(
                state=generation_settings,
                fn=_update_generation_settings,
                for_kobold=for_kobold,
            )

    return interface


def _build_character_settings_ui():    
    def char_file_upload(file_obj):
        file_data = json.loads(file_obj.decode('utf-8'))
        return file_data["char_name"], file_data["char_persona"], file_data["char_greeting"], file_data["world_scenario"], file_data["example_dialogue"]
        
    def char_file_create(char_name, char_persona, char_greeting, world_scenario, example_dialogue):
        with open(char_name + ".json", "w") as f:
            f.write(json.dumps({"char_name": char_name, "char_persona": char_persona, "char_greeting": char_greeting, "world_scenario": world_scenario, "example_dialogue": example_dialogue}))
        return char_name + ".json"

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

        with gr.Row():
            with gr.Column():
                charfile = gr.File(type="binary", file_types=[".json"])
                charfile.upload(fn=char_file_upload, inputs=[charfile], outputs=[char_name, char_persona, char_greeting, world_scenario, example_dialogue])

                save_char_btn = gr.Button(value="Generate Character File")
                save_char_btn.click(char_file_create, inputs=[char_name, char_persona, char_greeting, world_scenario, example_dialogue], outputs=[charfile])
            with gr.Column():
                gr.Markdown("""
                    ### To save a character
                    Click "Generate Character File". The file will appear above the button and you can click to download it.

                    ### To upload a character
                    Drag a valid .json file onto the upload box, or click the box to browse.
                """)

    return char_name, user_name, char_persona, char_greeting, world_scenario, example_dialogue


def _build_generation_settings_ui(state, fn, for_kobold):
    generation_defaults = get_generation_defaults(for_kobold=for_kobold)

    with gr.Row():
        with gr.Column():
            max_new_tokens = gr.Slider(
                16,
                512,
                value=generation_defaults["max_new_tokens"],
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
                value=generation_defaults["temperature"],
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
                value=generation_defaults["top_p"],
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
                value=generation_defaults["typical_p"],
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
                value=generation_defaults["repetition_penalty"],
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
                value=generation_defaults["top_k"],
                step=1,
                label="top_k",
            )
            top_k.change(
                lambda state, value: fn(state, "top_k", value),
                inputs=[state, top_k],
                outputs=state,
            )

            if not for_kobold:
                penalty_alpha = gr.Slider(
                    0,
                    1,
                    value=generation_defaults["penalty_alpha"],
                    step=0.05,
                    label="penalty_alpha",
                )
                penalty_alpha.change(
                    lambda state, value: fn(state, "penalty_alpha", value),
                    inputs=[state, penalty_alpha],
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
        - `typical_p`: Alternative sampling method described in the paper "Typical_p Decoding for Natural Language Generation" (10.48550/ARXIV.2202.00666). The paper suggests 0.2 as a good value for this setting. Set this setting to 1 to disable its effect.
        - `repetition_penalty`: Used to penalize words that were already generated or belong to the context (Going over 1.2 breaks 6B models. Set to 1.0 to disable).
        - `penalty_alpha`: The alpha coefficient when using contrastive search.

        Some settings might not show up depending on which inference backend is being used.
        """)
