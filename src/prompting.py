import logging
import typing as t

from parsing import parse_messages_from_str

logger = logging.getLogger(__name__)


def build_prompt_for(
    history: t.List[str],
    user_message: str,
    char_name: str,
    char_persona: t.Optional[str] = None,
    example_dialogue: t.Optional[str] = None,
    world_scenario: t.Optional[str] = None,
) -> str:
    '''Converts all the given stuff into a proper input prompt for the model.'''

    # If example dialogue is given, parse the history out from it and append
    # that at the beginning of the dialogue history.
    example_history = parse_messages_from_str(
        example_dialogue, ["You", char_name]) if example_dialogue else []
    concatenated_history = [*example_history, *history]

    # Construct the base turns with the info we already have.
    prompt_turns = [
        # TODO(11b): Shouldn't be here on the original 350M.
        "<START>",

        # TODO(11b): Arbitrary limit. See if it's possible to vary this
        # based on available context size and VRAM instead.
        *concatenated_history[-8:],
        f"You: {user_message}",
        f"{char_name}:",
    ]

    # If we have a scenario or the character has a persona definition, add those
    # to the beginning of the prompt.
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

    # Done!
    logger.debug("Constructed prompt is: `%s`", prompt_turns)
    prompt_str = "\n".join(prompt_turns)
    return prompt_str
