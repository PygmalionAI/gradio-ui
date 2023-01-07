import re
import typing as t

BAD_CHARS_FOR_REGEX_REGEX = re.compile(r"[-\/\\^$*+?.()|[\]{}]")


def _sanitize_string_for_use_in_a_regex(string: str) -> str:
    '''Sanitizes `string` so it can be used inside of a regexp.'''
    return BAD_CHARS_FOR_REGEX_REGEX.sub(r"\\\g<0>", string)


def parse_messages_from_str(string: str, names: t.List[str]) -> t.List[str]:
    '''
    Given a big string containing raw chat history, this function attempts to
    parse it out into a list where each item is an individual message.
    '''
    sanitized_names = [
        _sanitize_string_for_use_in_a_regex(name) for name in names
    ]

    speaker_regex = re.compile(rf"^({'|'.join(sanitized_names)}): ?",
                               re.MULTILINE)

    message_start_indexes = []
    for match in speaker_regex.finditer(string):
        message_start_indexes.append(match.start())

    # FIXME(11b): One of these returns is silently dropping the last message.
    if len(message_start_indexes) < 2:
        # Single message in the string.
        return [string.strip()]

    prev_start_idx = message_start_indexes[0]
    messages = []

    for start_idx in message_start_indexes[1:]:
        message = string[prev_start_idx:start_idx].strip()
        messages.append(message)
        prev_start_idx = start_idx

    return messages


def serialize_chat_history(history: t.List[str]) -> str:
    '''Given a structured chat history object, collapses it down to a string.'''
    return "\n".join(history)
