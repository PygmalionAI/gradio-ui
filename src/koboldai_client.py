import logging

import requests

logger = logging.getLogger(__name__)


class KoboldApiServerException(Exception):
    pass


def run_raw_inference_on_kai(
    koboldai_url: str,
    prompt: str,
    **kwargs,
) -> str:
    endpoint = f"{koboldai_url}/api/v1/generate"
    payload = {
        "prompt": prompt,

        # Incredibly low max len for reasons explained in the "while True" loop
        # below.
        "max_length": 32,

        # Disable any pre or post-processing on the KoboldAI side, we'd rather
        # take care of things on our own.
        "frmttriminc": False,
        "frmtrmspch": False,
        "frmtrmblln": False,
        "frmtadsnsp": False,

        # Append any other generation parameters that we didn't handle manually.
        **kwargs,
    }
    generated_text = ""

    # Currently, Kobold doesn't support custom stopping criteria, and their chat
    # mode can't handle multi-line responses. To work around both of those, we
    # use the regular adventure mode generation but keep asking for more tokens
    # until the model starts trying to talk as the user, then we stop.
    while True:
        response = requests.post(endpoint, json=payload)
        if not response.ok:
            error_message = response.text
            raise KoboldApiServerException(
                "The KoboldAI API server returned an error"
                f" (HTTP status code {response.status_code}): {error_message}")

        inference_result = response.json()["results"][0]["text"]
        generated_text += inference_result

        # Model started to talk as us. Stop generating and return results, the
        # rest of the code will take care of trimming it properly.
        if "\nYou:" in generated_text:
            return generated_text

        # Model still hasn't finished what it had to say. Append its output to
        # the prompt and feed it back in.
        logger.debug("Got another %s tokens, but still not done: `%s`",
                     payload["max_length"], generated_text)
        payload["prompt"] += inference_result
