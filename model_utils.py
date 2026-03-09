import os


def get_model_basename(model: str) -> str:
    return os.path.basename(os.path.normpath(model))


def _normalized_model_name(model: str) -> str:
    return get_model_basename(model).lower()


def is_llama_family(model: str) -> bool:
    return "llama" in _normalized_model_name(model)


def uses_llama3_chat_template(model: str) -> bool:
    model_name = _normalized_model_name(model)
    return is_llama_family(model) and any(
        token in model_name for token in ("llama-3", "llama3")
    )
