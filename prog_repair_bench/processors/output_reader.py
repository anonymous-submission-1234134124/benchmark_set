import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from prog_repair_bench.bench_runners.method_gen_multiturn_langgraph import TestResultState


def dict_to_message(msg_dict: dict) -> BaseMessage:
    """Convert a dictionary back to a LangChain message object."""
    msg_type = msg_dict.get("type")
    content = msg_dict.get("content", "")

    if msg_type == "human":
        return HumanMessage(content=content)
    elif msg_type == "ai":
        return AIMessage(content=content)
    elif msg_type == "system":
        return SystemMessage(content=content)
    else:
        # Fallback - try to determine from class name or other indicators
        if "HumanMessage" in str(msg_dict):
            return HumanMessage(content=content)
        elif "AIMessage" in str(msg_dict):
            return AIMessage(content=content)
        elif "SystemMessage" in str(msg_dict):
            return SystemMessage(content=content)
        else:
            # Default to BaseMessage
            return BaseMessage(content=content, type="generic")


def reconstruct_object(obj: Any) -> Any:
    """Recursively reconstruct objects from dictionary representation."""
    if isinstance(obj, dict):
        # Check if this looks like a message object
        if "content" in obj and (
            "type" in obj or any(key in str(obj) for key in ["Message", "content"])
        ):
            # Try to determine if it's a message object
            if "type" in obj or any(
                msg_type in str(obj) for msg_type in ["human", "ai", "system", "Message"]
            ):
                return dict_to_message(obj)

        # Recursively process dictionary values
        return {k: reconstruct_object(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively process list items
        return [reconstruct_object(item) for item in obj]
    else:
        # Return primitive types as-is
        return obj


def load_results_from_jsonl(filename: str | Path) -> list[TestResultState]:
    """
    Load and reconstruct langgraph results from JSONL file.

    Args:
        filename: Path to the JSONL file containing serialized results

    Returns:
        List of reconstructed TestResultState objects (as tuples with indices)
    """
    results = []

    with open(filename, encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                # Parse JSON line
                data = json.loads(line.strip())
                reconstructed_result = reconstruct_object(data)
                results.append(reconstructed_result)

    return results
