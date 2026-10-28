"""JSON Schema assets for funding-optimizer (issue #8).

Publishes ``plan.schema.json`` — the stable wire contract for allocation
plans — plus a dependency-free validator covering the schema subset the
SDK relies on. Use this module to validate payloads produced by
:meth:`AllocationPlan.to_dict` or received from external systems.

The optimizer recommends allocations. It never signs or submits payouts.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from funding_optimizer.errors import ValidationError

__all__ = ["load_plan_schema", "validate_plan_record", "validate_payload"]

_SCHEMA_RESOURCE = "plan.schema.json"


def load_plan_schema() -> dict[str, Any]:
    """Return the published JSON Schema for allocation plans (issue #8)."""
    resource = resources.files("funding_optimizer.schemas").joinpath(_SCHEMA_RESOURCE)
    text = resource.read_text(encoding="utf-8")
    schema = json.loads(text)
    if not isinstance(schema, dict):
        raise ValidationError("plan.schema.json must contain a JSON object")
    return schema


def _check_type(value: Any, expected: Any, path: str) -> None:
    """Validate ``value`` against a JSON-schema ``type`` (or list of types)."""
    allowed = expected if isinstance(expected, list) else [expected]
    checks: dict[str, Any] = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if not any(checks[t](value) for t in allowed if t in checks):
        rendered = "/".join(allowed)
        raise ValidationError(f"{path}: expected type {rendered}, got {type(value).__name__}")


def validate_payload(payload: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate ``payload`` against the supported JSON Schema subset.

    Supports ``type`` (single or union), ``required``, ``properties``,
    ``items``, ``enum``, ``minimum``/``maximum``,
    ``exclusiveMinimum``/``exclusiveMaximum``, ``minLength``, and
    ``maxLength`` — the subset used by ``plan.schema.json``.

    Raises :class:`~funding_optimizer.errors.ValidationError` with a
    ``path``-prefixed message on the first violation.
    """
    if not isinstance(schema, dict):
        raise ValidationError(f"{path}: schema fragment must be an object")

    expected_type = schema.get("type")
    if expected_type is not None:
        _check_type(payload, expected_type, path)

    if "enum" in schema and payload not in schema["enum"]:
        raise ValidationError(f"{path}: value {payload!r} is not one of {schema['enum']}")

    if isinstance(payload, str):
        minimum_length = schema.get("minLength")
        if minimum_length is not None and len(payload) < minimum_length:
            raise ValidationError(f"{path}: string shorter than minLength {minimum_length}")
        maximum_length = schema.get("maxLength")
        if maximum_length is not None and len(payload) > maximum_length:
            raise ValidationError(f"{path}: string longer than maxLength {maximum_length}")

    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        minimum = schema.get("minimum")
        if minimum is not None and payload < minimum:
            raise ValidationError(f"{path}: {payload} is below minimum {minimum}")
        maximum = schema.get("maximum")
        if maximum is not None and payload > maximum:
            raise ValidationError(f"{path}: {payload} is above maximum {maximum}")
        exclusive_minimum = schema.get("exclusiveMinimum")
        if exclusive_minimum is not None and payload <= exclusive_minimum:
            raise ValidationError(f"{path}: {payload} must be greater than {exclusive_minimum}")
        exclusive_maximum = schema.get("exclusiveMaximum")
        if exclusive_maximum is not None and payload >= exclusive_maximum:
            raise ValidationError(f"{path}: {payload} must be less than {exclusive_maximum}")

    if isinstance(payload, dict):
        for required in schema.get("required", []):
            if required not in payload:
                raise ValidationError(f"{path}: missing required property {required!r}")
        properties = schema.get("properties", {})
        for key, value in payload.items():
            if key in properties:
                validate_payload(value, properties[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise ValidationError(f"{path}: unexpected property {key!r}")

    if isinstance(payload, list) and "items" in schema:
        for index, item in enumerate(payload):
            validate_payload(item, schema["items"], f"{path}[{index}]")


def validate_plan_record(record: Any) -> None:
    """Validate an allocation-plan payload against ``plan.schema.json``.

    Args:
        record: Parsed plan payload (``AllocationPlan.to_dict()`` output).

    Raises:
        ValidationError: When the record violates the published schema.
    """
    validate_payload(record, load_plan_schema())
