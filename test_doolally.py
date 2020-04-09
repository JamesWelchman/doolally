
import pytest

from doolally import (
    validate,
    jsonschema,
    ValidationValueError,
    ValidationTypeError,
)
from testexamples import *


def test_animals():
    p1 = {
        "pet": {
            "name": "Dave",
            "age": 5,
            "breed": "German Shepherd",
        },
    }
    validate(p1, AnimalPerson)
    p2 = {
        "pet": {
            "name": "Mr. Tiddles",
            "age": 11,
            "breed": "Burmese",
        },
    }
    validate(p2, AnimalPerson)

    # Missing pet key
    p3 = {
        "boo": {},
    }
    with pytest.raises(ValidationValueError):
        validate(p3, AnimalPerson)

    # Pet has wrong JSON type
    p4 = {
        "pet": ["Mr. Tiddles", "Clyde"],
    }
    with pytest.raises(ValidationValueError):
        validate(p4, AnimalPerson)

    # Non Integer age
    p5 = {
        "pet": {
            "name": "Dave",
            "age": 15.4,
            "breed": "German Shepherd",
        },
    }
    with pytest.raises(ValidationValueError):
        validate(p5, AnimalPerson)


def test_email():
    e1 = {
        "provider": None,
    }
    validate(e1, EmailAddr)

    e2 = {
        "provider": "gmail",
    }
    validate(e2, EmailAddr)


def test_numbers():
    n1 = {
        "numbers": [
            "one",
            1,
            "4",
            "ten",
            7,
        ],
    }
    validate(n1, Numbers)

    valid_latitudes = 34.5, 0, 90, 12
    for v in valid_latitudes:
        validate({"latitude": v}, Numbers)
    invalid_latitudes = -0.01, 90.01, 100
    for v in invalid_latitudes:
        with pytest.raises(ValidationValueError):
            validate({"latitude": v}, Numbers)
    wrong_type_latitudes = "56.4", {"value": 15}
    for v in wrong_type_latitudes:
        with pytest.raises(ValidationTypeError):
            validate({"latitude": v}, Numbers)