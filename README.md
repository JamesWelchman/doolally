# Doolally

Doolally is a JSON sanity checker.
It has a strong focus on good, readable, descriptive errors.
(If your JSON isn't sane, then you'll go doolally!, Geddit!?) 

Project Goals:

    * Zero-copy JSON sanity checker 
    * Strong error reporting including path/expected type etc.
    * Context depdendent sanity checking via collection validators
    * API to generate doolally classes at Runtime

Project Non-Goals:

    * Python Object (de)serializer
    * Allow for collection modification

TODO:

    * Implement [json-schema](https://json-schema.org) generator
    * Recursive schema definitions

Example:

```python
from doolally import (
    Schema,
    String,
    Number,
    validate,
    ValidationValueError,
)

class Person(Schema):
    name = String(required=True)
    age = Number(required=True, is_int=True, signed=False)

# This will pass just fine
validate({"name": "John", "age": 34}, Person)

# And this will raise an exception
try:
    validate({"wrongKey": "John"}, Person)
except ValidationValueError as exc:
    print(exc)
```

[1. Installation](#Installation)

## Installation

To install doolally simply copy and paste doolally.py.
This project is too small/new to be worth packaging.
There are currently no third party dependencies in doolally.

### Testing

To run the test suite [pytest](https://docs.pytest.org/en/latest/)
is required. First clone this repo before running the following
command.

```bash
    $ pytest --doctest-modules -v .
```