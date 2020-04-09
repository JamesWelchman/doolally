# Doolally

Doolally is a JSON sanity checker and
[json-schema](https://json-schema.org/) generator.
It has a strong focus on good, readable, descriptive errors.
(If your JSON isn't sane, then you'll go doolally!, Geddit!?) 

A trivial example:

```python
from doolally import (
    Schema,
    String,
    Number,
    validate,
    jsonschema,
    ValidationValueError,
)

class Person(Schema):
    jsonschema_description = "A Person instance"

    name = String(required=True, description="Full name of Person")
    age = Number(required=True, is_int=True, signed=False,
                 description="How old the person is")

# This will pass just fine
validate({"name": "John", "age": 34}, Person)

# And this will raise an exception
try:
    validate({"wrongKey": "John"}, Person)
except ValidationValueError as exc:
    print(exc)

# A newline + jsonschema
from pprint import pprint
print()
print(pprint(jsonschema(Person)))
```

Output:

```
[:Person(name(required,String()), age(required,Number(unsigned,int)))] - unrecognised key (wrongKey)

{'additionalProperties': False,
 'description': 'A Person instance',
 'properties': {'age': {'description': 'How old the person is',
                        'minimum': 0,
                        'title': 'age',
                        'type': 'integer'},
                'name': {'description': 'Full name of Person',
                         'title': 'name',
                         'type': 'string'}},
 'required': ['name', 'age'],
 'title': 'Person',
 'type': 'object'}
```

Project Goals:

    * Zero-copy JSON sanity checker 
    * Strong error reporting including path/expected type etc.
    * Context depdendent sanity checking via collection validators
    * API to generate doolally classes at Runtime
    * Generate json-schema documents

Project Non-Goals:

    * Python Object (de)serializer
    * Allow for collection modification

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