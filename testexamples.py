"""
testexamples is the file containing both project examples
and the data for the test suite.
"""

from doolally import (
    Schema,
    StaticTypeArray,
    StaticTypeObject,
    Union,
    Number,
    String,
    Bool,
    StringEnum,
    IntegerEnum,
    union_with_null,
    AnyAtomic,
    AnyCollection,
    Any,
)


## Animals
CAT_BREEDS = ["Sphynx", "Abyssinian", "Burmese"]
DOG_BREEDS = ["Labrador", "German Shepherd", "Boxer Terrier"]


class Animal(Schema):
    name = String(required=True,
                  description="The name of the animal")
    age = Number(required=True, is_int=True, signed=False,
                 description="The age of the animal in human years")


class Cat(Animal):
    doolally_description = "Cat represents a common housecat"

    breed = StringEnum(required=True,
                       whitelist=CAT_BREEDS,
                       description="Breed of the cat")


class Dog(Animal):
    doolally_description = "Dog is a pet dog"

    breed = StringEnum(required=True,
                       whitelist=DOG_BREEDS,
                       description="Breed of the dog")


class AnimalPerson(Schema):
    # A person optionally may have a pet
    # they can have a dog or a cat, but not both
    pet = Union(required=False,
                element_fields=[Dog(), Cat()])


class EmailAddr(Schema):
    # Which provider is the email?
    # This is either a string, or null if not known
    provider = union_with_null(required=True,
                               element_fields=[String()])


class Numbers(Schema):
    NUMBERS = [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        *map(lambda x: str(x), range(1, 11)),
    ]
    numbers = StaticTypeArray(required=False,
                              element_field=Union(
                                  element_fields=[
                                      StringEnum(whitelist=NUMBERS),
                                      Number(),
                               ],
                           ))

    latitude = Number(required=False,
                      min_value=0,
                      max_value=90,
                      is_int=False,
                      description="latitude on the earth")