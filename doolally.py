"""
doolally is a Python JSON schema validator
"""

from collections import namedtuple
from itertools import chain
from functools import partial
from hashlib import md5


__author__ = "James Welchman"
__all__ = [
    "validate",
    "jsonschema",
    "Schema",
    "StaticTypeArray",
    "TagObject",
    "SchemaLessObject",
    "AnyCollection",
    "String",
    "Number",
    "Bool",
    "Null",
    "AnyAtomic",
    "Union",
    "union_with_null",
    "Any",
]

# These are the python primatives which we support
# see https://docs.python.org/3/library/json.html#json.JSONDecoder
JSON_PRIMATIVES = (int, float, str, bool, type(None))
JSON_COLLECTIONS = (list, dict)

# validate/jsonschema functions
#####################

def validate(json, schema):
    instance = schema()
    # Create the Context
    ctx = Context()
    ctx.push_element_field("", instance)
    tokenizer = Tokenizer(json)
    instance.validate_collection(ctx, tokenizer)


def jsonschema(schema):
    return schema().jsonschema()


# Exceptions
##############

class ValidationError(Exception):
    pass


class ValidationTypeError(ValidationError, TypeError):
    pass


class ValidationValueError(ValidationError, ValueError):
    pass


# Tokenizer
#############

# Tokens come in ident, value pairs and take one of four forms
#     1. start of a collection
#     ident=  [ or { character
#     value=  list of dict instance
#     2. element of an array
#     ident=  positive integer, the index
#     value=  value of the element
#     3. element of a dictionary
#     ident=  string, the key
#     value=  value of the element
#     4. sentinel for dictionary/array
#     ident=  ] or } character
#     value=  None

Token = namedtuple("Token", ("ident", "value"))

class Tokenizer:
    """
    Tokenizer will tokenize a python list or dictionary

    >>> t = Tokenizer(['a', 'b', 'c'])
    >>> t.next()
    Token(ident='[', value=['a', 'b', 'c'])
    >>> t.next()
    Token(ident=0, value='a')
    >>> t.next()
    Token(ident=1, value='b')
    >>> t.next()
    Token(ident=2, value='c')
    >>> t.next()
    Token(ident=']', value=None)
    >>> t.next()
    Traceback (most recent call last):
    ...
    StopIteration

    >>> data = {"hello": [1, 2], "world": ['a', 'b']}
    >>> t = Tokenizer(data)
    >>> t.next() == Token('{', value=data)
    True
    >>> t.next()
    Token(ident='hello', value=[1, 2])
    >>> t.next()
    Token(ident='[', value=[1, 2])
    >>> t.drain_collection()
    >>> t.next()
    Token(ident='world', value=['a', 'b'])
    """

    def __init__(self, json, json_path=None):
        self.json = json
        self._current = None
        self._tokenizer = self.tokenize_top()
        self.json_path = json_path or []

    def next(self):
        self._current = next(self._tokenizer)
        return self._current

    def __next__(self):
        return self.next()

    def ctx_err(self, msg, *args):
        path = "/".join(
            map(lambda x: str(x), self.json_path),
        )
        error = f"[{path}] - " + msg.format(*args)
        return ValidationTypeError(error)

    def drain_collection(self):
        if self._current.ident not in "{[":
            # Only if we just issued start of
            # collection token are we able to drain
            error = "drain_collection not called on collection"
            raise RuntimeError(error)

        # Classic stack to drain brackets
        # { and [ will push to the stack
        # } and ] will pop from the stack
        stack = [self._current.ident]
        while stack:
            token = self.next()
            if isinstance(token.ident, int):
                continue
            if token.ident in "{[":
                stack.append(token.ident)
            elif token.ident in "}]":
                stack.pop()

    def tokenize_top(self):
        if isinstance(self.json, dict):
            yield from self.tokenize_object(self.json)
        elif isinstance(self.json, list):
            yield from self.tokenize_array(self.json)
        else:
            actual_type = type(self.json).__name__
            error = "json must be list or dict, not {!s}"
            raise self.ctx_err(error, actual_type)

    def tokenize_object(self, item):
        yield Token('{', item)

        for key, value in item.items():
            # JSON dict keys must be strings
            if not isinstance(key, str):
                actual_type = type(key).__name__
                error = "json object key must be str, not {!s}"
                raise self.ctx_err(error, actual_type)

            self.json_path.append(key)
            yield from self.tokenize_typeswitch(key, value)
            self.json_path.pop()

        yield Token('}', None)

    def tokenize_array(self, item):
        yield Token('[', item)

        for index, value in enumerate(item, 0):
            self.json_path.append(index)
            yield from self.tokenize_typeswitch(index, value)
            self.json_path.pop()

        yield Token(']', None)

    def tokenize_typeswitch(self, ident, value):
        if isinstance(value, dict):
            yield Token(ident, value)
            yield from self.tokenize_object(value)
        elif isinstance(value, list):
            yield Token(ident, value)
            yield from self.tokenize_array(value)
        elif isinstance(value, JSON_PRIMATIVES):
            yield Token(ident, value)
        else:
            actual_type = type(value).__name__
            error = "invalid type in json collection {!s}"
            raise self.ctx_err(error, actual_type)


# Context
############

class Context:
    """
    Context is similar to Tokenizer, whereas as Tokenizer
    keeps track of ident/value pairs in a list, Context
    keeps track of ident/elem_field pairs so we know what
    our validation element_field is.

    >>> ctx = Context()
    >>> ctx.push_element_field("hello", String())
    >>> raise ctx.ctx_err("testing error")
    Traceback (most recent call last):
    ...
    doolally.ValidationValueError: [hello:String()] - testing error
    """

    def __init__(self, elem_fields=None):
        # elem_field is ident/elem_field pairs
        self._elem_fields = elem_fields or []

    def push_element_field(self, ident, elem_field):
        self._elem_fields.append((ident, elem_field))

    def pop_element_field(self):
        self._elem_fields.pop()

    def ctx_err(self, error, *args, exc=None):
        exc = exc or ValidationValueError
        _, elem_field = self._elem_fields[-1]
        path = "/".join(
            map(lambda x: str(x[0]), self._elem_fields),
        )
        info = elem_field.type_info()
        msg = f"[{path}:{info}] - " + error.format(*args)
        return exc(msg)


# SchemaMeta
################ 

class SchemaMeta(type):
    def __new__(cls, name, bases, attrs):
        # See if there are any fields in the
        # base classes, we do this first so the
        # child can over-ride them.
        fields = {}
        for base in bases:
            if not hasattr(base, 'doolally_fields'):
                continue

            for attr_name, elem in base.doolally_fields.items():
                fields[attr_name] = elem

        # Delete element_fields during class creation
        to_delete = []

        # Place all ElementFields in the class dictionary
        # into our fields dictionary.
        for attr_name, item in attrs.items():
            if not isinstance(item, ElementField):
                continue

            to_delete.append(attr_name)
            attr_name = to_camel_case(attr_name)
            fields[attr_name] = item

        # Delete the element fields from the class dictionary
        for elem_field_name in to_delete:
            del attrs[elem_field_name]

        attrs['doolally_fields'] = fields
        attrs['doolally_required_fields'] = set(
            k for k, v in fields.items() if v.required
        )

        # If the fields don't have a custom title
        # then use the name of the key from this schema
        for attr_name, field in fields.items():
            if not field.title:
                field.title = attr_name

        # Build the class
        return super().__new__(cls, name, bases, attrs)

    def __repr__(self):
        return repr(self())


# Element Fields
###################

class ElementField:
    def __init__(self,
                 required=False,
                 validator=None,
                 title="",
                 description=""):
        self.required = required
        self.validator = validator or no_validate
        self.title = title
        self.description = description

        # We lazily calculate this but
        # store it once it is computed
        self._jschema_dict = {}

    def validate_atomic(self, ctx, value):
        raise NotImplementedError

    def validate_collection(self, ctx, tokenizer):
        raise NotImplementedError

    def type_info(self, recurse=False):
        raise NotImplementedError

    def is_atomic(self):
        return False

    def is_collection(self):
        return False

    def is_union(self):
        return False
    
    def __repr__(self):
        return self.type_info()

    def jschema(self):
        raise NotImplementedError

    def jsonschema(self):
        if self._jschema_dict:
            return self._jschema_dict

        jschema = self.jschema()
        if self.title:
            jschema['title'] = self.title
        else:
            jschema['title'] = "title"
        if self.description:
            jschema['description'] = self.description
        else:
            jschema['description'] = self.__class__.__name__

        self._jschema_dict = jschema
        return self._jschema_dict


class AtomicElement(ElementField):
    def validate_collection(self, _ctx, _tokenizer):
        # We shouldn't ever call validate_collection
        # on an atomic element - runtime error
        error = "validate_collection called on AtomicElement"
        raise RuntimeError(error)

    def is_atomic(self):
        return True


class CollectionElement(ElementField):
    def __init__(self,
                 required=False,
                 min_length=0,
                 max_length=-1,
                 validator=None,
                 title="",
                 description=""):
        super().__init__(
            required=required,
            validator=validator,
            title=title,
            description=description,
        )
        self.min_length = min_length
        self.max_length = max_length

    def is_collection(self):
        return True

    def validate_atomic(self, _ctx, _value):
        # We shouldn't ever call validate_atomic
        # on a collection element - runtime error
        error = "validate_atomic called on CollectionElement"
        raise RuntimeError(error)

    def validate_length(self, ctx, value):
        # Check if the collection is too short
        if len(value) < self.min_length:
            cond = f"{len(value)} < {self.min_length}"
            error = "collection too short {!s}"
            raise ctx.ctx_err(error, cond)

        # Check if the collection is too long
        if self.max_length != -1 and len(value) > self.max_length:
            cond = f"{len(value)} > {self.max_length}"
            error = "collection too long {!s}"
            raise ctx.ctx_err(error, cond)

    def validate_leading_token(self, ctx, tokenizer, ident):
        token = tokenizer.next()
        # If the ident matches then theres nothing left to do
        if token.ident == ident:
            return token.value

        raise ctx.ctx_err(
            "expected {!s}, received {!s}",
            {"[": "list", "{": "dict"}[ident],
            type(token.value).__name__,
        )


class Union(ElementField):
    """
    Union may be either a CollectionField or an AtomicField

    >>> u = Union(element_fields=[String(), Number(is_int=True)])
    >>> u
    Union(String(), Number(int))
    >>> ctx = Context()
    >>> ctx.push_element_field("someKey", u)
    >>> u.validate_atomic(ctx, 1)
    >>> u.validate_atomic(ctx, "hello")

    >>> number_array = StaticTypeArray(element_field=Number())
    >>> u = Union(element_fields=[Number(), number_array])
    >>> u
    Union(Number(), StaticTypeArray(Number()))
    """
    def __init__(self,
                 required=False,
                 element_fields=None,
                 validator=None,
                 title="",
                 description=""):
        super().__init__(
            required=required,
            validator=validator,
            title=title,
            description=description,
        )
        self._atomic_fields = []
        self._collection_fields = []

        for elem_field in element_fields:
            if elem_field.is_atomic():
                self._atomic_fields.append(elem_field)
            elif elem_field.is_collection():
                self._collection_fields.append(elem_field)
            elif elem_field.is_union():
                error = "union field in top level of another union"
                raise RuntimeError(error)
            else:
                actual_type = type(elem_field).__name__
                error = f"not an element field, {actual_type}"
                raise RuntimeError(error)

    def is_union(self):
        return True

    def validate_atomic(self, ctx, value):
        for elem_field in self._atomic_fields:
            try:
                elem_field.validate_atomic(ctx, value)
            except ValidationError:
                pass
            else:
                # Found a valid elem_field
                break
        else:
            # None of the element fields match
            raise ctx.ctx_err(
                "no element_field in union passes validation"
            )

    def validate_collection(self, ctx, tokenizer):
        token = tokenizer.next()
        for elem_field in self._collection_fields:
            # Creating a new tokenizer won't cause any
            # copying of the underlying data.
            tkz = Tokenizer(token.value,
                            json_path=tokenizer.json_path[:])
            try:
                elem_field.validate_collection(ctx, tkz)
            except ValidationError:
                pass
            else:
                # We found an element_field which validates
                # this union. We need to drain the validated
                # collection from the tokenizer before continuing.
                tokenizer.drain_collection()
                break
        else:
            # None of the element fields match
            raise ctx.ctx_err(
                "no element_field in union passes validation"
            )

    def type_info(self, recurse=False):
        name = self.__class__.__name__
        info = ""
        if not recurse:
            info = ", ".join(
                # We print the complete type information
                # if we are in a top level union.
                e.type_info(recurse=False)
                for e in chain(
                    self._atomic_fields,
                    self._collection_fields,
                )
            )
        else:
            info = ".."

        return f"{name}({info})"

    def jschema(self):
        elems = []
        iterator = chain(self._atomic_fields,
                         self._collection_fields)
        for elem_field in iterator:
            elems.append(elem_field.jsonschema())

        return {"anyOf": elems}


class Number(AtomicElement):
    """
    >>> n = Number()
    >>> ctx = Context()
    >>> ctx.push_element_field("amount", n)
    >>> n.validate_atomic(ctx, 1)
    >>> n.validate_atomic(ctx, -15.4)
    >>> n
    Number()
    >>> n.jschema()
    {'type': 'number'}

    >>> n = Number(signed=False, is_int=True)
    >>> n
    Number(unsigned,int)
    >>> n.jschema()
    {'type': 'integer', 'minimum': 0}

    >>> n = Number(min_value=0, max_value=90)
    >>> n
    Number(min_value=0,max_value=90)
    >>> n.jschema()
    {'type': 'number', 'minimum': 0, 'maximum': 90}
    """
    def __init__(self,
                 required=False,
                 signed=True,
                 is_int=False,
                 min_value=None,
                 max_value=None,
                 validator=None,
                 title="",
                 description=""):
        super().__init__(
            required=required,
            validator=validator,
            title=title,
            description=description,
        )
        self.signed = signed
        self.is_int = is_int
        self.min_value = min_value
        self.max_value = max_value

    def validate_atomic(self, ctx, value):
        validate_type(ctx, value, int, float)

        # Check if the number is negative
        if not self.signed and value < 0:
            error = f"negative number invalid {value}"
            raise ctx.ctx_err(error)

        # Don't accept decimals if we want an int
        if self.is_int and int(value) != value:
            error = f"expected int, received {value}"
            raise ctx.ctx_err(error)

        # Check if the number is too small
        if (
            self.min_value is not None and
            self.min_value > value
        ):
            cond = f"{value} < {self.min_value}"
            error = "number below min {!s}"
            raise ctx.ctx_err(error, cond)

        # Check if the number is too large
        if (
            self.max_value is not None and
            self.max_value < value
        ):
            cond = f"{value} > {self.max_value}"
            error = "number above max {!s}"
            raise ctx.ctx_err(error, cond)

        # Run any custom validator
        self.validator(ctx.ctx_err, value)

    def type_info(self, recurse=False):
        name = self.__class__.__name__
        tags = []
        if not self.signed:
            tags.append("unsigned") 
        if self.is_int:
            tags.append("int")
        if self.min_value is not None:
            tags.append(f"min_value={self.min_value}")
        if self.max_value is not None:
            tags.append(f"max_value={self.max_value}")
    
        return f"{name}(" + ",".join(tags) + ")"

    def jschema(self):
        jschema = {}
        if self.is_int:
            jschema['type'] = "integer"
        else:
            jschema['type'] = "number"

        if self.min_value is not None:
            jschema['minimum'] = self.min_value
        if self.max_value is not None:
            jschema['maximum'] = self.max_value

        if 'minimum' not in jschema and not self.signed:
            jschema['minimum'] = 0

        return jschema


class String(AtomicElement):
    """
    >>> s = String()
    >>> s
    String()
    >>> s.jschema()
    {'type': 'string'}
    >>> ctx = Context()
    >>> ctx.push_element_field("key", s)
    >>> s.validate_atomic(ctx, "hello")

    >>> s = String(min_length=3, max_length=6)
    >>> s
    String(min_length=3,max_length=6)
    >>> s.jschema()
    {'type': 'string', 'minLength': 3, 'maxLength': 6}
    """
    def __init__(self,
                 required=False,
                 min_length=0,
                 max_length=-1,
                 validator=None,
                 title="",
                 description=""):
        super().__init__(
            required=required,
            validator=validator,
            title=title,
            description=description,
        )
        self.min_length = min_length
        self.max_length = max_length

    def validate_atomic(self, ctx, value):
        validate_type(ctx, value, str)

        # validate the length
        if self.min_length and len(value) < self.min_length:
            cond = f"{len(value)} < {self.min_length}"
            error = "string too short {!s}"
            raise ctx.ctx_err(error, cond)

        if self.max_length != -1 and len(value) > self.max_length:
            cond = f"{len(value)} > {self.max_length}"
            error = "string too long {!s}"
            raise ctx.ctx_err(error, cond)

        # Run any custom validator
        self.validator(ctx.ctx_err, value)

    def type_info(self, recurse=False):
        name = self.__class__.__name__
        tags = []
        if self.min_length != 0:
            tags.append(f"min_length={self.min_length}")
        if self.max_length != -1:
            tags.append(f"max_length={self.max_length}")

        return f"{name}(" + ",".join(tags) + ")"

    def jschema(self):
        jschema = {"type": "string"}
        if self.min_length != 0:
            jschema['minLength'] = self.min_length
        if self.max_length != -1:
            jschema['maxLength'] = self.max_length

        return jschema


class Bool(AtomicElement):
    """
    >>> b = Bool()
    >>> b
    Bool()
    >>> b.jschema()
    {'type': 'boolean'}
    >>> ctx = Context()
    >>> ctx.push_element_field("flag", b)
    >>> b.validate_atomic(ctx, True)
    >>> b.validate_atomic(ctx, False)
    """
    def validate_atomic(self, ctx, value):
        validate_type(ctx, value, bool)

    def type_info(self, recurse=False):
        return "Bool()"

    def jschema(self):
        return {"type": "boolean"}


class Null(AtomicElement):
    """
    >>> n = Null()
    >>> n
    Null()
    >>> n.jschema()
    {'type': 'null'}
    >>> ctx = Context()
    >>> ctx.push_element_field("flag", n)
    >>> n.validate_atomic(ctx, None)
    """
    def validate_atomic(self, ctx, value):
        validate_type(ctx, value, type(None))

    def type_info(self, recurse=False):
        return "Null()"

    def jschema(self):
        return {"type": "null"}
    

def union_with_null(required=False,
                    element_fields=None,
                    validator=None):
    """
    >>> s = String()
    >>> u = union_with_null(element_fields=[s])
    >>> u
    Union(String(), Null())
    >>> types = [{'type': "string"}, {"type": "null"}]
    >>> types[0]['title'] = "title"
    >>> types[0]['description'] = "String"
    >>> types[1]['title'] = "title"
    >>> types[1]['description'] = "Null"
    >>> u.jschema() == {"anyOf": types}
    True
    """
    element_fields = element_fields or []
    return Union(
        required=required,
        element_fields=element_fields + [Null()],
        validator=validator,
    )


class AnyAtomic(AtomicElement):
    """
    >>> a = AnyAtomic()
    >>> a
    AnyAtomic()
    >>> a.jschema()
    {'type': ['number', 'integer', 'string', 'boolean', 'null']}
    """
    def __init__(self,
                 required=False,
                 validator=None,
                 title="",
                 description=""):
        super().__init__(
            required=required,
            validator=validator,
            title=title,
            description=description,
        )

    def validate_atomic(self, ctx, value):
        validate_type(ctx, value, *JSON_PRIMATIVES)

    def type_info(self, recurse=False):
        return f"{self.__class__.__name__}()"

    def jschema(self):
        return {
            "type": [
                "number",
                "integer",
                "string",
                "boolean",
                "null",
            ],
        }


class ArrayCollection(CollectionElement):
    def validate_leading_token(self, ctx, tokenizer):
        return super().validate_leading_token(
            ctx,
            tokenizer,
            ident='[',
        )


class ObjectCollection(CollectionElement):
    def validate_leading_token(self, ctx, tokenizer):
        return super().validate_leading_token(
            ctx,
            tokenizer,
            ident='{',
        )


class StaticTypeArray(ArrayCollection):
    """
    >>> s = StaticTypeArray(element_field=Number())
    >>> s
    StaticTypeArray(Number())
    >>> jschema = {'type': 'array'}
    >>> jschema['items'] = {'type': 'number'}
    >>> jschema['items']['title'] = 'title'
    >>> jschema['items']['description'] = 'Number'
    >>> s.jschema() == jschema
    True
    >>> ctx = Context()
    >>> ctx.push_element_field("number", s)
    >>> t = Tokenizer([1, 2, 3])
    >>> s.validate_collection(ctx, t)
    """
    def __init__(self,
                 required=True,
                 min_length=0,
                 max_length=-1,
                 element_field=None,
                 validator=None,
                 title="",
                 description=""):
        super().__init__(
            required=required,
            validator=validator,
            title=title,
            description=description,
        )
        self.min_length = min_length
        self.max_length = max_length
        self.element_field = element_field

    def validate_collection(self, ctx, tokenizer):
        collection = self.validate_leading_token(ctx, tokenizer)
        self.validate_length(ctx, collection)

        # Iterate over the array
        elem_field = self.element_field
        while True:
            # Break when we leave the list
            token = tokenizer.next()
            if token.ident == ']':
                break

            # If the element field is a union then switch
            # base on the value rather than the element field
            switch_type = None
            if elem_field.is_union():
                if isinstance(token.value, JSON_PRIMATIVES):
                    switch_type = "atomic"
                elif isinstance(token.value, JSON_COLLECTIONS):
                    switch_type = "collection"
            elif elem_field.is_collection():
                switch_type = "collection"
            elif elem_field.is_atomic():
                switch_type = "atomic"
            else:
                error = "not union, collection or atomic"
                raise RuntimeError(error)

            ctx.push_element_field(token.ident, elem_field)
            try:
                if switch_type == "atomic":
                    elem_field.validate_atomic(ctx, token.value)
                else:
                    elem_field.validate_collection(ctx, tokenizer)
            finally:
                # This must go into a finally block
                # This is because a union might try
                # calling this method and handling
                # the exception gracefully.
                ctx.pop_element_field()

        # Run any custom validator
        self.validator(ctx.ctx_err, collection)

    def type_info(self, recurse=False):
        name = self.__class__.__name__
        if recurse:
            return f"{name}(..)"

        tags = []
        if self.min_length != 0:
            tags.append(f"min_length={self.min_length}")
        if self.max_length != -1:
            tags.append(f"max_length={self.max_length}")

        elem_info = self.element_field.type_info(recurse=True)
        tags.append(elem_info)
        return f"{name}(" + ",".join(tags) + ")"

    def jschema(self):
        jschema = {"type": "array"}
        jschema['items'] = self.element_field.jsonschema()
        return jschema


class TagObject(ObjectCollection):
    """
    TagObject is a JSON object where the values
    must all be strings. The usual min_length and
    max_length params for collections are avaliable.

    >>> t = TagObject()
    >>> t
    TagObject()
    >>> jschema = {"type": "object", "properties": {}}
    >>> jschema["additionalProperties"] = {"type": "string"}
    >>> t.jschema() == jschema
    True

    >>> t = TagObject(max_length=10)
    >>> t
    TagObject(max_length=10)
    >>> jschema = {"type": "object", "properties": {}}
    >>> jschema["additionalProperties"] = {"type": "string"}
    >>> jschema['maxLength'] = 10
    >>> t.jschema() == jschema
    True
    >>> ctx = Context()
    >>> ctx.push_element_field("tags", t)
    >>> tk = Tokenizer({"hello": "yes", "world": "no"})
    >>> t.validate_collection(ctx, tk)
    """
    def validate_collection(self, ctx, tokenizer):
        collection = self.validate_leading_token(ctx, tokenizer)
        self.validate_length(ctx, collection)

        while True:
            token = tokenizer.next()
            # Break when we leave the object
            if token.ident == '}':
                break

            ctx.push_element_field(token.ident, String())
            try:
                if not isinstance(token.value, str):
                    actual_type = type(token.value).__name__
                    error = "TagObject values must be str not {!s}"
                    raise ctx.ctx_err(error, actual_type)
            finally:
                ctx.pop_element_field()

        # Run any custom validator
        self.validator(ctx.ctx_err, collection)

    def type_info(self, recurse=False):
        name = self.__class__.__name__
        if recurse:
            return f"{name}(..)"

        tags = []
        if self.min_length != 0:
            tags.append(f"min_length={self.min_length}")
        if self.max_length != -1:
            tags.append(f"max_length={self.max_length}")

        return f"{name}(" + ",".join(tags) + ")"

    def jschema(self):
        jschema = {"type": "object"}
        jschema['properties'] = {}
        jschema["additionalProperties"] = {"type": "string"}
        if self.min_length != 0:
            jschema['minLength'] = self.min_length
        if self.max_length != -1:
            jschema['maxLength'] = self.max_length

        return jschema


class StaticTypeObject(ObjectCollection):
    """
    StaticTypeObject puts no constraint on the keys
    used in the object, however all values have a static
    type.

    >>> s = StaticTypeObject(element_field=Number())
    >>> s
    StaticTypeObject(Number())
    >>> jschema = {"type": "object", "properties": {}}
    >>> jschema["additionalProperties"] = {"type": "number"}
    >>> jschema["additionalProperties"]['title'] = 'title'
    >>> jschema["additionalProperties"]['description'] = 'Number'
    >>> s.jschema() == jschema
    True
    >>> ctx = Context()
    >>> ctx.push_element_field("data", s)
    >>> t = Tokenizer({"hello": 1, "world": 2})
    >>> s.validate_collection(ctx, t)
    """

    def __init__(self,
                 required=False,
                 min_length=0,
                 max_length=-1,
                 element_field=None,
                 unique_items=False,
                 validator=None,
                 title="",
                 description=""):
        super().__init__(
            required=required,
            min_length=min_length,
            max_length=max_length,
            validator=validator,
            title=title,
            description=description,
        )
        self.element_field = element_field
        if unique_items:
            # We only allow enforce-uniqueness on AtomicFields
            if not element_field.is_atomic():
                error = "uniqueness on non AtomicElement"
                raise RuntimeError(error)

        self.unique_items = unique_items

    def validate_collection(self, ctx, tokenizer):
        collection = self.validate_leading_token(ctx, tokenizer)
        self.validate_length(ctx, collection)

        seen_values = set()
        while True:
            token = tokenizer.next()
            # Break when we leave the object
            if token.ident == '}':
                break

            # Decide which function to call
            elem_field = self.element_field
            switch_type = None
            if elem_field.is_union():
                if isinstance(token.value, JSON_PRIMATIVES):
                    switch_type = "atom"
                elif isinstance(token.value, JSON_COLLECTIONS):
                    switch_type = "collection"
                else:
                    # Something weird has come from the tokenizer
                    error = f"invalid token value {token.value}"
                    raise RuntimeError(error)
            elif elem_field.is_atomic():
                switch_type = "atom"
            elif elem_field.is_collection():
                switch_type = "collection"
            else:
                # A programming error
                error = f"not an element field {elem_field}"
                raise RuntimeError(error)

            ctx.push_element_field(token.ident, elem_field)
            try:
                # Deal with the case of unique values
                seen = token.value in seen_values
                if self.unique_items and seen:
                    # We've seen this value before
                    error = "duplicate value in array {!s}"
                    ctx.ctx_err(error, token.value)
                elif self.unique_items and not seen:
                    seen.add(token.value)

                if switch_type == "atom":
                    elem_field.validate_atomic(ctx, token.value)
                else:
                    elem_field.validate_collection(ctx, tokenizer)

            finally:
                ctx.pop_element_field()

    def type_info(self, recurse=False):
        name = self.__class__.__name__
        if recurse:
            return f"{name}(..)"

        tags = []
        if self.min_length != 0:
            tags.append(f"min_length={self.min_length}")
        if self.max_length != -1:
            tags.append(f"max_length={self.max_length}")
        if self.unique_items:
            tags.append("unique")

        elem_info = self.element_field.type_info(recurse=True)
        tags.append(elem_info)
        return f"{name}(" + ",".join(tags) + ")"

    def jschema(self):
        jschema = {"type": "object", "properties": {}}
        elem_schema = self.element_field.jsonschema()
        jschema['additionalProperties'] = elem_schema
        if self.min_length != 0:
            jschema['minLength'] = self.min_length
        if self.max_length != -1:
            jschema['maxLength'] = self.max_length
        if self.unique_items:
            jschema['uniqueItems'] = True

        return jschema


class SchemaLessObject(StaticTypeObject):
    """
    >>> s = SchemaLessObject()
    >>> s
    SchemaLessObject()
    >>> jschema = {'type': 'object', 'properties': {}}
    >>> jschema['additionalProperties'] = Any().jsonschema()
    >>> s.jschema() == jschema
    True

    >>> s = SchemaLessObject(min_length=2)
    >>> ctx = Context()
    >>> ctx.push_element_field("tags", s)
    >>> t = Tokenizer({"hello": 1, "world": 2})
    >>> s.validate_collection(ctx, t)
    >>> t = Tokenizer({"boo": "a", "goose": "b", "hen": "c"})
    >>> s.validate_collection(ctx, t)
    """
    def __init__(self,
                 required=False,
                 min_length=0,
                 max_length=-1,
                 validator=None,
                 title="",
                 description=""):
        super().__init__(
            required=required,
            min_length=min_length,
            max_length=max_length,
            element_field=Any(),
            validator=validator,
            title=title,
            description=description,
        )

    def type_info(self, recurse=False):
        name = self.__class__.__name__
        if recurse:
            return f"{name}(..)"

        tags = []
        if self.min_length != 0:
            tags.append(f"min_length={self.min_length}")
        if self.max_length != -1:
            tags.append(f"max_length={self.max_length}")

        return f"{name}(" + ",".join(tags) + ")"


class Schema(ObjectCollection, metaclass=SchemaMeta):
    jsonschema_description = ""

    def __init__(self,
                 required=False,
                 min_length=0,
                 max_length=-1,
                 validator=None):
        super().__init__(
            required=required,
            min_length=min_length,
            max_length=max_length,
            validator=validator,
            title=self.__class__.__name__,
            description=(
                self.__class__.jsonschema_description or
                self.__class__.__name__
            ),
        )

    def validate_collection(self, ctx, tokenizer):
        collection = self.validate_leading_token(ctx, tokenizer)
        self.validate_length(ctx, collection)

        seen_fields = set()
        while True:
            token = tokenizer.next()
            # Break when we leave the object
            if token.ident == '}':
                break

            # Get the element field
            elem_field = self.doolally_fields.get(token.ident)
            if not elem_field:
                # This key is not used in the schema
                error = "unrecognised key ({!s})"
                raise ctx.ctx_err(error, token.ident)
            else:
                # We've this key if it's required
                seen_fields.add(token.ident)

            switch_type = None
            if elem_field.is_union():
                # Use the token value to decide atom/collection
                # if the element_type is a union.
                if isinstance(token.value, JSON_PRIMATIVES):
                    switch_type = "atom"
                elif isinstance(token.value, JSON_COLLECTIONS):
                    switch_type = "collection"
                else:
                    # Something weird has come from the tokenizer
                    error = f"invalid token value {token.value}"
                    raise RuntimeError(error)
            elif elem_field.is_atomic():
                switch_type = "atom"
            elif elem_field.is_collection():
                switch_type = "collection"
            else:
                # A programming error
                error = f"not an element field {elem_field}"
                raise RuntimeError(error)

            ctx.push_element_field(token.ident, elem_field)
            try:
                if switch_type == "atom":
                    elem_field.validate_atomic(ctx, token.value)
                else:
                    elem_field.validate_collection(ctx, tokenizer)
            finally:
                ctx.pop_element_field()

        # Check if we have any missing required fields
        for name in self.doolally_required_fields:
            if name not in seen_fields:
                error = "missing required field {!s}"
                raise ctx.ctx_err(error, name)

        # Run any custom validator
        self.validator(ctx.ctx_err, collection)

    def type_info(self, recurse=False):
        name = self.__class__.__name__
        if recurse:
            return f"{name}(..)"

        if len(self.doolally_fields) <= 3:
            # Less than three fields - get them all
            fields = []
            iterator = self.doolally_fields.items()
            for attr_name, elem_field in iterator:
                field = attr_name + '('
                tags = []
                if attr_name in self.doolally_required_fields:
                    field += "required,"
                else:
                    field += "optional,"
                field += elem_field.type_info(recurse=True)
                field += ')'
                fields.append(field)

            return f"{name}(" + ", ".join(fields) + ")"

        num_required = len(self.doolally_required_fields)
        num_optional = len(self.fields) - num_required
        info = f"num_required={num_required},"
        info += f"num_optional={num_optional}"
        return f"{name}({info})"

    def jschema(self):
        jschema = {"type": "object"}
        required = []
        properties = {}
        jschema['required'] = required
        jschema['properties'] = properties
        jschema['additionalProperties'] = False

        for key, elem_field in self.doolally_fields.items():
            if elem_field.required:
                required.append(key)
            properties[key] = elem_field.jsonschema()

        return jschema


def schema_factory(name, fields=None, bases=None):
    """
    >>> fields = {"name": String(required=True), "age": Number()}
    >>> my_schema = schema_factory("MySchema", fields)
    >>> my_schema
    MySchema(name(required,String()), age(optional,Number()))
    """
    bases = bases or (Schema,)
    fields = fields or {}
    return SchemaMeta(name, bases, fields)


class AnyCollection(CollectionElement):
    """
    >>> a = AnyCollection()
    >>> a
    AnyCollection()
    >>> a.jschema()
    {'type': ['array', 'object']}
    >>> ctx = Context()
    >>> ctx.push_element_field(2, a)
    >>> t = Tokenizer({"hello": 1})
    >>> a.validate_collection(ctx, t)
    >>> t = Tokenizer([1, 2, 3])
    >>> a.validate_collection(ctx, t)
    """
    def validate_collection(self, ctx, tokenizer):
        # We DO care that a collection comes
        # from the tokenizer, beyond that it's not
        # important what comes out.
        token = tokenizer.next()
        if token.ident not in "{[":
            actual_type = type(token.value).__name__
            error = "expected a collection, received {!s}"
            raise ctx.ctx_err(error, actual_type)

        # drain this collection from the tokenizer
        tokenizer.drain_collection()

    def type_info(self, recurse=False):
        name = self.__class__.__name__

        tags = []
        if self.min_length != 0:
            tags.append(f"min_length={self.min_length}")
        if self.max_length != -1:
            tags.append(f"max_length={self.max_length}")

        return f"{name}(" + ",".join(tags) + ")"

    def jschema(self):
        jschema = {"type": ["array", "object"]}
        if self.min_length != 0:
            jschema['minLength'] = self.min_length
        if self.max_length != -1:
            jschema['maxLength'] = self.max_length

        return jschema


class Any(Union):
    """
    >>> a = Any()
    >>> a
    Any()
    >>> ctx = Context()
    >>> ctx.push_element_field("data", a)
    >>> a.validate_atomic(ctx, 1)
    >>> a.validate_atomic(ctx, "hello")
    >>> t = Tokenizer(["a", "b", "c"])
    >>> a.validate_collection(ctx, t)
    """
    def __init__(self,
                 required=False,
                 validator=None,
                 title="",
                 description=""):
        super().__init__(
            required=required,
            element_fields=[AnyAtomic(), AnyCollection()],
            validator=validator,
            title=title,
            description=description,
        )

    def type_info(self, recurse=False):
        name = self.__class__.__name__
        return f"{name}()"

    def jschema(self):
        return {
            'type': [
                "number",
                "integer",
                "string",
                "boolean",
                "null",
                "object",
                "array",
            ],
        }


# Misc functions
###################

def no_validate(ctx_err, value):
    pass


def validate_type(ctx, value, *types):
    if isinstance(value, types):
        return

    valid_types = ",".join(map(lambda x: x.__name__, types))
    error = "expected type in ({!s})"
    raise ctx.ctx_err(
        error,
        valid_types,
        exc=ValidationTypeError,
    )

def to_camel_case(string):
    """
    >>> to_camel_case("hello")
    'hello'
    >>> to_camel_case("hello_world")
    'helloWorld'
    >>> to_camel_case("hello_world_yes")
    'helloWorldYes'
    """
    parts = string.split('_')
    ret = parts[0]
    for p in parts[1:]:
        ret += p.title()
    return ret


def call_many(*callables):
    """
    call_many will execute a sequence of callables
    when it it _itself_ called. It returns a tuple
    of the calculated values. It is not consumable
    and the returned callable may be used multiple
    times.

    >>> a = call_many(lambda: 1, lambda: 2)
    >>> a()
    (1, 2)
    >>> a()
    (1, 2)

    >>> b = call_many(lambda x, y: x + y, lambda x, y: x * y)
    >>> b(5, 10)
    (15, 50)
    """
    def execute(*args):
        return tuple(c(*args) for c in callables)
    return execute


def validate_whitelist(whitelist, ctx_err, value):
    """
    validate_whitelist checks if a value is in a given
    whitelist.

    >>> def ctx_err(error, *args):
    ...    return ValidationValueError(error.format(*args))
    >>> validate_whitelist({1, 2, 3}, ctx_err, 3)
    >>> validate_whitelist({1, 2, 3}, ctx_err, 1)
    >>> validate_whitelist({"ab", "cd"}, ctx_err, "ab")
    >>> validate_whitelist({"ab", "cd"}, ctx_err, "xyz")
    Traceback (most recent call last):
    ...
    doolally.ValidationValueError: value (xyz) not in whitelist
    """
    if value not in whitelist:
        error = "value ({!s}) not in whitelist"
        raise ctx_err(error, value)


# Enums
############

class IntegerEnum(Number):
    """
    IntegerEnum is an integer which must come from a whitelist
    of values.

    >>> i = IntegerEnum(whitelist={1, 2, 3})
    >>> i
    IntegerEnum(int)
    >>> i.jschema()
    {'type': 'integer', 'enum': [1, 2, 3]}
    >>> ctx = Context()
    >>> ctx.push_element_field("intValue", i)
    >>> i.validate_atomic(ctx, 1)
    >>> i.validate_atomic(ctx, 3)
    """
    def __init__(self,
                 required=False,
                 signed=True,
                 validator=None,
                 title="",
                 description="",
                 whitelist=None):

        self.whitelist = set(whitelist or [])

        validator = call_many(
            partial(validate_whitelist, self.whitelist),
            validator or no_validate,
        )
        super().__init__(
            required=required,
            signed=signed,
            is_int=True,
            validator=validator,
            title=title,
            description=description,
        )

    def jschema(self):
        jschema = {'type': "integer"}
        if len(self.whitelist) < 10:
            jschema['enum'] = sorted(self.whitelist)
        else:
            jschema['enum'] = list(self.whitelist[:5] + ['..'])

        return jschema


class StringEnum(String):
    """
    StringEnum is a string which must come from a whitelist
    of values.

    >>> s = StringEnum(whitelist={"ab", "cde", "z"})
    >>> s
    StringEnum()
    >>> s.jschema()
    {'type': 'string', 'enum': ['ab', 'cde', 'z']}
    >>> ctx = Context()
    >>> ctx.push_element_field("stringValue", s)
    >>> s.validate_atomic(ctx, "ab")
    >>> s.validate_atomic(ctx, "z")
    """
    def __init__(self,
                 required=False,
                 validator=None,
                 title="",
                 description="",
                 whitelist=None):

        self.whitelist = set(whitelist or [])

        validator = call_many(
            partial(validate_whitelist, self.whitelist),
            validator or no_validate,
        )
        super().__init__(
            required=required,
            validator=validator,
            title=title,
            description=description,
        )

    def jschema(self):
        jschema = {'type': "string"}
        if len(self.whitelist) < 5:
            jschema['enum'] = sorted(self.whitelist)
        else:
            jschema['enum'] = list(self.whitelist[:5]) + ['..']

        return jschema