Pydantic is a Python library for **data validation and settings management** using Python type hints. It ensures data conforms to expected types/shapes at runtime.

---

### Without Pydantic (no validation)

```python
# Plain dict — no validation, anything goes
user = {"name": "Alice", "age": "not_a_number"}  # no error!
print(user["age"] + 1)  # 💥 crashes later, hard to debug
```

---

### With Pydantic

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

# ✅ Valid
user = User(name="Alice", age=30, email="alice@example.com")
print(user.name)   # Alice
print(user.age)    # 30

# ❌ Invalid — raises ValidationError immediately
user = User(name="Alice", age="not_a_number", email="alice@example.com")
# pydantic_core.ValidationError: age: Input should be a valid integer
```

---

### Key Features

#### 1. **Auto type coercion**
```python
class Item(BaseModel):
    price: float
    quantity: int

item = Item(price="9.99", quantity="3")  # strings auto-converted
print(item.price)     # 9.99  (float)
print(item.quantity)  # 3     (int)
```

#### 2. **Default values**
```python
class Config(BaseModel):
    host: str = "localhost"
    port: int = 8080
    debug: bool = False

cfg = Config()           # uses all defaults
cfg2 = Config(port=9090) # override one
```

#### 3. **Optional fields**
```python
from typing import Optional

class Profile(BaseModel):
    username: str
    bio: Optional[str] = None  # not required

p = Profile(username="alice")
print(p.bio)  # None
```
**Optional[str] = None Under the Hood**

Optional[str] is just Python's typing shorthand:

```python
Optional[str]  ==  Union[str, None]
Pydantic sees this and internally builds a field definition equivalent to:
pythonfrom pydantic.fields import FieldInfo

FieldInfo(
    annotation=Union[str, None],
    default=None,
    is_required=False
)
```
So three things happen:

- Schema: field is NOT added to "required" list in JSON Schema
- Validation: accepts str or None, rejects anything else
- Default: if not provided, field is set to None automatically

```python
class Profile(BaseModel):
    username: str
    bio: Optional[str] = None

# What Pydantic generates internally:
Profile.model_json_schema()
# {
#   "properties": {
#     "username": {"type": "string"},
#     "bio": {"anyOf": [{"type": "string"}, {"type": "null"}]}  ← Union[str, None]
#   },
#   "required": ["username"]   ← bio is NOT here
# }

Profile(username="alice")           # ✅ bio=None
Profile(username="alice", bio="hi") # ✅ bio="hi"
Profile(username="alice", bio=123)  # ❌ ValidationError
```

#### 4. **Nested models**
```python
class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Person(BaseModel):
    name: str
    age: int
    address: Address   # nested model

person = Person(
    name="Alice",
    age=30,
    address={"street": "123 Main St", "city": "NYC", "zip_code": "10001"}
)
print(person.address.city)  # NYC
```

#### 5. **Field validators**
```python
from pydantic import BaseModel, field_validator
from decimal import Decimal

class Product(BaseModel):
    name: str
    price: float

    @field_validator("price")
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Price must be positive")
        return v

    @field_validator("price")
    def price_max_two_decimals(cls, v):
        # Convert to string and check decimal places
        decimal_val = Decimal(str(v))
        if decimal_val.as_tuple().exponent < -2:
            raise ValueError("Price cannot have more than 2 decimal places")
        return v

    @field_validator("price")
    def price_must_not_exceed_limit(cls, v):
        if v > 1000.00:
            raise ValueError("Price cannot exceed 1000.00")
        return v
```

**Another approach: One validator, multiple rules**
```py
from pydantic import BaseModel, field_validator
from decimal import Decimal

class Product(BaseModel):
    name: str
    price: float

    @field_validator("price")
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError("Price must be positive")

        if Decimal(str(v)).as_tuple().exponent < -2:
            raise ValueError("Price cannot have more than 2 decimal places")

        if v > 1000.00:
            raise ValueError("Price cannot exceed 1000.00")

        return v  # always return the value!

# Tests
Product(name="A", price=-1)       # ❌ Price must be positive
Product(name="A", price=9.999)    # ❌ more than 2 decimal places
Product(name="A", price=1500.00)  # ❌ Price cannot exceed 1000.00
Product(name="A", price=9.99)     # ✅
```

**Bonus — @model_validator for cross-field validation**
```py
from pydantic import BaseModel, model_validator

class Order(BaseModel):
    price: float
    discount: float

    @model_validator(mode="after")
    def discount_less_than_price(self):
        if self.discount >= self.price:
            raise ValueError("Discount must be less than price")
        return self

Order(price=10.00, discount=15.00)  # ❌ Discount must be less than price
Order(price=10.00, discount=2.00)   # ✅
```

#### 6. **Serialize to/from dict & JSON**
```python
user = User(name="Alice", age=30, email="alice@example.com")

# To dict
print(user.model_dump())
# {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}

# To JSON string
print(user.model_dump_json())
# '{"name":"Alice","age":30,"email":"alice@example.com"}'

# From dict
data = {"name": "Bob", "age": 25, "email": "bob@example.com"}
user2 = User.model_validate(data)
```

---

### Common Use Cases

#### API request/response validation (FastAPI)
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CreateUserRequest(BaseModel):
    name: str
    age: int

@app.post("/users")
def create_user(body: CreateUserRequest):  # auto-validated from JSON body
    return {"message": f"Created user {body.name}"}
```

#### Structured LLM outputs
```python
from pydantic import BaseModel
from openai import OpenAI

class WeatherResponse(BaseModel):
    city: str
    temperature: float
    condition: str

client = OpenAI()
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Weather in NYC?"}],
    response_format=WeatherResponse,  # forces structured output
)
print(response.parsed.temperature)  # guaranteed to be a float
```

---

### Pydantic v1 vs v2

| | v1 | v2 |
|---|---|---|
| Validate from dict | `.parse_obj(d)` | `.model_validate(d)` |
| To dict | `.dict()` | `.model_dump()` |
| To JSON | `.json()` | `.model_dump_json()` |
| Speed | Slower | ~5-50x faster (Rust core) |

---

### Summary

| Feature | What it does |
|---|---|
| Type validation | Rejects wrong types at creation time |
| Auto coercion | Converts compatible types automatically |
| Nested models | Validates deeply nested structures |
| Validators | Custom business logic on fields |
| Serialization | Easy conversion to dict/JSON |

Install: `pip install pydantic`

Pydantic is the backbone of **FastAPI**, **LangChain**, and most modern Python AI/API frameworks.



Great question. Let me break down exactly what happens under the hood.

---

### Does OpenAI Accept Pydantic Directly?

**No** — the OpenAI REST API has no concept of Pydantic. It only understands **JSON Schema**. The OpenAI Python SDK does the conversion automatically.

---

### What Happens Step by Step

#### Step 1: SDK converts Pydantic → JSON Schema
When you pass `response_format=WeatherResponse`, the SDK internally calls:

```python
WeatherResponse.model_json_schema()
```

Which produces:
```json
{
  "title": "WeatherResponse",
  "type": "object",
  "properties": {
    "city":        { "type": "string" },
    "temperature": { "type": "number" },
    "condition":   { "type": "string" }
  },
  "required": ["city", "temperature", "condition"]
}
```

#### Step 2: SDK sends JSON Schema to OpenAI API
The actual HTTP request sent to OpenAI looks like this:

```json
{
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "Weather in NYC?"}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "WeatherResponse",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "city":        { "type": "string" },
          "temperature": { "type": "number" },
          "condition":   { "type": "string" }
        },
        "required": ["city", "temperature", "condition"],
        "additionalProperties": false
      }
    }
  }
}
```

OpenAI's model uses **constrained decoding** — it literally cannot generate tokens that would violate the schema.

#### Step 3: OpenAI returns a JSON string
The raw API response content is just a plain string:

```json
"{\"city\": \"NYC\", \"temperature\": 22.5, \"condition\": \"Sunny\"}"
```

#### Step 4: SDK parses JSON string → Pydantic object
The `.parse()` method (vs regular `.create()`) does this extra step:

```python
import json

raw_json = response.choices[0].message.content
data = json.loads(raw_json)
parsed = WeatherResponse.model_validate(data)  # ← this is what .parse() adds
```

So `response.parsed` is a fully validated Pydantic instance.

---

### `.parse()` vs `.create()` — the key difference

```python
# .create() — you get raw JSON string, must parse yourself
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    response_format={"type": "json_schema", "json_schema": {...}}  # manual schema
)
raw = response.choices[0].message.content  # string
data = json.loads(raw)                     # dict
weather = WeatherResponse.model_validate(data)  # you do this yourself

# .parse() — SDK does all of the above automatically
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[...],
    response_format=WeatherResponse  # just pass the class
)
weather = response.parsed  # already a WeatherResponse instance ✅
```

---

### Full Picture

```
Your code                    OpenAI Python SDK              OpenAI API
─────────────────────────────────────────────────────────────────────
response_format=             → model_json_schema()      
  WeatherResponse              converts to JSON Schema
                             → sends HTTP POST with
                               "response_format": {
                                 "type": "json_schema",  → constrained
                                 "schema": {...}            decoding
                               }                           by model
                             ← receives raw JSON string ←
response.parsed              ← json.loads()
  .city                      ← model_validate()
  .temperature                 returns Pydantic instance
  .condition
```

---

### Summary

| Step | What happens |
|---|---|
| You pass `WeatherResponse` | SDK calls `.model_json_schema()` on it |
| SDK → API | Sends JSON Schema, not Pydantic |
| API → SDK | Returns plain JSON string |
| `.parse()` extra step | `json.loads()` + `model_validate()` → Pydantic object |

So Pydantic is used **twice** — once to generate the schema going in, and once to validate/deserialize the response coming out. The OpenAI API itself never sees Pydantic at all.


## Pydantic Classes are Full Python Classes

Yes — `BaseModel` is just a Python class. You get full OOP support:

```python
from pydantic import BaseModel, field_validator, computed_field
from typing import Optional

class Product(BaseModel):
    name: str
    price: float
    quantity: int
    discount: Optional[float] = None

    # ✅ Custom __init__ not needed (Pydantic generates one)
    # but you CAN run logic after init:
    def model_post_init(self, __context):
        print(f"Product created: {self.name}")

    # ✅ Regular methods
    def apply_discount(self) -> float:
        if self.discount:
            return self.price * (1 - self.discount)
        return self.price

    def is_in_stock(self) -> bool:
        return self.quantity > 0

    # ✅ Computed/derived properties (Pydantic v2)
    @computed_field
    @property
    def total_value(self) -> float:
        return self.price * self.quantity

    # ✅ Class methods
    @classmethod
    def free_item(cls, name: str) -> "Product":
        return cls(name=name, price=0.01, quantity=1)

    # ✅ String representation
    def __str__(self) -> str:
        return f"Product({self.name}, ${self.price})"

    # ✅ Comparison
    def __eq__(self, other: "Product") -> bool:
        return self.name == other.name and self.price == other.price


# Usage
p = Product(name="Widget", price=100.0, quantity=5, discount=0.1)
# Product created: Widget      ← model_post_init fired

print(p.apply_discount())   # 90.0
print(p.is_in_stock())      # True
print(p.total_value)        # 500.0  ← computed_field
print(str(p))               # Product(Widget, $100.0)

free = Product.free_item("Freebie")
print(free.price)           # 0.01
```

---

### One caveat — don't override `__init__`

```python
# ❌ Don't do this — breaks Pydantic's validation
class Bad(BaseModel):
    name: str
    def __init__(self, name):
        self.name = name  # bypasses validation!

# ✅ Use model_post_init instead for post-init logic
class Good(BaseModel):
    name: str
    def model_post_init(self, __context):
        self.name = self.name.strip().title()  # runs AFTER validation
```

---

### Summary

| Question | Answer |
|---|---|
| `Optional[str] = None` | Sugar for `Union[str, None]`, removes field from `required` in schema |
| Multiple validators | Yes — multiple decorators or one with multiple checks; use `@model_validator` for cross-field |
| Full class support | Yes — methods, classmethods, properties, `__str__`, `__eq__` all work; avoid overriding `__init__` |
