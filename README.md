# python-help

## Basic Types:
```py
# Integer (whole numbers) ────────────────────────────────────────────────
age = 25                        # most common way
age: int = 25                   # explicit type hint (recommended in serious code)

# Float (decimal numbers) ────────────────────────────────────────────────
price = 19.99
price: float = 19.99

height: float = 175.3           # height in cm

# Boolean (True / False) ─────────────────────────────────────────────────
is_adult = True
is_adult: bool = True

has_subscription: bool = False

# String (text) ──────────────────────────────────────────────────────────
name = "VK"
name: str = "VK"

greeting: str = 'Hello, @stocks4v!'
multiline: str = """This is a
multi-line string"""

# None (absence of value) ────────────────────────────────────────────────
result = None
result: None = None             # rare to annotate None explicitly
```

## Collection / Container Types

```py
# List ───────────────────────────────────────────────────────────────────
scores: list[int] = [85, 92, 78, 95]
names: list[str] = ["Alice", "Bob", "Charlie"]
mixed: list = [1, "hello", 3.14, True]           # not recommended (avoid mixed types)

# Tuple (immutable list) ─────────────────────────────────────────────────
point: tuple[float, float] = (3.14, 2.718)
date: tuple[int, int, int] = (2026, 1, 21)
rgb: tuple[int, int, int] = (255, 128, 0)

# Dictionary (key-value pairs) ───────────────────────────────────────────
person: dict[str, any] = {
    "name": "VK",
    "age": 30,
    "city": "Jersey City",
    "active": True
}

portfolio: dict[str, float] = {
    "AAPL": 245.67,
    "TSLA": 412.30,
    "VOO": 528.15
}

# Set (unique unordered items) ───────────────────────────────────────────
tags: set[str] = {"python", "finance", "investing", "stocks"}
numbers: set[int] = {1, 2, 3, 3, 2, 1}          # duplicates removed → {1,2,3}
```

## Specialized or MOdern Types (Python 3.9+)
```py
# Built-in generic types (cleaner syntax since Python 3.9)
from typing import List, Dict, Tuple, Set, Any     # older style (still common)

numbers_old: List[int] = [1, 2, 3]                 # old way
numbers_new: list[int] = [1, 2, 3]                 # preferred now

users: dict[str, int] = {"alice": 25, "bob": 31}

# Union (can be one of several types) ────────────────────────────────────
from typing import Union

result: Union[int, float, None] = 42
result = 3.14
result = None

# Modern | syntax (Python 3.10+)
result: int | float | None = 100
result = 99.99
result = None

# Optional (shorthand for X | None)
from typing import Optional

username: Optional[str] = "vkcode7"  # ✓ Valid (string)
username = None                       # ✓ Valid (None)
# username = 123                      # ✗ Invalid (not str or None)

# Optional[T] is just shorthand for Union[T, None]
# These are EXACTLY the same:
name1: Optional[str] = None
name2: Union[str, None] = None

# In Python 3.10+, these are also the same:
name3: str | None = None

# Literal (very specific allowed values) ─────────────────────────────────
from typing import Literal

mode: Literal["buy", "sell", "hold"] = "buy"
# mode = "cancel"                   # type checker will complain
```

## Operations
```py
    # Division.
    # Result of division is float number.
    assert 5 / 3 == 1.6666666666666667
    assert 8 / 4 == 2
    assert isinstance(5 / 3, float)
    assert isinstance(8 / 4, float)

    # Modulus.
    assert 5 % 3 == 2

    # Exponentiation.
    assert 5 ** 3 == 125
    assert 2 ** 3 == 8
    assert 2 ** 4 == 16
    assert 2 ** 5 == 32
    assert isinstance(5 ** 3, int)

    # Floor division.
    assert 5 // 3 == 1
    assert 6 // 3 == 2
    assert 7 // 3 == 2
    assert 9 // 3 == 3
    assert isinstance(5 // 3, int)
```

## Operators
```text
Arithmetic Operators (+, -, *, /, //, %, **)
Bitwise Operators (&, |, ^, >>, <<, ~)
Assignment Operators (=, +=, -=, /=, //= etc.)
Comparison Operator (==, !=, >, <, >=, <=)
Logical Operators (and, or, not)
Identity Operators (is, is not)
Membership Operators (in, not in)
```

**Python Does NOT Have ++ or -- Operators**
```python
# ❌ THESE DON'T WORK IN PYTHON
# x++   # SyntaxError
# ++x   # Not what you think! ++x is interpreted as +(+x); Which is just: positive of positive x = x
# x--   # SyntaxError
# --x   # Not what you think! --x is seen as: -(-x) = -(-5) = 5

# ✅ USE THESE INSTEAD
x = 5
x += 1  # Increment by 1
x -= 1  # Decrement by 1
```

## for loop
```py
# Example: Demonstrating for loops with str, list, dict, tuple, set
# All in one script with clear enumeration

print("=== 1. For loop over a STRING (character by character) ===")
text = "STOCKS"

for i, char in enumerate(text, start=1):
    print(f"  {i}. Character: {char}")

print("\n=== 2. For loop over a LIST (with index) ===")
tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

for i, ticker in enumerate(tickers, start=1):
    print(f"  {i}. Ticker: {ticker}")

print("\n=== 3. For loop over a DICTIONARY (keys and values) ===")
portfolio = {
    "AAPL": 245.67,
    "MSFT": 418.40,
    "GOOGL": 182.15,
    "TSLA": 412.30,
    "NVDA": 135.60
}

for i, (ticker, price) in enumerate(portfolio.items(), start=1):
    print(f"  {i}. {ticker}: ${price:.2f}")

print("\n=== 4. For loop over a TUPLE (unpacking values) ===")
holding = ("VOO", 528.15, 150, "2026-01-21")  # ticker, price, shares, purchase_date

for i, value in enumerate(holding, start=1):
    if i == 1:
        print(f"  {i}. Ticker: {value}")
    elif i == 2:
        print(f"  {i}. Price: ${value:.2f}")
    elif i == 3:
        print(f"  {i}. Shares: {value}")
    else:
        print(f"  {i}. Date: {value}")

print("\n=== 5. For loop over a SET (unique items, order not guaranteed) ===")
sectors = {"Tech", "Finance", "Energy", "Tech", "Utilities", "Healthcare"}

for i, sector in enumerate(sorted(sectors), start=1):  # sorted just for consistent output
    print(f"  {i}. Sector: {sector}")
```

dict:
```py
portfolio = {
    "AAPL": 245.67,
    "TSLA": 412.30,
    "VOO": 528.15,
    "BND": 72.40
}

# Option A: Loop over keys only (default)
for ticker in portfolio:
    print(ticker)

# Option B: Loop over keys and values (most useful)
for ticker, price in portfolio.items():
    print(f"{ticker}: ${price:.2f}")

# Option C: Just values
for price in portfolio.values():
    print(f"Price: ${price}")
```

tuple:
```py
holding = ("VOO", 528.15, 150)  # (ticker, price, shares)

for item in holding:
    print(item)

transactions = [
    ("AAPL", 245.67, 20),
    ("TSLA", 412.30, 5),
    ("BND", 72.40, 100)
]

for ticker, price, shares in transactions:
    total = price * shares
    print(f"{ticker}: {shares} shares @ ${price:.2f} → Total: ${total:.2f}")
```

## variadic params
Function can be called with an arbitrary number of arguments. These arguments will be wrapped up in
a tuple. Before the variable number of arguments, zero or more normal arguments may occur.

```py
    def test_function(first_param, *arguments):
        """This function accepts its arguments through "arguments" tuple"""
        assert first_param == 'first param'
        assert arguments == ('second param', 'third param')

    test_function('first param', 'second param', 'third param')

    # Normally, these variadic arguments will be last in the list of formal parameters, because
    # they scoop up all remaining input arguments that are passed to the function. Any formal
    # parameters which occur after the *args parameter are ‘keyword-only’ arguments, meaning that
    # they can only be used as keywords rather than positional arguments.
    def concat(*args, sep='/'):
        return sep.join(args)

    assert concat('earth', 'mars', 'venus') == 'earth/mars/venus'
    assert concat('earth', 'mars', 'venus', sep='.') == 'earth.mars.venus'
```

## unpacking args with * and **
'*' <= unpacks from a list or tuple <br>
'**' <= unpacks from a dict
```py
    # write the function call with the *-operator to unpack the arguments out
    # of a list or tuple:

    # Normal call with separate arguments:
    assert list(range(3, 6)) == [3, 4, 5]

    # Call with arguments unpacked from a list.
    arguments_list = [3, 6]
    assert list(range(*arguments_list)) == [3, 4, 5]

    # In the same fashion, dictionaries can deliver keyword arguments with the **-operator:
    def function_that_receives_names_arguments(first_word, second_word):
        return first_word + ', ' + second_word + '!'

    arguments_dictionary = {'first_word': 'Hello', 'second_word': 'World'}
    assert function_that_receives_names_arguments(**arguments_dictionary) == 'Hello, World!'
```

## Classes

### Inheritance
```py
"""Inheritance

@see: https://docs.python.org/3/tutorial/classes.html#inheritance

Inheritance is one of the principles of object-oriented programming. Since classes may share a lot
of the same code, inheritance allows a derived class to reuse the same code and modify accordingly
"""


# pylint: disable=too-few-public-methods
class Person:
    """Example of the base class"""
    def __init__(self, name):
        self.name = name

    def get_name(self):
        """Get person name"""
        return self.name


# The syntax for a derived class definition looks like this.
# pylint: disable=too-few-public-methods
class Employee(Person):
    """Example of the derived class

    The Base Class (in our case Person) must be defined in a scope containing the derived class
    definition. In place of a base class name, other arbitrary expressions are also allowed.

    Derived classes may override methods of their base classes. Because methods have no special
    privileges when calling other methods of the same object, a method of a base class that calls
    another method defined in the same base class may end up calling a method of a derived class
    that overrides it.

    An overriding method in a derived class may in fact want to extend rather than simply replace
    the base class method of the same name. There is a simple way to call the base class method
    directly: just call BaseClassName.methodname(self, arguments). This is occasionally useful to
    clients as well. (Note that this only works if the base class is accessible as BaseClassName
    in the global scope.)
    """
    def __init__(self, name, staff_id):
        Person.__init__(self, name)
        # You may also use super() here in order to avoid explicit using of parent class name:
        # >>> super().__init__(name)
        self.staff_id = staff_id

    def get_full_id(self):
        """Get full employee id"""
        return self.get_name() + ', ' + self.staff_id


def test_inheritance():
    """Inheritance."""

    # There’s nothing special about instantiation of derived classes: DerivedClassName() creates a
    # new instance of the class. Method references are resolved as follows: the corresponding class
    # attribute is searched, descending down the chain of base classes if necessary, and the method
    # reference is valid if this yields a function object.
    person = Person('Bill')
    employee = Employee('John', 'A23')

    assert person.get_name() == 'Bill'
    assert employee.get_name() == 'John'
    assert employee.get_full_id() == 'John, A23'

    # Python has two built-in functions that work with inheritance:
    #
    # - Use isinstance() to check an instance’s type: isinstance(obj, int) will be True only if
    # obj.__class__ is int or some class derived from int.
    #
    # - Use issubclass() to check class inheritance: issubclass(bool, int) is True since bool is
    # a subclass of int. However, issubclass(float, int) is False since float is not a subclass
    # of int.

    assert isinstance(employee, Employee)
    assert not isinstance(person, Employee)

    assert isinstance(person, Person)
    assert isinstance(employee, Person)

    assert issubclass(Employee, Person)
    assert not issubclass(Person, Employee)
```

### Multiple inheritance
```py
"""Multiple Inheritance

@see: https://docs.python.org/3/tutorial/classes.html#multiple-inheritance

Some classes may derive from multiple classes. This means that the derived class would have
its attributes, along with the attributes of all the classes that it was derived from.
"""


def test_multiple_inheritance():
    """Multiple Inheritance"""

    # pylint: disable=too-few-public-methods
    class Clock:
        """Clock class"""

        time = '11:23 PM'

        def get_time(self):
            """Get current time

            Method is hardcoded just for multiple inheritance illustration.
            """
            return self.time

    # pylint: disable=too-few-public-methods
    class Calendar:
        """Calendar class"""

        date = '12/08/2018'

        def get_date(self):
            """Get current date

            Method is hardcoded just for multiple inheritance illustration.
            """
            return self.date

    # Python supports a form of multiple inheritance as well. A class definition with multiple
    # base classes looks like this.
    class CalendarClock(Clock, Calendar):
        """Class that uses multiple inheritance.

        For most purposes, in the simplest cases, you can think of the search for attributes
        inherited from a parent class as depth-first, left-to-right, not searching twice in the same
        class where there is an overlap in the hierarchy. Thus, if an attribute is not found in
        CalendarClock, it is searched for in Clock, then (recursively) in the base classes of
        Clock, and if it was not found there, it was searched for in Calendar, and so on.

        In fact, it is slightly more complex than that; the method resolution order changes
        dynamically to support cooperative calls to super(). This approach is known in some other
        multiple-inheritance languages as call-next-method and is more powerful than the super call
        found in single-inheritance languages.

        Dynamic ordering is necessary because all cases of multiple inheritance exhibit one or more
        diamond relationships (where at least one of the parent classes can be accessed through
        multiple paths from the bottommost class). For example, all classes inherit from object,
        so any case of multiple inheritance provides more than one path to reach object. To keep
        the base classes from being accessed more than once, the dynamic algorithm linearizes the
        search order in a way that preserves the left-to-right ordering specified in each class,
        that calls each parent only once, and that is monotonic (meaning that a class can be
        subclassed without affecting the precedence order of its parents).
        """

    calendar_clock = CalendarClock()

    assert calendar_clock.get_date() == '12/08/2018'
    assert calendar_clock.get_time() == '11:23 PM'
```

## Exceptions
```py
    # Exception handlers may be chained.
    exception_has_been_handled = False
    try:
        result = 10 * (1 / 0)  # division by zero
        # We should not get here at all.
        assert result
    except NameError:
        # We should get here because of division by zero.
        exception_has_been_handled = True
    except ZeroDivisionError:
        # We should get here because of division by zero.
        exception_has_been_handled = Tru
```

### catching multiple exceptions:
```py
try:
    num = int("abc")          # ValueError
    result = 10 / 0           # ZeroDivisionError
except (ValueError, ZeroDivisionError) as e:
    print(f"Error occurred: {e}")
```

### else and finally block:
else → runs only if no exception occurred <br>
finally → runs always (cleanup code)
```py
try:
    file = open("data.txt", "r")
    content = file.read()
except FileNotFoundError:
    print("File not found!")
else:
    print("File read successfully!")
    print(content[:50])     # first 50 chars
finally:
    if 'file' in locals():
        file.close()
        print("File closed.")
```

### raising your own
```py
def withdraw(balance, amount):
    if amount > balance:
        raise ValueError("Insufficient funds!")
    if amount <= 0:
        raise ValueError("Amount must be positive!")
    
    return balance - amount

try:
    new_balance = withdraw(100, 150)
except ValueError as e:
    print(f"Transaction failed: {e}")
```

### custom exception:
```py

class InsufficientFundsError(Exception):
    """Raised when withdrawal amount exceeds balance"""
    pass

def withdraw(balance, amount):
    if amount > balance:
        raise InsufficientFundsError(f"Cannot withdraw ${amount}. Balance: ${balance}")
    return balance - amount

def test_user_defined_exception():
    """User-defined Exceptions"""

    # Programs may name their own exceptions by creating a new exception class. Exceptions should
    # typically be derived from the Exception class, either directly or indirectly.
    # Most exceptions are defined with names that end in “Error,” similar to the naming of the
    # standard exceptions. Many standard modules define their own exceptions to report errors
    # that may occur in functions they define.
    class MyCustomError(Exception):
        """Example of MyCustomError exception."""
        def __init__(self, message):
            super().__init__(message)
            self.message = message

    custom_exception_is_caught = False

    try:
        raise MyCustomError('My custom message')
    except MyCustomError:
        custom_exception_is_caught = True

    assert custom_exception_is_caught
```

## public and private
Public, Private, and Protected variables in Python are conventions, not strict rules enforced by the language.

Python uses naming conventions to signal to other developers (and yourself) how a variable or method should be treated regarding access and modification.

### public
Everything without any leading underscore is public by default.
```py
class Portfolio:
    def __init__(self):
        self.cash = 10000.0          # public
        self.positions = {}          # public

    def add_stock(self, ticker, shares):
        self.positions[ticker] = shares

p = Portfolio()
print(p.cash)           # 10000.0 → fine
p.cash = 5000.0         # allowed
print(p.positions)      # {} → allowed
```

### Protected Variables (Single Underscore _)
The single leading underscore _ is a convention that says:

"This is intended for internal use within the class or module. Please don't touch it from outside unless you really know what you're doing."

Python does not prevent access — it's just a signal to other programmers.

```py
class Stock:
    def __init__(self, ticker, price):
        self.ticker = ticker
        self._current_price = price      # protected
        self._last_updated = "2026-01-21"

    def update_price(self, new_price):
        self._current_price = new_price
        self._last_updated = "now"

s = Stock("AAPL", 245.67)
print(s._current_price)         # 245.67 → technically works, but...
# → Developers should treat this as "don't touch"
```

### Private Variables (Double Underscore __) – Name Mangling
Double leading underscore triggers name mangling — Python automatically renames the attribute to make it harder (but not impossible) to access from outside the class.

```py
class BrokerAccount:
    def __init__(self, account_id, balance):
        self.account_id = account_id
        self.__balance = balance          # private
        self.__api_key = "secret123"      # private

    def get_balance(self):
        return self.__balance

    def _internal_update(self, amount):
        self.__balance += amount

acc = BrokerAccount("ACC123", 50000)

print(acc.account_id)           # ACC123 → public, fine
print(acc.get_balance())        # 50000 → fine (via method)
# print(acc.__balance)          # AttributeError!

# But... you can still access it if you really want (name mangled)
print(acc._BrokerAccount__balance)   # 50000 → works, but please don't do this
```

What Python actually does:

- __balance inside the class becomes _BrokerAccount__balance
- This prevents accidental name clashes in subclasses (inheritance safety)

Real rule of thumb:

- Use __ when you want name mangling (mostly for avoiding conflicts in inheritance)
- Use it sparingly — most Python developers prefer _ for "protected" and document "don't touch" in docstrings/comments.


### properties @property + @setter

Use properties @property + @setter instead of raw private attributes when you need controlled access:

Python properties (using the @property decorator) are a clean and powerful way to:

- Control access to an attribute (make it read-only, computed, or validated)
- Keep the public interface simple (users still use obj.price instead of obj.get_price())
- Hide the implementation details (you can change how the value is stored/calculated without breaking existing code)

```py
class Account:
    def __init__(self):
        self._balance = 0.0

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, value):
        if value < 0:
            raise ValueError("Balance cannot be negative")
        self._balance = value
```

They let you turn a simple attribute access into a method call under the hood, while keeping the syntax looking like a normal variable.

Core Idea – Three Main Parts

- Getter   → @property decorator (makes reading the attribute call a method)
- Setter   → @name.setter decorator (makes assignment obj.attr = value call a method)
- Deleter  → @name.deleter (optional – called when del obj.attr)

Classic Example – Bank Account Balance

```py
class BankAccount:
    def __init__(self, owner, initial_balance=0.0):
        self._balance = initial_balance      # protected (internal storage)
        self.owner = owner

    @property
    def balance(self):
        """Read-only access to balance with nice formatting."""
        return f"${self._balance:,.2f}"

    @balance.setter
    def balance(self, value):
        """Control what can be assigned to balance."""
        if not isinstance(value, (int, float)):
            raise TypeError("Balance must be a number")
        if value < 0:
            raise ValueError("Balance cannot be negative!")
        self._balance = float(value)          # store the real value

    @balance.deleter
    def balance(self):
        print("Balance deleted – account frozen?")
        self._balance = 0.0


# Usage
acc = BankAccount("VK", 5000)

print(acc.balance)          # $5,000.00     ← looks like attribute, calls getter
acc.balance = 7500          # calls setter → validates & stores
print(acc.balance)          # $7,500.00

# These fail nicely:
# acc.balance = -100       → ValueError
# acc.balance = "thousand" → TypeError

del acc.balance             # calls deleter → prints message
```
