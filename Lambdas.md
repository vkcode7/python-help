Lambda expressions are a way to create small, anonymous (nameless) functions in one line. They're perfect when you need a tiny function for a short time — usually passed directly to another function (like map(), filter(), sorted(), etc.).

## Basic Syntax:
```py
# Normal named function
def add(a, b):
    return a + b

# Same thing as lambda
add_lambda = lambda a, b: a + b

sq = lambda n: n*n
get_first_letter = lambda s: s[0]
check_if_even = lambda n: n % 2 == 0
length = lambda s: len(s)
```

## Common real world use cases

### sorted() with custom key:
```py
# Sort list of tuples by second element
pairs = [(1, 'one'), (3, 'three'), (2, 'two'), (4, 'four')]

# Using lambda
sorted_pairs = sorted(pairs, key=lambda x: x[1])   # sort by word
print(sorted_pairs)
# → [(1, 'one'), (4, 'four'), (3, 'three'), (2, 'two')]

# Sort list of dicts by age
people = [
    {"name": "Anna", "age": 34},
    {"name": "Ben", "age": 22},
    {"name": "Clara", "age": 41}
]

sorted_people = sorted(people, key=lambda p: p["age"])
print(sorted_people)
```

### map() -> apply to every item:
```py
numbers = [1, 2, 3, 4, 5]

# Square every number
squares = list(map(lambda x: x*x, numbers))
print(squares)          # [1, 4, 9, 16, 25]

# Double every number
doubles = list(map(lambda x: x*2, numbers))
print(doubles)          # [2, 4, 6, 8, 10]
```

### filter() – keep only items that match condition
```py
numbers = [10, 15, 3, 7, 22, 8, 9]

# Keep only even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)            # [10, 22, 8]

# Keep numbers > 10
big = list(filter(lambda x: x > 10, numbers))
print(big)              # [15, 22]
```

### Combining with reduce() (from functools)
```py
from functools import reduce

numbers = [2, 3, 4, 5]

# Multiply all numbers
product = reduce(lambda x, y: x * y, numbers)
print(product)          # 120  (2×3×4×5)
```

### When to Use Lambda (and When NOT)
Use lambda when:

- The function is very small (1 expression)
- You're passing it immediately to sorted, map, filter, max, min, etc.
- You don't need to reuse the function elsewhere

Avoid lambda when:

- The logic is complicated (more than 1–2 operations)
- You need multiple statements (if, for, try, etc.)
- You want to reuse the function many times → use def instead
- Readability suffers (long lambdas are hard to understand)