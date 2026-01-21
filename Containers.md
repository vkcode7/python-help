# Python Collections & Containers - Practical Guide

## 1. Lists (Ordered, Mutable)
```python
# Creating lists
fruits = ['apple', 'banana', 'cherry']
numbers = [1, 2, 3, 4, 5]

# Common methods
fruits.append('date')              # Add to end: ['apple', 'banana', 'cherry', 'date']
fruits.insert(1, 'orange')         # Insert at index 1
fruits.extend(['fig', 'grape'])    # Add multiple items
fruits.remove('banana')            # Remove first occurrence
popped = fruits.pop()              # Remove and return last item
popped_at = fruits.pop(0)          # Remove at index
fruits.clear()                     # Empty the list

fruits = ['apple', 'banana', 'cherry']
print(fruits[0])                   # Access by index: 'apple'
print(fruits[-1])                  # Last item: 'cherry'
print(fruits[0:2])                 # Slicing: ['apple', 'banana']

count = fruits.count('apple')      # Count occurrences
index = fruits.index('banana')     # Find index of item
fruits.reverse()                   # Reverse in place
fruits.sort()                      # Sort in place
sorted_fruits = sorted(fruits)     # Return new sorted list
```

## 2. Tuples (Ordered, Immutable)

```python
# Creating tuples
coordinates = (10, 20)
single_item = (42,)                # Note the comma for single item
person = ('Alice', 25, 'Engineer')

# Common operations
x, y = coordinates                 # Unpacking
name, age, job = person           # Unpacking

print(person[0])                   # Access by index: 'Alice'
print(person.count('Alice'))       # Count occurrences
print(person.index(25))            # Find index: 1

# Tuples are immutable - this won't work:
# coordinates[0] = 15  # TypeError!
```

## 3. Dictionaries (Key-Value Pairs, Ordered since Python 3.7)
```python
# Creating dictionaries
student = {'name': 'Bob', 'age': 20, 'grade': 'A'}
empty_dict = {}

# Common methods
student['major'] = 'CS'            # Add/update key
age = student.get('age')           # Get value: 20
default = student.get('gpa', 0.0)  # Get with default if key missing

keys = student.keys()              # Get all keys
values = student.values()          # Get all values
items = student.items()            # Get key-value pairs

removed = student.pop('grade')     # Remove and return value
student.update({'gpa': 3.8})      # Update/add multiple items
student.clear()                    # Empty dictionary

student = {'name': 'Bob', 'age': 20}
print('name' in student)           # Check if key exists: True
```

## 4. Sets (Unordered, Unique Elements)
```python
# Creating sets
colors = {'red', 'green', 'blue'}
empty_set = set()                  # Note: {} creates empty dict!
from_list = set([1, 2, 2, 3])     # {1, 2, 3} - duplicates removed

# Common methods
colors.add('yellow')               # Add single item
colors.update(['purple', 'pink'])  # Add multiple items
colors.remove('red')               # Remove (raises error if missing)
colors.discard('orange')           # Remove (no error if missing)
popped = colors.pop()              # Remove and return arbitrary item
colors.clear()                     # Empty the set

# Set operations
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

union = a | b                      # {1, 2, 3, 4, 5, 6}
intersection = a & b               # {3, 4}
difference = a - b                 # {1, 2}
symmetric_diff = a ^ b             # {1, 2, 5, 6}

print(3 in a)                      # Membership test: True
```

## 5. Deque (Double-ended Queue)
```python
from collections import deque

# Creating deque
queue = deque([1, 2, 3])
limited = deque(maxlen=3)          # Max size of 3

# Common methods
queue.append(4)                    # Add to right: [1, 2, 3, 4]
queue.appendleft(0)                # Add to left: [0, 1, 2, 3, 4]
queue.pop()                        # Remove from right: 4
queue.popleft()                    # Remove from left: 0
queue.rotate(1)                    # Rotate right by 1
queue.extend([5, 6])               # Add multiple to right
queue.extendleft([0, -1])          # Add multiple to left
```

## 6. Counter (Count Hashable Objects)
```python
from collections import Counter

# Creating Counter
words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
counter = Counter(words)
# Counter({'apple': 3, 'banana': 2, 'cherry': 1})

letter_count = Counter('mississippi')
# Counter({'i': 4, 's': 4, 'p': 2, 'm': 1})

# Common methods
print(counter['apple'])            # Get count: 3
print(counter['grape'])            # Missing items return 0

counter.update(['apple', 'date'])  # Add counts
counter.subtract(['apple'])        # Subtract counts

most_common = counter.most_common(2)  # [('apple', 4), ('banana', 2)]
elements = list(counter.elements())    # Expand to list

# Combine counters
c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)
print(c1 + c2)                     # Counter({'a': 4, 'b': 3})
```

## 7. DefaultDict (Dict with Default Values)
```python
from collections import defaultdict

# Creating defaultdict
dd = defaultdict(int)              # Default value is 0
dd_list = defaultdict(list)        # Default value is []
dd_set = defaultdict(set)          # Default value is set()

# Usage
word_count = defaultdict(int)
for word in ['apple', 'banana', 'apple']:
    word_count[word] += 1          # No KeyError!

# Group items
groups = defaultdict(list)
for name, category in [('apple', 'fruit'), ('carrot', 'veg'), ('banana', 'fruit')]:
    groups[category].append(name)
# {'fruit': ['apple', 'banana'], 'veg': ['carrot']}
```

## 8. OrderedDict (Remembers Insertion Order)
```python
from collections import OrderedDict

# Note: Regular dicts are ordered in Python 3.7+, but OrderedDict has extra features
od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])

# Special methods
od.move_to_end('a')                # Move 'a' to end
od.move_to_end('c', last=False)    # Move 'c' to beginning
first_item = od.popitem(last=False)  # Remove from beginning
last_item = od.popitem(last=True)    # Remove from end
```

## Lambda Functions with filter, map, sorted, etc.

### filter() - Keep items that return True

``` python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Filter even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))
# [2, 4, 6, 8, 10]

# Filter names longer than 4 characters
names = ['Alice', 'Bob', 'Charlie', 'Dan']
long_names = list(filter(lambda name: len(name) > 4, names))
# ['Alice', 'Charlie']

# Filter positive numbers
nums = [-2, -1, 0, 1, 2, 3]
positives = list(filter(lambda x: x > 0, nums))
# [1, 2, 3]
```

### map() - Transform each item
```python
numbers = [1, 2, 3, 4, 5]

# Square each number
squared = list(map(lambda x: x ** 2, numbers))
# [1, 4, 9, 16, 25]

# Convert to uppercase
names = ['alice', 'bob', 'charlie']
upper = list(map(lambda name: name.upper(), names))
# ['ALICE', 'BOB', 'CHARLIE']

# Map with multiple iterables
nums1 = [1, 2, 3]
nums2 = [4, 5, 6]
sums = list(map(lambda x, y: x + y, nums1, nums2))
# [5, 7, 9]

# Extract specific field
people = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 30}]
ages = list(map(lambda p: p['age'], people))
# [25, 30]
```

### sorted() - Sort with custom key
```python
# Sort by absolute value
numbers = [-5, 2, -8, 3, -1]
sorted_abs = sorted(numbers, key=lambda x: abs(x))
# [-1, 2, 3, -5, -8]

# Sort by length
words = ['apple', 'pie', 'a', 'banana']
by_length = sorted(words, key=lambda w: len(w))
# ['a', 'pie', 'apple', 'banana']

# Sort descending
desc = sorted(numbers, key=lambda x: abs(x), reverse=True)
# [-8, -5, 3, 2, -1]

# Sort list of tuples by second element
pairs = [(1, 'b'), (2, 'a'), (3, 'c')]
sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
# [(2, 'a'), (1, 'b'), (3, 'c')]

# Sort dictionaries by value
students = [
    {'name': 'Alice', 'grade': 85},
    {'name': 'Bob', 'grade': 92},
    {'name': 'Charlie', 'grade': 78}
]
by_grade = sorted(students, key=lambda s: s['grade'], reverse=True)
# Bob, Alice, Charlie
```

### reduce() - Combine items into single value
```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# Sum all numbers
total = reduce(lambda acc, x: acc + x, numbers)
# 15

# Product of all numbers
product = reduce(lambda acc, x: acc * x, numbers)
# 120

# Find maximum
maximum = reduce(lambda a, b: a if a > b else b, numbers)
# 5

# Concatenate strings
words = ['Hello', ' ', 'World', '!']
sentence = reduce(lambda a, b: a + b, words)
# 'Hello World!'
```

### List Comprehensions with Lambdas
```python
numbers = [1, 2, 3, 4, 5]

# Apply lambda in list comprehension
squared = [(lambda x: x ** 2)(n) for n in numbers]
# [1, 4, 9, 16, 25]

# Filter and transform
evens_squared = [(lambda x: x ** 2)(n) for n in numbers if n % 2 == 0]
# [4, 16]

# Dictionary comprehension with lambda
transform = lambda x: x * 2
doubled_dict = {k: transform(k) for k in numbers}
# {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}
```

### any() and all() with Lambdas
```python
numbers = [2, 4, 6, 8, 10]

# Check if any number is odd
has_odd = any(map(lambda x: x % 2 != 0, numbers))
# False

# Check if all numbers are positive
all_positive = all(map(lambda x: x > 0, numbers))
# True

# Works with filter too
names = ['Alice', 'Bob', 'Charlie']
any_short = any(filter(lambda name: len(name) < 4, names))
# True (Bob)
```

### Combining Multiple Operations
```python
# Real-world example: process sales data
sales = [
    {'product': 'Apple', 'price': 1.20, 'quantity': 50},
    {'product': 'Banana', 'price': 0.50, 'quantity': 100},
    {'product': 'Cherry', 'price': 3.00, 'quantity': 30},
    {'product': 'Date', 'price': 2.50, 'quantity': 20}
]

# Filter expensive items, calculate total, sort by revenue
expensive = filter(lambda item: item['price'] > 1, sales)
with_revenue = map(lambda item: {**item, 'revenue': item['price'] * item['quantity']}, expensive)
sorted_sales = sorted(with_revenue, key=lambda item: item['revenue'], reverse=True)

for item in sorted_sales:
    print(f"{item['product']}: ${item['revenue']}")
# Cherry: $90.0
# Apple: $60.0
# Date: $50.0
```

Stack, Tree, and Other Data Structures in Python

### 1. Stack (LIFO - Last In, First Out)

Python doesn't have a built-in Stack class, but you can implement it using lists or deque.
#### Using List as Stack
```python
# Creating a stack
stack = []

# Stack operations
stack.append(1)           # Push: [1]
stack.append(2)           # Push: [1, 2]
stack.append(3)           # Push: [1, 2, 3]

top = stack[-1]           # Peek at top: 3 (without removing)
popped = stack.pop()      # Pop: 3, stack is now [1, 2]
is_empty = len(stack) == 0  # Check if empty: False
size = len(stack)         # Get size: 2

# Example: Balanced parentheses checker
def is_balanced(expression):
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for char in expression:
        if char in pairs:
            stack.append(char)
        elif char in pairs.values():
            if not stack or pairs[stack.pop()] != char:
                return False
    
    return len(stack) == 0

print(is_balanced("({[]})"))  # True
print(is_balanced("({[})"))   # False
```

#### Using Deque as Stack (More Efficient)
```python
from collections import deque

# Creating a stack with deque
stack = deque()

# Stack operations
stack.append(1)           # Push
stack.append(2)
stack.append(3)

top = stack[-1]           # Peek: 3
popped = stack.pop()      # Pop: 3
is_empty = len(stack) == 0

# Example: Reverse a string using stack
def reverse_string(s):
    stack = deque()
    for char in s:
        stack.append(char)
    
    result = ''
    while stack:
        result += stack.pop()
    
    return result

print(reverse_string("hello"))  # "olleh"
```

#### Stack Class Implementation

```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Pop from empty stack")
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Peek from empty stack")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
    
    def __str__(self):
        return str(self.items)

# Usage
stack = Stack()
stack.push(10)
stack.push(20)
stack.push(30)
print(stack)              # [10, 20, 30]
print(stack.peek())       # 30
print(stack.pop())        # 30
print(stack.size())       # 2
```

### 2. Queue (FIFO - First In, First Out)
```python
from collections import deque

# Creating a queue
queue = deque()

# Queue operations
queue.append(1)           # Enqueue (add to back): [1]
queue.append(2)           # [1, 2]
queue.append(3)           # [1, 2, 3]

first = queue[0]          # Peek at front: 1
removed = queue.popleft() # Dequeue (remove from front): 1
is_empty = len(queue) == 0
size = len(queue)

# Queue Class Implementation
class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
        raise IndexError("Dequeue from empty queue")
    
    def front(self):
        if not self.is_empty():
            return self.items[0]
        raise IndexError("Front from empty queue")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Usage
queue = Queue()
queue.enqueue('A')
queue.enqueue('B')
queue.enqueue('C')
print(queue.dequeue())    # 'A'
print(queue.front())      # 'B'
```

### 3. Binary Tree
```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        """Insert value in level-order (for general binary tree)"""
        if not self.root:
            self.root = TreeNode(value)
            return
        
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            
            if not node.left:
                node.left = TreeNode(value)
                return
            else:
                queue.append(node.left)
            
            if not node.right:
                node.right = TreeNode(value)
                return
            else:
                queue.append(node.right)
    
    # Depth-First Traversals
    def inorder(self, node, result=None):
        """Left -> Root -> Right"""
        if result is None:
            result = []
        if node:
            self.inorder(node.left, result)
            result.append(node.value)
            self.inorder(node.right, result)
        return result
    
    def preorder(self, node, result=None):
        """Root -> Left -> Right"""
        if result is None:
            result = []
        if node:
            result.append(node.value)
            self.preorder(node.left, result)
            self.preorder(node.right, result)
        return result
    
    def postorder(self, node, result=None):
        """Left -> Right -> Root"""
        if result is None:
            result = []
        if node:
            self.postorder(node.left, result)
            self.postorder(node.right, result)
            result.append(node.value)
        return result
    
    # Breadth-First Traversal
    def level_order(self):
        """Level by level from left to right"""
        if not self.root:
            return []
        
        result = []
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            result.append(node.value)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return result
    
    def height(self, node):
        """Calculate height of tree"""
        if not node:
            return 0
        return 1 + max(self.height(node.left), self.height(node.right))
    
    def count_nodes(self, node):
        """Count total nodes"""
        if not node:
            return 0
        return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)
    
    def search(self, node, value):
        """Search for a value"""
        if not node:
            return False
        if node.value == value:
            return True
        return self.search(node.left, value) or self.search(node.right, value)

# Usage
tree = BinaryTree()
tree.insert(1)
tree.insert(2)
tree.insert(3)
tree.insert(4)
tree.insert(5)

#       1
#      / \
#     2   3
#    / \
#   4   5

print("Inorder:", tree.inorder(tree.root))      # [4, 2, 5, 1, 3]
print("Preorder:", tree.preorder(tree.root))    # [1, 2, 4, 5, 3]
print("Postorder:", tree.postorder(tree.root))  # [4, 5, 2, 3, 1]
print("Level Order:", tree.level_order())       # [1, 2, 3, 4, 5]
print("Height:", tree.height(tree.root))        # 3
print("Count:", tree.count_nodes(tree.root))    # 5
print("Search 4:", tree.search(tree.root, 4))   # True
```

### 4. Binary Search Tree (BST)
```python
class BSTNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        """Insert maintaining BST property"""
        if not self.root:
            self.root = BSTNode(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = BSTNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = BSTNode(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        """Search for value in BST"""
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
    
    def find_min(self, node=None):
        """Find minimum value"""
        if node is None:
            node = self.root
        while node.left:
            node = node.left
        return node.value
    
    def find_max(self, node=None):
        """Find maximum value"""
        if node is None:
            node = self.root
        while node.right:
            node = node.right
        return node.value
    
    def delete(self, value):
        """Delete a node"""
        self.root = self._delete_recursive(self.root, value)
    
    def _delete_recursive(self, node, value):
        if node is None:
            return None
        
        if value < node.value:
            node.left = self._delete_recursive(node.left, value)
        elif value > node.value:
            node.right = self._delete_recursive(node.right, value)
        else:
            # Node to delete found
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            
            # Node has two children
            min_val = self.find_min(node.right)
            node.value = min_val
            node.right = self._delete_recursive(node.right, min_val)
        
        return node
    
    def inorder(self, node=None, result=None):
        """Inorder gives sorted output for BST"""
        if node is None:
            node = self.root
        if result is None:
            result = []
        if node:
            self.inorder(node.left, result)
            result.append(node.value)
            self.inorder(node.right, result)
        return result

# Usage
bst = BinarySearchTree()
values = [50, 30, 70, 20, 40, 60, 80]
for val in values:
    bst.insert(val)

#       50
#      /  \
#    30    70
#   / \    / \
#  20 40  60 80

print("Inorder (sorted):", bst.inorder())  # [20, 30, 40, 50, 60, 70, 80]
print("Search 40:", bst.search(40))        # True
print("Search 25:", bst.search(25))        # False
print("Min:", bst.find_min())              # 20
print("Max:", bst.find_max())              # 80

bst.delete(30)
print("After deleting 30:", bst.inorder()) # [20, 40, 50, 60, 70, 80]
```

### 5. Heap (Priority Queue)
```python
import heapq

# Min Heap (default in Python)
min_heap = []

heapq.heappush(min_heap, 5)
heapq.heappush(min_heap, 2)
heapq.heappush(min_heap, 8)
heapq.heappush(min_heap, 1)

print(min_heap[0])              # Peek at minimum: 1
smallest = heapq.heappop(min_heap)  # Remove minimum: 1

# Convert list to heap
numbers = [5, 7, 9, 1, 3]
heapq.heapify(numbers)          # [1, 3, 9, 7, 5]

# Get n smallest/largest
data = [5, 2, 8, 1, 9, 3]
print(heapq.nsmallest(3, data)) # [1, 2, 3]
print(heapq.nlargest(3, data))  # [9, 8, 5]

# Max Heap (negate values)
max_heap = []
for num in [5, 2, 8, 1]:
    heapq.heappush(max_heap, -num)

maximum = -heapq.heappop(max_heap)  # Get maximum: 8

# Heap with custom objects
class Task:
    def __init__(self, priority, description):
        self.priority = priority
        self.description = description
    
    def __lt__(self, other):
        return self.priority < other.priority

task_heap = []
heapq.heappush(task_heap, Task(3, "Low priority"))
heapq.heappush(task_heap, Task(1, "High priority"))
heapq.heappush(task_heap, Task(2, "Medium priority"))

urgent = heapq.heappop(task_heap)
print(urgent.description)  # "High priority"
```

### 6. Trie (Prefix Tree)
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """Insert a word into the trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word):
        """Search for exact word"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def starts_with(self, prefix):
        """Check if any word starts with prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def get_all_words_with_prefix(self, prefix):
        """Get all words that start with prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        words = []
        self._dfs(node, prefix, words)
        return words
    
    def _dfs(self, node, current_word, words):
        if node.is_end_of_word:
            words.append(current_word)
        
        for char, child_node in node.children.items():
            self._dfs(child_node, current_word + char, words)

# Usage
trie = Trie()
words = ["apple", "app", "application", "apply", "banana", "band"]
for word in words:
    trie.insert(word)

print(trie.search("apple"))           # True
print(trie.search("app"))             # True
print(trie.search("appl"))            # False (prefix but not word)
print(trie.starts_with("app"))        # True
print(trie.get_all_words_with_prefix("app"))  
# ['app', 'apple', 'application', 'apply']
```

### 7. Graph
```
python
from collections import defaultdict, deque

class Graph:
    def __init__(self, directed=False):
        self.graph = defaultdict(list)
        self.directed = directed
    
    def add_edge(self, u, v):
        """Add an edge from u to v"""
        self.graph[u].append(v)
        if not self.directed:
            self.graph[v].append(u)
    
    def bfs(self, start):
        """Breadth-First Search"""
        visited = set()
        queue = deque([start])
        result = []
        
        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                
                for neighbor in self.graph[vertex]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return result
    
    def dfs(self, start, visited=None, result=None):
        """Depth-First Search"""
        if visited is None:
            visited = set()
            result = []
        
        visited.add(start)
        result.append(start)
        
        for neighbor in self.graph[start]:
            if neighbor not in visited:
                self.dfs(neighbor, visited, result)
        
        return result
    
    def has_path(self, start, end):
        """Check if path exists between start and end"""
        visited = set()
        queue = deque([start])
        
        while queue:
            vertex = queue.popleft()
            if vertex == end:
                return True
            
            if vertex not in visited:
                visited.add(vertex)
                for neighbor in self.graph[vertex]:
                    queue.append(neighbor)
        
        return False
    
    def get_all_paths(self, start, end, path=None):
        """Find all paths from start to end"""
        if path is None:
            path = []
        
        path = path + [start]
        
        if start == end:
            return [path]
        
        paths = []
        for neighbor in self.graph[start]:
            if neighbor not in path:
                new_paths = self.get_all_paths(neighbor, end, path)
                paths.extend(new_paths)
        
        return paths

# Usage
g = Graph()
g.add_edge('A', 'B')
g.add_edge('A', 'C')
g.add_edge('B', 'D')
g.add_edge('C', 'D')
g.add_edge('D', 'E')

#    A
#   / \
#  B   C
#   \ /
#    D
#    |
#    E

print("BFS from A:", g.bfs('A'))           # ['A', 'B', 'C', 'D', 'E']
print("DFS from A:", g.dfs('A'))           # ['A', 'B', 'D', 'C', 'E']
print("Has path A to E:", g.has_path('A', 'E'))  # True
print("All paths A to E:", g.get_all_paths('A', 'E'))
# [['A', 'B', 'D', 'E'], ['A', 'C', 'D', 'E']]
```

### 8. Linked List (Singly)
```py
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def is_empty(self):
        """Check if list is empty"""
        return self.head is None
    
    def append(self, data):
        """Add node at the end"""
        new_node = Node(data)
        
        if self.is_empty():
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, data):
        """Add node at the beginning"""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def insert_after(self, prev_data, data):
        """Insert after a specific node"""
        current = self.head
        
        while current and current.data != prev_data:
            current = current.next
        
        if current is None:
            print(f"Node with data {prev_data} not found")
            return
        
        new_node = Node(data)
        new_node.next = current.next
        current.next = new_node
    
    def delete(self, data):
        """Delete first occurrence of data"""
        if self.is_empty():
            print("List is empty")
            return
        
        # If head needs to be deleted
        if self.head.data == data:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next and current.next.data != data:
            current = current.next
        
        if current.next is None:
            print(f"Node with data {data} not found")
            return
        
        current.next = current.next.next
    
    def delete_at_position(self, position):
        """Delete node at specific position (0-indexed)"""
        if self.is_empty():
            print("List is empty")
            return
        
        if position == 0:
            self.head = self.head.next
            return
        
        current = self.head
        for i in range(position - 1):
            if current.next is None:
                print("Position out of range")
                return
            current = current.next
        
        if current.next is None:
            print("Position out of range")
            return
        
        current.next = current.next.next
    
    def search(self, data):
        """Search for data in the list"""
        current = self.head
        position = 0
        
        while current:
            if current.data == data:
                return position
            current = current.next
            position += 1
        
        return -1  # Not found
    
    def get(self, position):
        """Get data at specific position"""
        current = self.head
        
        for i in range(position):
            if current is None:
                raise IndexError("Position out of range")
            current = current.next
        
        if current is None:
            raise IndexError("Position out of range")
        
        return current.data
    
    def length(self):
        """Get length of the list"""
        count = 0
        current = self.head
        
        while current:
            count += 1
            current = current.next
        
        return count
    
    def reverse(self):
        """Reverse the linked list"""
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev
    
    def get_middle(self):
        """Get middle element using slow and fast pointers"""
        if self.is_empty():
            return None
        
        slow = fast = self.head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow.data
    
    def has_cycle(self):
        """Detect if list has a cycle (Floyd's algorithm)"""
        if self.is_empty():
            return False
        
        slow = fast = self.head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
        
        return False
    
    def remove_duplicates(self):
        """Remove duplicate values from sorted list"""
        current = self.head
        
        while current and current.next:
            if current.data == current.next.data:
                current.next = current.next.next
            else:
                current = current.next
    
    def to_list(self):
        """Convert to Python list"""
        result = []
        current = self.head
        
        while current:
            result.append(current.data)
            current = current.next
        
        return result
    
    def display(self):
        """Display the list"""
        elements = self.to_list()
        print(" -> ".join(map(str, elements)))
    
    def __str__(self):
        return " -> ".join(map(str, self.to_list()))

# Usage Examples
ll = LinkedList()

# Append elements
ll.append(1)
ll.append(2)
ll.append(3)
ll.display()  # 1 -> 2 -> 3

# Prepend
ll.prepend(0)
ll.display()  # 0 -> 1 -> 2 -> 3

# Insert after
ll.insert_after(2, 2.5)
ll.display()  # 0 -> 1 -> 2 -> 2.5 -> 3

# Delete
ll.delete(2.5)
ll.display()  # 0 -> 1 -> 2 -> 3

# Search
position = ll.search(2)
print(f"Found 2 at position: {position}")  # 2

# Get element
element = ll.get(1)
print(f"Element at position 1: {element}")  # 1

# Length
print(f"Length: {ll.length()}")  # 4

# Get middle
print(f"Middle element: {ll.get_middle()}")  # 1

# Reverse
ll.reverse()
ll.display()  # 3 -> 2 -> 1 -> 0

# Remove duplicates (for sorted list)
ll2 = LinkedList()
for val in [1, 1, 2, 3, 3, 3, 4]:
    ll2.append(val)
ll2.display()  # 1 -> 1 -> 2 -> 3 -> 3 -> 3 -> 4
ll2.remove_duplicates()
ll2.display()  # 1 -> 2 -> 3 -> 4
```

### 9. Linked List (Double)
```py
class DNode:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    
    def is_empty(self):
        """Check if list is empty"""
        return self.head is None
    
    def append(self, data):
        """Add node at the end"""
        new_node = DNode(data)
        
        if self.is_empty():
            self.head = self.tail = new_node
            return
        
        self.tail.next = new_node
        new_node.prev = self.tail
        self.tail = new_node
    
    def prepend(self, data):
        """Add node at the beginning"""
        new_node = DNode(data)
        
        if self.is_empty():
            self.head = self.tail = new_node
            return
        
        new_node.next = self.head
        self.head.prev = new_node
        self.head = new_node
    
    def insert_after(self, prev_data, data):
        """Insert after a specific node"""
        current = self.head
        
        while current and current.data != prev_data:
            current = current.next
        
        if current is None:
            print(f"Node with data {prev_data} not found")
            return
        
        new_node = DNode(data)
        new_node.next = current.next
        new_node.prev = current
        
        if current.next:
            current.next.prev = new_node
        else:
            self.tail = new_node
        
        current.next = new_node
    
    def insert_before(self, next_data, data):
        """Insert before a specific node"""
        current = self.head
        
        while current and current.data != next_data:
            current = current.next
        
        if current is None:
            print(f"Node with data {next_data} not found")
            return
        
        new_node = DNode(data)
        new_node.prev = current.prev
        new_node.next = current
        
        if current.prev:
            current.prev.next = new_node
        else:
            self.head = new_node
        
        current.prev = new_node
    
    def delete(self, data):
        """Delete first occurrence of data"""
        current = self.head
        
        while current and current.data != data:
            current = current.next
        
        if current is None:
            print(f"Node with data {data} not found")
            return
        
        if current.prev:
            current.prev.next = current.next
        else:
            self.head = current.next
        
        if current.next:
            current.next.prev = current.prev
        else:
            self.tail = current.prev
    
    def delete_from_end(self):
        """Delete last node"""
        if self.is_empty():
            print("List is empty")
            return
        
        if self.head == self.tail:
            self.head = self.tail = None
            return
        
        self.tail = self.tail.prev
        self.tail.next = None
    
    def delete_from_beginning(self):
        """Delete first node"""
        if self.is_empty():
            print("List is empty")
            return
        
        if self.head == self.tail:
            self.head = self.tail = None
            return
        
        self.head = self.head.next
        self.head.prev = None
    
    def reverse(self):
        """Reverse the doubly linked list"""
        current = self.head
        self.head, self.tail = self.tail, self.head
        
        while current:
            current.prev, current.next = current.next, current.prev
            current = current.prev
    
    def display_forward(self):
        """Display list from head to tail"""
        current = self.head
        elements = []
        
        while current:
            elements.append(current.data)
            current = current.next
        
        print(" <-> ".join(map(str, elements)))
    
    def display_backward(self):
        """Display list from tail to head"""
        current = self.tail
        elements = []
        
        while current:
            elements.append(current.data)
            current = current.prev
        
        print(" <-> ".join(map(str, elements)))
    
    def to_list(self):
        """Convert to Python list"""
        result = []
        current = self.head
        
        while current:
            result.append(current.data)
            current = current.next
        
        return result

# Usage Examples
dll = DoublyLinkedList()

# Append
dll.append(1)
dll.append(2)
dll.append(3)
dll.display_forward()  # 1 <-> 2 <-> 3

# Prepend
dll.prepend(0)
dll.display_forward()  # 0 <-> 1 <-> 2 <-> 3

# Insert after
dll.insert_after(2, 2.5)
dll.display_forward()  # 0 <-> 1 <-> 2 <-> 2.5 <-> 3

# Insert before
dll.insert_before(0, -1)
dll.display_forward()  # -1 <-> 0 <-> 1 <-> 2 <-> 2.5 <-> 3

# Delete
dll.delete(2.5)
dll.display_forward()  # -1 <-> 0 <-> 1 <-> 2 <-> 3

# Display backward
dll.display_backward()  # 3 <-> 2 <-> 1 <-> 0 <-> -1

# Delete from end
dll.delete_from_end()
dll.display_forward()  # -1 <-> 0 <-> 1 <-> 2

# Delete from beginning
dll.delete_from_beginning()
dll.display_forward()  # 0 <-> 1 <-> 2

# Reverse
dll.reverse()
dll.display_forward()  # 2 <-> 1 <-> 0
```

### 10. Common Linked List Problems with Lambda Functions 
```py
# Merge two sorted linked lists
def merge_sorted_lists(l1, l2):
    """Merge two sorted linked lists"""
    dummy = Node(0)
    current = dummy
    
    while l1 and l2:
        if l1.data < l2.data:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    current.next = l1 or l2
    return dummy.next

# Find nth node from end
def nth_from_end(head, n):
    """Find nth node from end using two pointers"""
    fast = slow = head
    
    for _ in range(n):
        if fast is None:
            return None
        fast = fast.next
    
    while fast:
        slow = slow.next
        fast = fast.next
    
    return slow.data if slow else None

# Palindrome check
def is_palindrome(head):
    """Check if linked list is palindrome"""
    values = []
    current = head
    
    while current:
        values.append(current.data)
        current = current.next
    
    return values == values[::-1]

# Using lambda with linked list operations
ll = LinkedList()
for i in range(1, 6):
    ll.append(i)

# Filter even numbers
evens = list(filter(lambda x: x % 2 == 0, ll.to_list()))
print("Even numbers:", evens)  # [2, 4]

# Map: square all elements
squared = list(map(lambda x: x ** 2, ll.to_list()))
print("Squared:", squared)  # [1, 4, 9, 16, 25]

# Sorted with custom key
data_ll = LinkedList()
for val in [5, 2, 8, 1, 9]:
    data_ll.append(val)

sorted_data = sorted(data_ll.to_list(), key=lambda x: -x)  # Descending
print("Sorted descending:", sorted_data)  # [9, 8, 5, 2, 1]

# Create new linked list from filtered/mapped data
new_ll = LinkedList()
filtered_squared = map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, data_ll.to_list()))
for val in filtered_squared:
    new_ll.append(val)

new_ll.display()  # 4 -> 64 (squares of 2 and 8)
```

### 11. LRU Cache using Doubly Linked List
```py
class LRUCache:
    """Least Recently Used Cache"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> DNode
        self.head = DNode(0)  # Dummy head
        self.tail = DNode(0)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove(self, node):
        """Remove node from list"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_head(self, node):
        """Add node right after head"""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
    
    def get(self, key):
        """Get value and move to front (most recently used)"""
        if key not in self.cache:
            return -1
        
        node = self.cache[key]
        self._remove(node)
        self._add_to_head(node)
        return node.data
    
    def put(self, key, value):
        """Put key-value pair"""
        if key in self.cache:
            self._remove(self.cache[key])
        
        node = DNode(value)
        node.key = key
        self.cache[key] = node
        self._add_to_head(node)
        
        if len(self.cache) > self.capacity:
            # Remove least recently used (node before tail)
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]

# Usage
lru = LRUCache(2)
lru.put(1, 1)
lru.put(2, 2)
print(lru.get(1))     # 1
lru.put(3, 3)         # Evicts key 2
print(lru.get(2))     # -1 (not found)
```