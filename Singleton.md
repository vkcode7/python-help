# Singleton Pattern in Python
The Singleton pattern ensures a class has only one instance and provides a global point of access to it.

## 1. Using __new__ Method (Most Common)
```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Initialize only once
        if not hasattr(self, 'initialized'):
            self.value = 0
            self.initialized = True

# Usage
s1 = Singleton()
s2 = Singleton()

print(s1 is s2)  # True - same instance
s1.value = 42
print(s2.value)  # 42 - shared state
```

## 2. Thread-Safe Singleton
```python
import threading

class ThreadSafeSingleton:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.data = []
            self.initialized = True

# Usage
def create_instance():
    instance = ThreadSafeSingleton()
    print(f"Instance ID: {id(instance)}")

# Create multiple threads
threads = [threading.Thread(target=create_instance) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# All threads get the same instance ID
```

## 3. Using Metaclass
```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "Database Connection"
    
    def query(self, sql):
        return f"Executing: {sql}"

# Usage
db1 = Database()
db2 = Database()

print(db1 is db2)  # True
print(db1.query("SELECT * FROM users"))
```

## 4. Using Decorator
```python
def singleton(cls):
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class Logger:
    def __init__(self):
        self.logs = []
    
    def log(self, message):
        self.logs.append(message)
        print(f"LOG: {message}")
    
    def show_logs(self):
        return self.logs

# Usage
logger1 = Logger()
logger2 = Logger()

print(logger1 is logger2)  # True

logger1.log("First message")
logger2.log("Second message")
print(logger1.show_logs())  # ['First message', 'Second message']
```

## 5. Using Module (Pythonic Way)
```python
# singleton_module.py
class _ConfigManager:
    def __init__(self):
        self.settings = {}
    
    def set(self, key, value):
        self.settings[key] = value
    
    def get(self, key):
        return self.settings.get(key)

# Create single instance
config_manager = _ConfigManager()

# In your code, import the instance:
# from singleton_module import config_manager
# config_manager.set('api_key', '12345')
```

## 6. Using __init_subclass__ (Python 3.6+)
```python
class SingletonBase:
    _instances = {}
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

class AppConfig(SingletonBase):
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.config = {}
            self.initialized = True
    
    def set_config(self, key, value):
        self.config[key] = value
    
    def get_config(self, key):
        return self.config.get(key)

# Usage
config1 = AppConfig()
config2 = AppConfig()

print(config1 is config2)  # True
config1.set_config('debug', True)
print(config2.get_config('debug'))  # True
```

## 7. Real-World Example: Database Connection Pool
```python
import threading
from typing import Optional

class DatabaseConnectionPool:
    _instance: Optional['DatabaseConnectionPool'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.max_connections = 10
            self.active_connections = 0
            self.connection_pool = []
            self.initialized = True
            print("Database Connection Pool initialized")
    
    def get_connection(self):
        if self.active_connections < self.max_connections:
            self.active_connections += 1
            connection = f"Connection-{self.active_connections}"
            self.connection_pool.append(connection)
            return connection
        else:
            return "No available connections"
    
    def release_connection(self, connection):
        if connection in self.connection_pool:
            self.connection_pool.remove(connection)
            self.active_connections -= 1
            return f"Released {connection}"
        return "Connection not found"
    
    def get_stats(self):
        return {
            'active': self.active_connections,
            'max': self.max_connections,
            'available': self.max_connections - self.active_connections
        }

# Usage
pool1 = DatabaseConnectionPool()
pool2 = DatabaseConnectionPool()

print(pool1 is pool2)  # True

conn1 = pool1.get_connection()
conn2 = pool2.get_connection()

print(f"Connection 1: {conn1}")
print(f"Connection 2: {conn2}")
print(f"Stats: {pool1.get_stats()}")

pool1.release_connection(conn1)
print(f"Stats after release: {pool2.get_stats()}")
```

## 8. Real-World Example: Application Settings
```python
import json
from pathlib import Path

class Settings:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self._settings = {
                'app_name': 'MyApp',
                'version': '1.0.0',
                'debug': False,
                'max_retries': 3,
                'timeout': 30
            }
            self.initialized = True
    
    def get(self, key, default=None):
        return self._settings.get(key, default)
    
    def set(self, key, value):
        self._settings[key] = value
    
    def update(self, settings_dict):
        self._settings.update(settings_dict)
    
    def load_from_file(self, filepath):
        try:
            with open(filepath, 'r') as f:
                self._settings.update(json.load(f))
        except FileNotFoundError:
            print(f"Settings file {filepath} not found")
    
    def save_to_file(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self._settings, f, indent=4)
    
    def all(self):
        return self._settings.copy()

# Usage
settings1 = Settings()
settings2 = Settings()

print(settings1 is settings2)  # True

settings1.set('debug', True)
settings1.set('api_key', 'secret123')

print(settings2.get('debug'))  # True
print(settings2.get('api_key'))  # secret123

# All settings
print(settings1.all())
```

## 9. Comparing Multiple Singleton Implementations
```python
import time

# Method 1: __new__
class Singleton1:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Method 2: Metaclass
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton2(metaclass=SingletonMeta):
    pass

# Method 3: Decorator
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Singleton3:
    pass

# Test all methods
print("Method 1 (__new__):")
s1a = Singleton1()
s1b = Singleton1()
print(f"Same instance: {s1a is s1b}")  # True

print("\nMethod 2 (Metaclass):")
s2a = Singleton2()
s2b = Singleton2()
print(f"Same instance: {s2a is s2b}")  # True

print("\nMethod 3 (Decorator):")
s3a = Singleton3()
s3b = Singleton3()
print(f"Same instance: {s3a is s3b}")  # True
```

## 10. Anti-Pattern Warning and Best Practices
```python
# ❌ BAD: Forcing Singleton when not needed
class UnnecessarySingleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def calculate(self, x, y):
        return x + y  # This doesn't need to be a singleton!

# ✅ GOOD: Use Singleton only when you need exactly one instance
class CacheManager:
    """Cache should be shared across the application"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.cache = {}
            self.initialized = True
    
    def set(self, key, value):
        self.cache[key] = value
    
    def get(self, key):
        return self.cache.get(key)

# ✅ BETTER: For simple cases, just use a module-level variable

# cache_manager.py
_cache = {}

def set_cache(key, value):
    _cache[key] = value

def get_cache(key):
    return _cache.get(key)
```
## Key Points:
```text
__new__ method is the most common and straightforward
Metaclass approach is more powerful but harder to understand
Decorator approach is clean and explicit
Module-level instance is the most Pythonic for simple cases
Thread-safe implementation is crucial for multi-threaded applications
Always check if you truly need a Singleton - it can make testing harder
Use Singletons for resources that should genuinely be shared (config, logging, connection pools)
```
The Singleton pattern is useful but should be used sparingly - only when you truly need exactly one instance of a class!