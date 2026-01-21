# Tasks and Synchronization in Python

## 1. Threading Basics
```python
import threading
import time

# Simple thread example
def worker(name, delay):
    print(f"{name} starting")
    time.sleep(delay)
    print(f"{name} finished")

# Create threads
thread1 = threading.Thread(target=worker, args=("Thread-1", 2))
thread2 = threading.Thread(target=worker, args=("Thread-2", 1))

# Start threads
thread1.start()
thread2.start()

# Wait for threads to complete
thread1.join()
thread2.join()

print("All threads completed")

# Using Thread class
class WorkerThread(threading.Thread):
    def __init__(self, name, delay):
        super().__init__()
        self.name = name
        self.delay = delay
    
    def run(self):
        print(f"{self.name} starting")
        time.sleep(self.delay)
        print(f"{self.name} finished")

# Usage
t1 = WorkerThread("Worker-1", 2)
t2 = WorkerThread("Worker-2", 1)
t1.start()
t2.start()
t1.join()
t2.join()
```

## 2. Threading with Return Values
```python
import threading
from typing import Any

class ThreadWithReturnValue(threading.Thread):
    def __init__(self, target, args=()):
        super().__init__()
        self._target = target
        self._args = args
        self._return = None
    
    def run(self):
        if self._target:
            self._return = self._target(*self._args)
    
    def join(self, *args):
        super().join(*args)
        return self._return

def calculate_square(n):
    time.sleep(1)
    return n * n

# Usage
thread = ThreadWithReturnValue(target=calculate_square, args=(5,))
thread.start()
result = thread.join()
print(f"Result: {result}")  # 25

# Using concurrent.futures (Better approach)
from concurrent.futures import ThreadPoolExecutor

def calculate_cube(n):
    time.sleep(1)
    return n ** 3

with ThreadPoolExecutor(max_workers=3) as executor:
    future = executor.submit(calculate_cube, 5)
    result = future.result()
    print(f"Cube: {result}")  # 125
```

## 3. Lock (Mutual Exclusion)
```python
import threading

# Problem without lock (Race condition)
counter = 0

def increment_without_lock():
    global counter
    for _ in range(100000):
        counter += 1  # Not atomic!

threads = [threading.Thread(target=increment_without_lock) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Counter without lock: {counter}")  # Likely less than 1000000

# Solution with lock
counter = 0
lock = threading.Lock()

def increment_with_lock():
    global counter
    for _ in range(100000):
        with lock:  # or lock.acquire() ... lock.release()
            counter += 1

threads = [threading.Thread(target=increment_with_lock) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Counter with lock: {counter}")  # Exactly 1000000

# Lock context manager (try/finally pattern)
def manual_lock_usage():
    lock.acquire()
    try:
        # Critical section
        print("Working with shared resource")
    finally:
        lock.release()
```

## 4. RLock (Reentrant Lock)
```python
import threading

# Regular Lock causes deadlock
lock = threading.Lock()

def recursive_function(n):
    with lock:
        print(f"Level {n}")
        if n > 0:
            recursive_function(n - 1)  # Deadlock! Can't acquire same lock

# Solution: RLock
rlock = threading.RLock()

def recursive_function_safe(n):
    with rlock:
        print(f"Level {n}")
        if n > 0:
            recursive_function_safe(n - 1)  # Works! Same thread can acquire

thread = threading.Thread(target=recursive_function_safe, args=(5,))
thread.start()
thread.join()

# RLock example: Bank account
class BankAccount:
    def __init__(self):
        self.balance = 0
        self.lock = threading.RLock()
    
    def deposit(self, amount):
        with self.lock:
            self.balance += amount
            print(f"Deposited {amount}, Balance: {self.balance}")
    
    def withdraw(self, amount):
        with self.lock:
            if self.balance >= amount:
                self.balance -= amount
                print(f"Withdrew {amount}, Balance: {self.balance}")
                return True
            return False
    
    def transfer_to(self, other_account, amount):
        with self.lock:  # Lock this account
            if self.withdraw(amount):  # Can acquire lock again (reentrant)
                other_account.deposit(amount)
                return True
            return False

account1 = BankAccount()
account2 = BankAccount()
account1.deposit(100)
account1.transfer_to(account2, 50)
```

## 5. Semaphore (Limited Resources)
```python
import threading
import time

# Semaphore: Limit concurrent access to N resources
semaphore = threading.Semaphore(3)  # Allow 3 concurrent accesses

def access_resource(worker_id):
    print(f"Worker {worker_id} waiting...")
    with semaphore:
        print(f"Worker {worker_id} accessing resource")
        time.sleep(2)
        print(f"Worker {worker_id} released resource")

threads = [threading.Thread(target=access_resource, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Real-world example: Connection Pool
class ConnectionPool:
    def __init__(self, max_connections):
        self.semaphore = threading.Semaphore(max_connections)
        self.connections = []
    
    def get_connection(self, client_id):
        self.semaphore.acquire()
        connection = f"Connection-{client_id}"
        self.connections.append(connection)
        print(f"{client_id} acquired {connection}")
        return connection
    
    def release_connection(self, connection, client_id):
        if connection in self.connections:
            self.connections.remove(connection)
            print(f"{client_id} released {connection}")
        self.semaphore.release()

pool = ConnectionPool(3)

def use_connection(client_id):
    conn = pool.get_connection(client_id)
    time.sleep(2)  # Simulate work
    pool.release_connection(conn, client_id)

threads = [threading.Thread(target=use_connection, args=(f"Client-{i}",)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

## 6. BoundedSemaphore (Prevents Release Above Limit)
```python
import threading

# Regular Semaphore
regular_sem = threading.Semaphore(2)
regular_sem.acquire()
regular_sem.acquire()
regular_sem.release()
regular_sem.release()
regular_sem.release()  # No error, but now value is 3!

# BoundedSemaphore
bounded_sem = threading.BoundedSemaphore(2)
bounded_sem.acquire()
bounded_sem.acquire()
bounded_sem.release()
bounded_sem.release()
# bounded_sem.release()  # ValueError: Semaphore released too many times

# Use case: Resource pool with strict limits
class StrictResourcePool:
    def __init__(self, max_resources):
        self.semaphore = threading.BoundedSemaphore(max_resources)
        self.resources_in_use = []
        self.lock = threading.Lock()
    
    def acquire_resource(self, resource_id):
        self.semaphore.acquire()
        with self.lock:
            self.resources_in_use.append(resource_id)
            print(f"Acquired: {resource_id}, In use: {len(self.resources_in_use)}")
    
    def release_resource(self, resource_id):
        with self.lock:
            if resource_id in self.resources_in_use:
                self.resources_in_use.remove(resource_id)
                print(f"Released: {resource_id}, In use: {len(self.resources_in_use)}")
                self.semaphore.release()
            else:
                print(f"Error: {resource_id} not in use!")
```

## 7. Event (Thread Signaling)
```python
import threading
import time

# Event for thread coordination
event = threading.Event()

def waiter(name):
    print(f"{name} waiting for event")
    event.wait()  # Block until event is set
    print(f"{name} received event, continuing")

def setter():
    print("Setter sleeping for 3 seconds")
    time.sleep(3)
    print("Setter setting event")
    event.set()  # Wake up all waiting threads

# Create threads
waiters = [threading.Thread(target=waiter, args=(f"Waiter-{i}",)) for i in range(3)]
setter_thread = threading.Thread(target=setter)

for w in waiters:
    w.start()
setter_thread.start()

for w in waiters:
    w.join()
setter_thread.join()

# Event methods
print(f"Is set: {event.is_set()}")  # True
event.clear()  # Reset event
print(f"Is set: {event.is_set()}")  # False

# Real-world example: Download manager
class DownloadManager:
    def __init__(self):
        self.download_complete = threading.Event()
        self.data = None
    
    def download(self, url):
        print(f"Downloading from {url}...")
        time.sleep(3)  # Simulate download
        self.data = f"Data from {url}"
        print("Download complete!")
        self.download_complete.set()
    
    def process_data(self):
        print("Waiting for download to complete...")
        self.download_complete.wait()  # Wait for download
        print(f"Processing: {self.data}")

manager = DownloadManager()
download_thread = threading.Thread(target=manager.download, args=("http://example.com",))
process_thread = threading.Thread(target=manager.process_data)

process_thread.start()
download_thread.start()

download_thread.join()
process_thread.join()
```

## 8. Condition Variable (Advanced Synchronization)
```python
import threading
import time
from collections import deque

# Producer-Consumer problem
class ProducerConsumer:
    def __init__(self):
        self.queue = deque()
        self.condition = threading.Condition()
        self.max_size = 5
    
    def produce(self, item):
        with self.condition:
            # Wait if queue is full
            while len(self.queue) >= self.max_size:
                print(f"Queue full, producer waiting...")
                self.condition.wait()
            
            self.queue.append(item)
            print(f"Produced: {item}, Queue size: {len(self.queue)}")
            
            # Notify consumers
            self.condition.notify()
    
    def consume(self):
        with self.condition:
            # Wait if queue is empty
            while len(self.queue) == 0:
                print(f"Queue empty, consumer waiting...")
                self.condition.wait()
            
            item = self.queue.popleft()
            print(f"Consumed: {item}, Queue size: {len(self.queue)}")
            
            # Notify producers
            self.condition.notify()
            return item

pc = ProducerConsumer()

def producer():
    for i in range(10):
        pc.produce(f"Item-{i}")
        time.sleep(0.5)

def consumer():
    for _ in range(10):
        pc.consume()
        time.sleep(1)

prod_thread = threading.Thread(target=producer)
cons_thread = threading.Thread(target=consumer)

prod_thread.start()
cons_thread.start()

prod_thread.join()
cons_thread.join()
```

## 9. Barrier (Wait for All Threads)
```python
import threading
import time

# Barrier: Wait for N threads before continuing
barrier = threading.Barrier(3)

def worker(worker_id):
    print(f"Worker {worker_id} starting")
    time.sleep(worker_id)  # Different work times
    
    print(f"Worker {worker_id} waiting at barrier")
    barrier.wait()  # Block until all 3 threads reach here
    
    print(f"Worker {worker_id} continuing after barrier")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Real-world example: Parallel algorithm phases
class ParallelProcessor:
    def __init__(self, num_workers):
        self.barrier = threading.Barrier(num_workers)
        self.results = []
        self.lock = threading.Lock()
    
    def process_phase1(self, worker_id):
        result = worker_id * 2
        print(f"Worker {worker_id} completed phase 1: {result}")
        
        with self.lock:
            self.results.append(result)
        
        print(f"Worker {worker_id} waiting at barrier")
        self.barrier.wait()  # Wait for all workers
        
        print(f"Worker {worker_id} starting phase 2")
        # Now all workers can use self.results
        total = sum(self.results)
        print(f"Worker {worker_id} sees total: {total}")

processor = ParallelProcessor(3)
threads = [threading.Thread(target=processor.process_phase1, args=(i,)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

## 10. Thread Pool Executor
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def task(n):
    print(f"Processing {n}")
    time.sleep(1)
    return n * n

# Submit individual tasks
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(task, i) for i in range(5)]
    
    for future in as_completed(futures):
        result = future.result()
        print(f"Result: {result}")

# Map function (easier for simple cases)
with ThreadPoolExecutor(max_workers=3) as executor:
    results = executor.map(task, range(5))
    
    for result in results:
        print(f"Result: {result}")

# Real-world example: Parallel web scraping
import requests

def fetch_url(url):
    try:
        response = requests.get(url, timeout=5)
        return f"{url}: {response.status_code}"
    except Exception as e:
        return f"{url}: Error - {str(e)}"

urls = [
    "https://www.python.org",
    "https://www.github.com",
    "https://www.stackoverflow.com"
]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = executor.map(fetch_url, urls)
    
    for result in results:
        print(result)

# Handle exceptions in thread pool
def risky_task(n):
    if n == 3:
        raise ValueError(f"Error with {n}")
    return n * 2

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(risky_task, i) for i in range(5)]
    
    for future in as_completed(futures):
        try:
            result = future.result()
            print(f"Success: {result}")
        except Exception as e:
            print(f"Exception: {e}")
```

## 11. Asyncio (Async/Await) - For I/O-bound tasks
```python
import asyncio
import time

# Basic async function
async def say_hello(name, delay):
    print(f"Hello {name}, waiting {delay} seconds")
    await asyncio.sleep(delay)  # Non-blocking sleep
    print(f"Goodbye {name}")
    return f"Done with {name}"

# Run single coroutine
async def main():
    result = await say_hello("Alice", 2)
    print(result)

# asyncio.run(main())

# Run multiple coroutines concurrently
async def main_concurrent():
    results = await asyncio.gather(
        say_hello("Alice", 2),
        say_hello("Bob", 1),
        say_hello("Charlie", 3)
    )
    print(f"All results: {results}")

asyncio.run(main_concurrent())

# Create tasks
async def main_with_tasks():
    task1 = asyncio.create_task(say_hello("Alice", 2))
    task2 = asyncio.create_task(say_hello("Bob", 1))
    task3 = asyncio.create_task(say_hello("Charlie", 3))
    
    # Do other work here
    print("Tasks created, doing other work...")
    
    # Wait for tasks
    await task1
    await task2
    await task3

asyncio.run(main_with_tasks())
```

## 12. Asyncio with Real Examples
```python
import asyncio
import aiohttp  # pip install aiohttp

# Async web requests
async def fetch_url_async(session, url):
    async with session.get(url) as response:
        return f"{url}: {response.status}"

async def fetch_all():
    urls = [
        "https://www.python.org",
        "https://www.github.com",
        "https://www.stackoverflow.com"
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url_async(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        for result in results:
            print(result)

# asyncio.run(fetch_all())

# Async producer-consumer
async def producer(queue, n):
    for i in range(n):
        await asyncio.sleep(0.5)
        await queue.put(f"Item-{i}")
        print(f"Produced: Item-{i}")
    await queue.put(None)  # Signal completion

async def consumer(queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        print(f"Consumed: {item}")
        await asyncio.sleep(1)

async def main_producer_consumer():
    queue = asyncio.Queue()
    
    await asyncio.gather(
        producer(queue, 5),
        consumer(queue)
    )

asyncio.run(main_producer_consumer())
```

## 13. Asyncio Synchronization Primitives
```python
import asyncio

# Async Lock
lock = asyncio.Lock()
counter = 0

async def increment():
    global counter
    async with lock:
        temp = counter
        await asyncio.sleep(0.01)
        counter = temp + 1

async def main_lock():
    global counter
    tasks = [increment() for _ in range(100)]
    await asyncio.gather(*tasks)
    print(f"Counter: {counter}")  # 100

asyncio.run(main_lock())

# Async Event
event = asyncio.Event()

async def waiter(name):
    print(f"{name} waiting")
    await event.wait()
    print(f"{name} continuing")

async def setter():
    await asyncio.sleep(2)
    print("Setting event")
    event.set()

async def main_event():
    await asyncio.gather(
        waiter("Waiter-1"),
        waiter("Waiter-2"),
        setter()
    )

asyncio.run(main_event())

# Async Semaphore
semaphore = asyncio.Semaphore(2)

async def access_resource(worker_id):
    async with semaphore:
        print(f"Worker {worker_id} accessing")
        await asyncio.sleep(2)
        print(f"Worker {worker_id} done")

async def main_semaphore():
    tasks = [access_resource(i) for i in range(5)]
    await asyncio.gather(*tasks)

asyncio.run(main_semaphore())
```

## 14. Threading vs Multiprocessing vs Asyncio
```python
import threading
import multiprocessing
import asyncio
import time

# CPU-bound task
def cpu_bound(n):
    return sum(i * i for i in range(n))

# I/O-bound task
def io_bound():
    time.sleep(1)
    return "Done"

# Threading: Good for I/O-bound
def test_threading():
    start = time.time()
    threads = [threading.Thread(target=io_bound) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"Threading I/O: {time.time() - start:.2f}s")  # ~1s

# Multiprocessing: Good for CPU-bound
def test_multiprocessing():
    start = time.time()
    with multiprocessing.Pool(4) as pool:
        pool.map(cpu_bound, [10000000] * 4)
    print(f"Multiprocessing CPU: {time.time() - start:.2f}s")

# Asyncio: Best for I/O-bound
async def io_bound_async():
    await asyncio.sleep(1)
    return "Done"

async def test_asyncio():
    start = time.time()
    await asyncio.gather(*[io_bound_async() for _ in range(10)])
    print(f"Asyncio I/O: {time.time() - start:.2f}s")  # ~1s

if __name__ == "__main__":
    test_threading()
    test_multiprocessing()
    asyncio.run(test_asyncio())
```

## 15. Complete Example: Download Manager with Progress
```python
import threading
import time
from queue import Queue

class DownloadManager:
    def __init__(self, max_concurrent=3):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.lock = threading.Lock()
        self.completed = []
        self.failed = []
        self.progress = {}
    
    def download_file(self, file_id, url):
        with self.semaphore:
            try:
                print(f"Starting download: {file_id}")
                
                # Simulate download with progress
                for i in range(10):
                    time.sleep(0.2)
                    with self.lock:
                        self.progress[file_id] = (i + 1) * 10
                
                with self.lock:
                    self.completed.append(file_id)
                print(f"Completed: {file_id}")
                
            except Exception as e:
                with self.lock:
                    self.failed.append((file_id, str(e)))
                print(f"Failed: {file_id}")
    
    def get_status(self):
        with self.lock:
            return {
                'completed': len(self.completed),
                'failed': len(self.failed),
                'progress': self.progress.copy()
            }

# Usage
manager = DownloadManager(max_concurrent=2)
downloads = [
    ("File1", "http://example.com/file1"),
    ("File2", "http://example.com/file2"),
    ("File3", "http://example.com/file3"),
    ("File4", "http://example.com/file4")
]

threads = [
    threading.Thread(target=manager.download_file, args=download)
    for download in downloads
]

for t in threads:
    t.start()

# Monitor progress
while any(t.is_alive() for t in threads):
    status = manager.get_status()
    print(f"Status: {status}")
    time.sleep(1)

for t in threads:
    t.join()

print(f"Final status: {manager.get_status()}")
```

## Key Takeaways:

1. Threading: For I/O-bound tasks (file I/O, network)
2. Multiprocessing: For CPU-bound tasks (calculations, data processing)
3. Asyncio: For high-concurrency I/O-bound tasks (thousands of connections)
4. Lock: Protect shared resources from race conditions
5. Semaphore: Limit concurrent access to resources
6. Event: Signal between threads
7. Condition: Advanced coordination with wait/notify
8. Barrier: Synchronize multiple threads at checkpoints

Choose the right tool for your use case!
