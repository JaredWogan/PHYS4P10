





import concurrent.futures as cf

def do_something(index, i):
    print(f"Doing something with {i}")
    return index, i

res = [0 for _ in range(10)]

print(res)

with cf.ThreadPoolExecutor() as executor:
    params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    results = [
        executor.submit(do_something, i, param) 
        for i, param in enumerate(params)
    ]
    for r in cf.as_completed(results):
        print(r.result())
        res[r.result()[0]] = r.result()[1]

print(res)