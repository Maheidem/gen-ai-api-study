def bubble_sort(arr):
    n = len(arr)
    # Make a copy to avoid modifying the original
    arr_copy = arr.copy()
    for i in range(n):
        # Flag to check if any swap happened in this pass
        swapped = False
        for j in range(0, n-i-1):
            if arr_copy[j] > arr_copy[j+1]:
                # Swap the elements
                arr_copy[j], arr_copy[j+1] = arr_copy[j+1], arr_copy[j]
                swapped = True
        # If no swaps happened, array is already sorted
        if not swapped:
            break
    return arr_copy

# Test cases to verify correctness
def test_bubble_sort():
    test_cases = [
        ([], []),
        ([1], [1]),
        ([3, 2, 1], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 3]),  # Already sorted
        ([5, 2, 7, 3, 4, 6, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9]),
        ([1, 1, 1], [1, 1, 1]),      # With duplicates
    ]

    for i, (input_list, expected) in enumerate(test_cases):
        result = bubble_sort(input_list)
        assert result == expected, f'Test case {i+1} failed: {result} != {expected}'
    print("All test cases passed!")

# Benchmarking function
def benchmark():
    import time
    import random
    results = []
    sizes = [10, 100, 1000, 10000]
    
    for size in sizes:
        # Create a random list of the given size
        test_list = [random.randint(0, 1000) for _ in range(size)]
        
        # Time our bubble sort
        start_time = time.time()
        sorted_bubble = bubble_sort(test_list.copy())
        bubble_time = time.time() - start_time
        
        # Time built-in sorted()
        start_time = time.time()
        sorted_built_in = sorted(test_list)
        built_in_time = time.time() - start_time
        
        results.append({
            'size': size,
            'bubble_sort_time': bubble_time,
            'sorted_time': built_in_time
        })
    
    return results
