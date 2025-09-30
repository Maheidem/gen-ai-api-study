"""
ReACT Agent Test: Sorting Algorithm Implementation
Demonstrates the new one-liner API for running ReACT agents.
"""

import mlflow
from local_llm_sdk import create_client_with_tools

# Configure MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("react-sorting-sdk")

# Initialize client with tools
client = create_client_with_tools(
    base_url="http://169.254.83.107:1234/v1",
    model="mistralai/magistral-small-2509"  # Explicitly use model with good tool support
)

# The sorting algorithm task
task = """
Implement a sorting algorithm with Python primitives, test it and benchmark against standard sorting algorithms.

Requirements:
1. Check if a 'sorting_algorithm' folder exists, if not create it
2. Implement bubble sort algorithm in Python using only basic primitives (no built-in sort)
3. Save the implementation as 'bubble_sort.py' in the sorting_algorithm folder
4. Create test cases to verify the algorithm works correctly
5. Benchmark it against Python's built-in sorted() function with different array sizes
6. Save benchmark results to a file
"""

# Run ReACT agent with ONE LINE!
result = client.react(
    task=task,
    max_iterations=15,
    stop_condition=lambda content: "TASK_COMPLETE" in content.upper(),
    temperature=0.7,
    verbose=True
)

# Show results
print(f"\n{'='*80}")
print(f"Final Results")
print(f"{'='*80}")
print(f"Status: {result.status.value}")
print(f"Iterations: {result.iterations}")
print(f"Success: {'✓' if result.success else '✗'}")

if result.success:
    print(f"\nFinal Response:")
    print(result.final_response[:500] + "..." if len(result.final_response) > 500 else result.final_response)

print(f"\nMetadata:")
for key, value in result.metadata.items():
    print(f"  {key}: {value}")

print(f"\n{'='*80}")
print(f"MLflow Tracking")
print(f"{'='*80}")
print(f"Experiment: react-sorting-sdk")
print(f"View traces at: http://localhost:5001")
print(f"All {result.iterations} iterations captured in single unified trace!")