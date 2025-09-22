"""
Example script showing how to use file input with different length controls
"""

from inference import LlamaInferenceBenchmark, InferenceScenario
import torch


def run_file_based_benchmark():
    """Run benchmark using text from file with different length settings"""

    # Path to input file
    input_file = "/Users/anchovy-mac/Desktop/calculating/inference/input_text.txt"

    # Create benchmark instance
    benchmark = LlamaInferenceBenchmark(
        model_name="meta-llama/Llama-3.2-1B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16
    )

    # Create scenarios with different input lengths from the same file
    scenarios = [
        # Very short input (10 words)
        InferenceScenario.from_file(
            file_path=input_file,
            max_input_words=10,
            max_new_tokens=10
        ),

        # Short input (50 words)
        InferenceScenario.from_file(
            file_path=input_file,
            max_input_words=50,
            max_new_tokens=20
        ),

        # Medium input (100 words)
        InferenceScenario.from_file(
            file_path=input_file,
            max_input_words=100,
            max_new_tokens=30
        ),

        # Longer input (250 words)
        InferenceScenario.from_file(
            file_path=input_file,
            max_input_words=250,
            max_new_tokens=50
        ),

        # Very long input (500 words)
        InferenceScenario.from_file(
            file_path=input_file,
            max_input_words=500,
            max_new_tokens=100
        ),

        # Using character limit (500 characters)
        InferenceScenario.from_file(
            file_path=input_file,
            max_input_chars=500,
            max_new_tokens=25
        ),

        # Using character limit (2000 characters)
        InferenceScenario.from_file(
            file_path=input_file,
            max_input_chars=2000,
            max_new_tokens=75
        ),
    ]

    # Run benchmark
    results = benchmark.run_benchmark(
        scenarios=scenarios,
        warmup_runs=3,
        measurement_runs=10,
        output_csv="file_based_benchmark_results.csv"
    )

    # Print summary
    print("\n" + "="*60)
    print("FILE-BASED BENCHMARK SUMMARY")
    print("="*60)

    for idx, scenario in enumerate(scenarios):
        input_words = len(scenario.input_text.split())
        input_chars = len(scenario.input_text)
        print(f"\nScenario {idx + 1}:")
        print(f"  Input: {input_words} words, {input_chars} characters")
        print(f"  Max new tokens: {scenario.max_new_tokens}")


def run_custom_file_benchmark():
    """Example of using custom parameters with file input"""

    input_file = "/Users/anchovy-mac/Desktop/calculating/inference/input_text.txt"

    benchmark = LlamaInferenceBenchmark(
        model_name="meta-llama/Llama-3.2-1B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16
    )

    # Custom scenario with specific generation parameters
    custom_scenarios = [
        # Deterministic generation (greedy decoding)
        InferenceScenario.from_file(
            file_path=input_file,
            max_input_words=100,
            max_new_tokens=50,
            do_sample=False  # Greedy decoding
        ),

        # High temperature for creative output
        InferenceScenario.from_file(
            file_path=input_file,
            max_input_words=100,
            max_new_tokens=50,
            temperature=1.5,
            top_p=0.95,
            do_sample=True
        ),

        # Low temperature for focused output
        InferenceScenario.from_file(
            file_path=input_file,
            max_input_words=100,
            max_new_tokens=50,
            temperature=0.3,
            top_k=10,
            do_sample=True
        ),
    ]

    results = benchmark.run_benchmark(
        scenarios=custom_scenarios,
        warmup_runs=2,
        measurement_runs=5,
        output_csv="custom_file_benchmark_results.csv"
    )


def interactive_file_benchmark():
    """Interactive mode where user can specify input length"""

    input_file = "/Users/anchovy-mac/Desktop/calculating/inference/input_text.txt"

    print("Interactive File-based Benchmark")
    print("="*40)

    # Get user input
    try:
        max_words = int(input("Enter maximum input words (e.g., 100): "))
        max_tokens = int(input("Enter maximum tokens to generate (e.g., 50): "))
        num_runs = int(input("Enter number of measurement runs (e.g., 10): "))
    except ValueError:
        print("Invalid input. Using defaults: 100 words, 50 tokens, 10 runs")
        max_words = 100
        max_tokens = 50
        num_runs = 10

    # Create benchmark
    benchmark = LlamaInferenceBenchmark(
        model_name="meta-llama/Llama-3.2-1B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16
    )

    # Create scenario
    scenario = InferenceScenario.from_file(
        file_path=input_file,
        max_input_words=max_words,
        max_new_tokens=max_tokens
    )

    print(f"\nRunning benchmark with:")
    print(f"  Input: {len(scenario.input_text.split())} words")
    print(f"  Max new tokens: {max_tokens}")
    print(f"  Measurement runs: {num_runs}")

    # Run benchmark
    results = benchmark.run_benchmark(
        scenarios=[scenario],
        warmup_runs=3,
        measurement_runs=num_runs,
        output_csv=f"interactive_benchmark_{max_words}w_{max_tokens}t.csv"
    )


if __name__ == "__main__":
    print("Select benchmark mode:")
    print("1. Standard file-based benchmark (various lengths)")
    print("2. Custom parameters benchmark")
    print("3. Interactive mode")

    choice = input("\nEnter choice (1-3): ")

    if choice == "1":
        run_file_based_benchmark()
    elif choice == "2":
        run_custom_file_benchmark()
    elif choice == "3":
        interactive_file_benchmark()
    else:
        print("Invalid choice. Running standard benchmark...")
        run_file_based_benchmark()