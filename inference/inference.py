import torch
import time
import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from pathlib import Path


@dataclass
class InferenceScenario:
    """Inference scenario definition class"""
    input_text: str
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True

    def __repr__(self):
        return f"InferenceScenario(input_len={len(self.input_text.split())}, max_tokens={self.max_new_tokens})"

    @classmethod
    def from_file(cls,
                  file_path: str,
                  max_input_words: Optional[int] = None,
                  max_input_chars: Optional[int] = None,
                  max_new_tokens: int = 50,
                  temperature: float = 1.0,
                  top_p: float = 0.9,
                  top_k: int = 50,
                  do_sample: bool = True):
        """Create scenario from file with length control

        Args:
            file_path: Path to the input text file
            max_input_words: Maximum number of words to use from file
            max_input_chars: Maximum number of characters to use from file
            max_new_tokens: Maximum number of tokens to generate
            Other args: Same as InferenceScenario

        Returns:
            InferenceScenario instance
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Truncate by words if specified
        if max_input_words is not None:
            words = text.split()
            text = ' '.join(words[:max_input_words])

        # Truncate by characters if specified
        if max_input_chars is not None:
            text = text[:max_input_chars]

        return cls(
            input_text=text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample
        )


class LlamaInferenceBenchmark:
    """LLAMA model inference benchmark class"""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16
    ):
        """
        Args:
            model_name: Model name to use
            device: Execution device (cuda/cpu)
            dtype: Model data type
        """
        self.device = device
        self.dtype = dtype
        self.model_name = model_name

        print(f"Loading model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device if device == "cuda" else None
        )

        if device == "cpu":
            self.model = self.model.to(device)

        self.model.eval()

        # Padding token setting
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded successfully!")

    def measure_inference_latency(
        self,
        scenario: InferenceScenario
    ) -> Dict[str, float]:
        """Measure single inference latency

        Returns:
            Dict containing:
                - prefill_latency: Prefill phase latency (ms)
                - decode_latency_per_token: Average decoding latency per token (ms)
                - total_latency: Total latency (ms)
                - generated_tokens: Number of generated tokens
        """
        # Tokenize input
        inputs = self.tokenizer(
            scenario.input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        input_length = inputs['input_ids'].shape[1]

        # GPU synchronization (when using CUDA)
        if self.device == "cuda":
            torch.cuda.synchronize()

        # Prefill phase measurement
        start_prefill = time.perf_counter()

        with torch.no_grad():
            # Generate first token (prefill)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                temperature=scenario.temperature,
                top_p=scenario.top_p,
                top_k=scenario.top_k,
                do_sample=scenario.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True
            )

        if self.device == "cuda":
            torch.cuda.synchronize()

        end_prefill = time.perf_counter()
        prefill_latency = (end_prefill - start_prefill) * 1000  # ms

        # Full generation measurement (prefill + decoding)
        if self.device == "cuda":
            torch.cuda.synchronize()

        start_total = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=scenario.max_new_tokens,
                temperature=scenario.temperature,
                top_p=scenario.top_p,
                top_k=scenario.top_k,
                do_sample=scenario.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True
            )

        if self.device == "cuda":
            torch.cuda.synchronize()

        end_total = time.perf_counter()
        total_latency = (end_total - start_total) * 1000  # ms

        # Calculate generated tokens count
        generated_tokens = outputs.sequences.shape[1] - input_length

        # Calculate decoding phase latency
        decoding_latency = total_latency - prefill_latency
        decode_latency_per_token = decoding_latency / generated_tokens if generated_tokens > 0 else 0

        return {
            "prefill_latency": prefill_latency,
            "decode_latency_per_token": decode_latency_per_token,
            "total_latency": total_latency,
            "generated_tokens": generated_tokens,
            "input_tokens": input_length
        }

    def run_benchmark(
        self,
        scenarios: List[InferenceScenario],
        warmup_runs: int = 3,
        measurement_runs: int = 10,
        output_csv: str = "benchmark_results.csv"
    ) -> List[Dict]:
        """Run benchmark

        Args:
            scenarios: List of scenarios to run
            warmup_runs: Number of warmup runs
            measurement_runs: Number of measurement runs
            output_csv: CSV file path to save results

        Returns:
            List of measurement results
        """
        all_results = []

        for scenario_idx, scenario in enumerate(scenarios):
            print(f"\n{'='*60}")
            print(f"Running Scenario {scenario_idx + 1}/{len(scenarios)}: {scenario}")
            print(f"{'='*60}")

            # Warmup runs
            print(f"\nWarming up with {warmup_runs} runs...")
            for i in range(warmup_runs):
                print(f"  Warmup run {i+1}/{warmup_runs}", end="\r")
                _ = self.measure_inference_latency(scenario)
            print(f"  Warmup completed!{' '*20}")

            # Measurement runs
            print(f"\nMeasuring with {measurement_runs} runs...")
            scenario_results = []

            for i in range(measurement_runs):
                print(f"  Measurement run {i+1}/{measurement_runs}", end="\r")
                result = self.measure_inference_latency(scenario)
                result["scenario_idx"] = scenario_idx
                result["run_idx"] = i
                result["input_text"] = scenario.input_text[:100] + "..." if len(scenario.input_text) > 100 else scenario.input_text
                result["max_new_tokens"] = scenario.max_new_tokens
                scenario_results.append(result)

            print(f"  Measurement completed!{' '*20}")

            # Calculate statistics
            prefill_latencies = [r["prefill_latency"] for r in scenario_results]
            decode_latencies = [r["decode_latency_per_token"] for r in scenario_results]
            total_latencies = [r["total_latency"] for r in scenario_results]

            print(f"\nResults for Scenario {scenario_idx + 1}:")
            print(f"  Prefill Latency: {np.mean(prefill_latencies):.2f} +/- {np.std(prefill_latencies):.2f} ms")
            print(f"  Decode Latency per Token: {np.mean(decode_latencies):.2f} +/- {np.std(decode_latencies):.2f} ms")
            print(f"  Total Latency: {np.mean(total_latencies):.2f} +/- {np.std(total_latencies):.2f} ms")
            print(f"  Generated Tokens: {scenario_results[0]['generated_tokens']}")

            all_results.extend(scenario_results)

        # Save to CSV
        self.save_to_csv(all_results, output_csv)

        return all_results

    def save_to_csv(self, results: List[Dict], filename: str):
        """Save results to CSV file

        Args:
            results: List of measurement results
            filename: File name to save
        """
        if not results:
            print("No results to save!")
            return

        # Write CSV file
        fieldnames = [
            "scenario_idx", "run_idx", "input_text", "max_new_tokens",
            "input_tokens", "generated_tokens",
            "prefill_latency", "decode_latency_per_token", "total_latency"
        ]

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                writer.writerow({k: result[k] for k in fieldnames})

        print(f"\nResults saved to {filename}")


def main():
    """Main execution function"""

    # Create benchmark instance
    benchmark = LlamaInferenceBenchmark(
        model_name="meta-llama/Llama-3.2-1B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16
    )

    # Define test scenarios
    input_file_path = "/Users/anchovy-mac/Desktop/calculating/inference/input_text.txt"

    scenarios = [
        # Short input from file (50 words)
        InferenceScenario.from_file(
            file_path=input_file_path,
            max_input_words=50,
            max_new_tokens=20
        ),
        # Medium input from file (200 words)
        InferenceScenario.from_file(
            file_path=input_file_path,
            max_input_words=200,
            max_new_tokens=50
        ),
        # Long input from file (500 words)
        InferenceScenario.from_file(
            file_path=input_file_path,
            max_input_words=500,
            max_new_tokens=100
        ),
        # Using character limit instead (1000 chars)
        InferenceScenario.from_file(
            file_path=input_file_path,
            max_input_chars=1000,
            max_new_tokens=75
        ),
    ]

    # Run benchmark
    results = benchmark.run_benchmark(
        scenarios=scenarios,
        warmup_runs=3,
        measurement_runs=10,
        output_csv="llama_benchmark_results.csv"
    )

    print("\n" + "="*60)
    print("Benchmark completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()