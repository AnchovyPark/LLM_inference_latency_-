"""
사용 예제: LLAMA 3.2 1B 모델 추론 벤치마크

이 스크립트는 inference.py의 클래스들을 사용하여
다양한 시나리오에서 추론 성능을 측정하는 예제입니다.
"""

from inference import LlamaInferenceBenchmark, InferenceScenario
import torch


def run_custom_benchmark():
    """커스텀 시나리오로 벤치마크 실행"""

    # 1. 벤치마크 인스턴스 생성
    benchmark = LlamaInferenceBenchmark(
        model_name="meta-llama/Llama-3.2-1B",  # 모델 이름
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16  # 또는 torch.bfloat16
    )

    # 2. 커스텀 시나리오 정의
    custom_scenarios = [
        # 시나리오 1: 매우 짧은 입출력
        InferenceScenario(
            input_text="Hello",
            max_new_tokens=5,
            temperature=0.7,
            do_sample=True
        ),

        # 시나리오 2: 질문-답변
        InferenceScenario(
            input_text="What are the benefits of renewable energy?",
            max_new_tokens=30,
            temperature=0.8,
            top_p=0.95
        ),

        # 시나리오 3: 코드 생성
        InferenceScenario(
            input_text="Write a Python function to calculate fibonacci numbers:",
            max_new_tokens=75,
            temperature=0.2,  # 낮은 temperature로 더 결정적인 출력
            do_sample=True
        ),

        # 시나리오 4: 스토리텔링
        InferenceScenario(
            input_text="Once upon a time in a distant galaxy,",
            max_new_tokens=100,
            temperature=1.0,
            top_p=0.9,
            top_k=50
        ),
    ]

    # 3. 벤치마크 실행
    results = benchmark.run_benchmark(
        scenarios=custom_scenarios,
        warmup_runs=3,  # 워밍업 3회
        measurement_runs=10,  # 측정 10회
        output_csv="custom_benchmark_results.csv"
    )

    return results


def run_stress_test():
    """스트레스 테스트: 다양한 길이의 입출력 조합"""

    benchmark = LlamaInferenceBenchmark(
        model_name="meta-llama/Llama-3.2-1B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16
    )

    # 다양한 길이 조합 테스트
    stress_scenarios = []

    # 입력 길이별 테스트
    input_lengths = [10, 50, 100, 200]  # 단어 수 기준
    output_lengths = [10, 25, 50, 100]  # 토큰 수

    for input_len in input_lengths:
        for output_len in output_lengths:
            # 입력 텍스트 생성 (단순 반복)
            input_text = " ".join(["The quick brown fox jumps over the lazy dog."] * (input_len // 9))

            scenario = InferenceScenario(
                input_text=input_text,
                max_new_tokens=output_len,
                temperature=0.7,
                do_sample=True
            )
            stress_scenarios.append(scenario)

    # 스트레스 테스트 실행
    results = benchmark.run_benchmark(
        scenarios=stress_scenarios,
        warmup_runs=2,  # 빠른 테스트를 위해 워밍업 줄임
        measurement_runs=5,  # 측정 횟수도 줄임
        output_csv="stress_test_results.csv"
    )

    return results


def run_deterministic_test():
    """결정적 출력 테스트 (do_sample=False)"""

    benchmark = LlamaInferenceBenchmark(
        model_name="meta-llama/Llama-3.2-1B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16
    )

    # 결정적 시나리오 (sampling 없이)
    deterministic_scenarios = [
        InferenceScenario(
            input_text="The capital of France is",
            max_new_tokens=10,
            do_sample=False  # Greedy decoding
        ),
        InferenceScenario(
            input_text="1 + 1 equals",
            max_new_tokens=5,
            do_sample=False
        ),
    ]

    results = benchmark.run_benchmark(
        scenarios=deterministic_scenarios,
        warmup_runs=3,
        measurement_runs=10,
        output_csv="deterministic_test_results.csv"
    )

    return results


if __name__ == "__main__":
    print("="*60)
    print("LLAMA 3.2 1B Inference Benchmark Examples")
    print("="*60)

    # 어떤 테스트를 실행할지 선택
    test_mode = input("\nSelect test mode:\n1. Custom scenarios\n2. Stress test\n3. Deterministic test\n4. All tests\nEnter choice (1-4): ")

    if test_mode == "1" or test_mode == "4":
        print("\n>>> Running custom benchmark...")
        run_custom_benchmark()

    if test_mode == "2" or test_mode == "4":
        print("\n>>> Running stress test...")
        run_stress_test()

    if test_mode == "3" or test_mode == "4":
        print("\n>>> Running deterministic test...")
        run_deterministic_test()

    print("\n" + "="*60)
    print("All tests completed!")
    print("Check the generated CSV files for detailed results.")
    print("="*60)