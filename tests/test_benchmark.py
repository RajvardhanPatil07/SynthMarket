from synthmarket.benchmark import BenchmarkResult, summarize_results


def test_benchmark_summary() -> None:
    results = [
        BenchmarkResult("wgan-gp", 1, {"ks": 0.2}),
        BenchmarkResult("wgan-gp", 2, {"ks": 0.4}),
    ]
    summary = summarize_results(results)
    assert summary[0]["model"] == "wgan-gp"
    assert summary[0]["runs"] == 2
    assert summary[0]["metrics"]["ks"]["mean"] == 0.30000000000000004
