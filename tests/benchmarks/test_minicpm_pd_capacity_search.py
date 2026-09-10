from benchmarks.minicpmo.pd_capacity_search import archive_source, context_coverage, next_candidate, source_manifest


def choose(rows):
    return next_candidate(rows, first=8, step=8, resolution=2, maximum=48)


def test_search_large_steps_then_refines():
    assert choose([]) == 8
    rows = [{"users": 8, "input_capacity_pass": True}]
    assert choose(rows) == 16
    rows.append({"users": 16, "input_capacity_pass": False})
    assert choose(rows) == 12
    rows.append({"users": 12, "input_capacity_pass": True})
    assert choose(rows) == 14
    rows.append({"users": 14, "input_capacity_pass": False})
    assert choose(rows) is None


def test_initial_failure_searches_down():
    assert choose([{"users": 8, "input_capacity_pass": False}]) == 4


def test_installed_package_source_snapshot(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "module.py").write_text("x = 1\n")
    before = archive_source(package, tmp_path / "snapshot")
    assert before == source_manifest(package)
    assert (tmp_path / "snapshot/package-source.tar.gz").is_file()
    (package / "module.py").write_text("x = 2\n")
    assert before != source_manifest(package)


def test_context_coverage_requires_every_session():
    def user(name, prompts):
        return {
            "session_id": name,
            "pd_completion_witness": {
                "records": [{"input_unit_index": i, "prompt_tokens": prompt} for i, prompt in enumerate(prompts)]
            },
        }

    assert context_coverage({"users": [user("a", [35000, 400])]})["every_session_reset"]
    assert not context_coverage({"users": [user("a", [35000, 400]), user("b", [100, 400])]})["every_session_reset"]


def test_d_talker_placement_keeps_fixed_kv_budgets():
    from pathlib import Path

    from vllm_omni.config.stage_config import resolve_deploy_yaml

    folder = Path(__file__).resolve().parents[2] / "benchmarks/minicpmo"
    base = resolve_deploy_yaml(folder / "deploy_capacity_pd_4gpu_fp8.yaml")
    result = resolve_deploy_yaml(folder / "deploy_capacity_pd_d_talker_fp8.yaml")
    assert [s["devices"] for s in result["stages"]] == ["0", "1", "1", "3"]
    for before, after in zip(base["stages"], result["stages"], strict=True):
        expected = dict(before)
        if expected["stage_id"] == 2:
            expected.update(devices="1", gpu_memory_utilization=0.1)
        assert after == expected
