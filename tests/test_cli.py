import json

import numpy as np
import pandas as pd

from synthmarket.cli import main


def _csv(tmp_path):
    rng = np.random.default_rng(9)
    index = pd.date_range("2021-01-01", periods=300, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(index))))
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e6},
        index=index,
    ).to_csv(path)
    return path


def test_models_lists_baselines(capsys) -> None:
    assert main(["models"]) == 0
    names = {entry["name"] for entry in json.loads(capsys.readouterr().out)["models"]}
    assert {"block-bootstrap", "garch"} <= names


def test_generate_evaluate_split_and_validate(tmp_path, capsys) -> None:
    source = _csv(tmp_path)
    output = tmp_path / "synthetic.csv"
    assert main(["generate", "--csv", str(source), "--paths", "3", "--length", "40", "--seed", "1", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["paths"] == 3
    generated = pd.read_csv(output, index_col=0)
    assert generated.shape == (40, 3)
    assert (generated > 0).all().all()

    assert main(["evaluate", "--csv", str(source)]) == 0
    assert "tail_risk" in json.loads(capsys.readouterr().out)
    assert main(["split", "--csv", str(source), "--min-train-rows", "50"]) == 0
    split = json.loads(capsys.readouterr().out)
    assert split["train"]["end"] < split["test"]["start"]
    assert main(["validate", "--csv", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["is_valid"] is True


def test_unknown_model_is_a_clean_error(tmp_path, capsys) -> None:
    assert main(["generate", "--csv", str(_csv(tmp_path)), "--model", "missing"]) == 1
    assert "error:" in capsys.readouterr().err
