"""Unit tests for configuration loading and validation."""

import os
import pytest
from src.config import AppConfig, load_config


def test_load_default_config():
    config_file = "configs/default_highway.yaml"
    assert os.path.exists(config_file), "Default configuration YAML missing"

    cfg = load_config(config_file)
    assert isinstance(cfg, AppConfig)
    assert cfg.model.confidence_threshold == 0.30
    assert "car" in cfg.model.class_names.values()
    assert len(cfg.lanes) == 8  # 4 left + 4 right
    assert "left_gate" in cfg.counting_gates
    assert "right_gate" in cfg.counting_gates


def test_load_fallback_config_when_file_not_found():
    cfg = load_config("non_existent_config.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.model.weights == "yolov8n.pt"
    assert cfg.tracker.max_age == 25


def test_fast_config():
    config_file = "configs/highway_fast.yaml"
    assert os.path.exists(config_file)
    cfg = load_config(config_file)
    assert cfg.model.confidence_threshold == 0.40
    assert cfg.tracker.max_age == 15
