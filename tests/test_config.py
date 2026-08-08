"""Tests for environment-driven configuration and demo-mode fallback logic."""
from __future__ import annotations

from fraudlens import config


def test_settings_not_configured_when_password_missing(monkeypatch):
    monkeypatch.setenv("COGNODB_URI", "bolt+s://example.databases.cognodb.com:7687")
    monkeypatch.delenv("COGNODB_PASSWORD", raising=False)
    assert config.get_cognodb_settings().is_configured is False


def test_settings_configured_with_all_fields(monkeypatch):
    monkeypatch.setenv("COGNODB_URI", "bolt+s://example.databases.cognodb.com:7687")
    monkeypatch.setenv("COGNODB_USER", "cognodb")
    monkeypatch.setenv("COGNODB_PASSWORD", "secret")
    assert config.get_cognodb_settings().is_configured is True


def test_demo_mode_activates_automatically_when_not_configured(monkeypatch):
    monkeypatch.delenv("COGNODB_URI", raising=False)
    monkeypatch.delenv("COGNODB_PASSWORD", raising=False)
    monkeypatch.setenv("FRAUDLENS_DEMO_MODE", "false")
    assert config.is_demo_mode() is True


def test_demo_mode_can_be_forced_even_when_configured(monkeypatch):
    monkeypatch.setenv("COGNODB_URI", "bolt+s://example.databases.cognodb.com:7687")
    monkeypatch.setenv("COGNODB_USER", "cognodb")
    monkeypatch.setenv("COGNODB_PASSWORD", "secret")
    monkeypatch.setenv("FRAUDLENS_DEMO_MODE", "true")
    assert config.is_demo_mode() is True


def test_demo_mode_off_when_configured_and_not_forced(monkeypatch):
    monkeypatch.setenv("COGNODB_URI", "bolt+s://example.databases.cognodb.com:7687")
    monkeypatch.setenv("COGNODB_USER", "cognodb")
    monkeypatch.setenv("COGNODB_PASSWORD", "secret")
    monkeypatch.setenv("FRAUDLENS_DEMO_MODE", "false")
    assert config.is_demo_mode() is False
