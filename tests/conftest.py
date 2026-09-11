from __future__ import annotations

from pathlib import Path

import pytest

from fridge_watcher.config import Config, TriggerMode


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        frigate_url="http://frigate.local:5000",
        frigate_camera_name="fridge",
        mqtt_host="mqtt.local",
        mqtt_port=1883,
        mqtt_user=None,
        mqtt_password=None,
        trigger_mode=TriggerMode.EVENTS,
        zone_name="fridge_door",
        frame_count=6,
        confidence_threshold=0.75,
        anthropic_api_key="test-key",
        anthropic_model="claude-sonnet-4-6",
        supabase_url="https://test.supabase.co",
        supabase_service_key="service-key",
        captures_dir=tmp_path / "captures",
    )
