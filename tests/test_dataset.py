from adaptive_plasticity.environment_validation import minari_api_report


def test_minari_api_is_available_without_download() -> None:
    report = minari_api_report()
    assert "list_local_datasets" in report["public_api"]
    assert "load_dataset" in report["public_api"]
    assert report["download_performed"] is False
    assert report["local_dataset_count"] >= 0
