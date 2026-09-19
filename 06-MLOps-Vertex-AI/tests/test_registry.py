from aml_detection.registry import _label, _parse_gcs


def test_parse_gcs_splits_bucket_and_prefix():
    assert _parse_gcs("gs://bucket/a/b/c") == ("bucket", "a/b/c")
    assert _parse_gcs("gs://bucket") == ("bucket", "")
    assert _parse_gcs("gs://bucket/model/") == ("bucket", "model")


def test_label_formats_metric_as_valid_gcp_label():
    label = _label(0.0927361835693311)
    assert label == "0_093"
    assert "." not in label
    assert label.islower() or label.replace("_", "").isdigit()
