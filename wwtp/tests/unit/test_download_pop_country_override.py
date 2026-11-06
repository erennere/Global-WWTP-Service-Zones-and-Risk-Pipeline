"""Pins the exact semantics of download_pop's --country selector: unset
behaves exactly as before (every country, country_limit still applies),
--country selects exactly one country and takes precedence over
country_limit, and merged_output_path matches what process_single_country
actually writes to.
"""

import argparse

import pytest

from wwtp import download_pop


pytestmark = pytest.mark.unit


FAKE_COUNTRY_URLS = {
    "deu": ["https://example.test/deu.tif"],
    "usa": ["https://example.test/usa.tif"],
    "zaf": ["https://example.test/zaf.tif"],
}


def _fake_args(country=None):
    return argparse.Namespace(
        level=None,
        version=None,
        buffer=None,
        weight_method=None,
        weight_func=None,
        dynamic_buffering=None,
        size_measure=None,
        country=country,
    )


def _stub_main_dependencies(monkeypatch, cfg, args):
    monkeypatch.setattr(download_pop, "parse_args", lambda: args)
    monkeypatch.setattr(download_pop, "load_config", lambda script_name, **overrides: cfg)
    monkeypatch.setattr(
        download_pop,
        "get_urls",
        lambda **kwargs: dict(FAKE_COUNTRY_URLS),
    )

    captured = {}

    def fake_process_all_countries(country_urls, res, max_workers, data_dir):
        captured["country_urls"] = country_urls

    monkeypatch.setattr(download_pop, "process_all_countries", fake_process_all_countries)
    return captured


def _base_cfg(country_limit=0):
    return {
        "paths": {"pop_dir": "/tmp/pop"},
        "start_year": 2015,
        "end_year": 2016,
        "worldpop_2014_url_template": "x",
        "include_2014": False,
        "worldpop_yearly_url_template": "y",
        "country_limit": country_limit,
    }


def test_main_default_processes_all_countries_when_no_country_flag(monkeypatch):
    captured = _stub_main_dependencies(monkeypatch, _base_cfg(), _fake_args(country=None))

    download_pop.main()

    assert captured["country_urls"] == FAKE_COUNTRY_URLS


def test_main_country_flag_filters_to_exactly_one_country(monkeypatch):
    captured = _stub_main_dependencies(monkeypatch, _base_cfg(), _fake_args(country="deu"))

    download_pop.main()

    assert captured["country_urls"] == {"deu": FAKE_COUNTRY_URLS["deu"]}


def test_main_country_flag_is_case_insensitive(monkeypatch):
    captured = _stub_main_dependencies(monkeypatch, _base_cfg(), _fake_args(country="DEU"))

    download_pop.main()

    assert captured["country_urls"] == {"deu": FAKE_COUNTRY_URLS["deu"]}


def test_main_unknown_country_flag_raises_clear_error(monkeypatch):
    _stub_main_dependencies(monkeypatch, _base_cfg(), _fake_args(country="zzz"))

    with pytest.raises(ValueError, match="zzz"):
        download_pop.main()


def test_country_flag_takes_precedence_over_country_limit(monkeypatch):
    # country_limit=1 would alphabetically keep only "deu" - request "zaf",
    # which sorts outside that window, and confirm it's still selected.
    captured = _stub_main_dependencies(monkeypatch, _base_cfg(country_limit=1), _fake_args(country="zaf"))

    download_pop.main()

    assert captured["country_urls"] == {"zaf": FAKE_COUNTRY_URLS["zaf"]}


def test_country_limit_still_applies_when_no_country_flag(monkeypatch):
    captured = _stub_main_dependencies(monkeypatch, _base_cfg(country_limit=1), _fake_args(country=None))

    download_pop.main()

    # sorted(FAKE_COUNTRY_URLS) == ["deu", "usa", "zaf"]; limit=1 keeps "deu".
    assert captured["country_urls"] == {"deu": FAKE_COUNTRY_URLS["deu"]}


def test_merged_output_path_matches_process_single_country_output(monkeypatch, tmp_path):
    data_dir = tmp_path / "population"

    monkeypatch.setattr(
        download_pop,
        "download_save_and_unzip_pops",
        lambda country_urls, country, data_dir: str(tmp_path / "extracted"),
    )
    monkeypatch.setattr(
        download_pop,
        "find_files",
        lambda extract_folder: (["fake.tif"], True),
    )

    written_to = {}

    def fake_mosaic(result, merged_path):
        written_to["path"] = merged_path

    monkeypatch.setattr(download_pop, "mosaic_large_rasters", fake_mosaic)

    download_pop.process_single_country({"deu": FAKE_COUNTRY_URLS["deu"]}, "deu", data_dir=str(data_dir))

    assert written_to["path"] == download_pop.merged_output_path(str(data_dir), "deu")
