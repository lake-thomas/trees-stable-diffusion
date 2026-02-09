"""Tests for generation utilities."""

from trees_sd.generation import normalize_dataset_type, get_generation_prompts, get_negative_prompt


def test_dataset_type_aliases():
    assert normalize_dataset_type("inat") == "inaturalist"
    assert normalize_dataset_type("aa") == "autoarborist"


def test_generation_prompts_are_dataset_specific():
    inat_prompts = get_generation_prompts("inaturalist", "Quercus")
    aa_prompts = get_generation_prompts("autoarborist", "Quercus")

    assert len(inat_prompts) >= 1
    assert len(aa_prompts) >= 1
    assert inat_prompts != aa_prompts


def test_negative_prompt_available():
    assert "Illustration" in get_negative_prompt("inaturalist")
