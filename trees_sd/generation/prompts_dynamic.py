"""Prompt utilities for dataset-aware image generation with diversity injection."""

import random
from typing import List, Optional

SUPPORTED_DATASETS = {"autoarborist", "inaturalist"}

ALIASES = {
    "aa": "autoarborist",
    "auto_arborist": "autoarborist",
    "autoarborist": "autoarborist",
    "inat": "inaturalist",
    "i_nat": "inaturalist",
    "inaturalist": "inaturalist",
}


# ---------------------------------------------------------------------
# Vocabulary Pools
# ---------------------------------------------------------------------

SEASONS = [
    "spring", 
    "summer", 
    "autumn", 
    "winter"]

LIGHTING = [
    "soft morning light",
    "bright midday sunlight",
    "overcast sky",
    "dappled forest light",
    "golden hour lighting",
    "diffuse cloudy daylight",
]

CAMERA_ANGLES = [
    "eye-level perspective",
    "slight upward angle",
    "slight downward angle",
    "wide-angle lens perspective",
    "off-center framing",
]

TREE_STAGES = [
    "young sapling",
    "mature tree",
    "large old-growth tree",
    "recently pruned tree",
    "asymmetrical canopy",
]

FRAMING = [
    "full canopy visible",
    "partial canopy view",
    "close-up of bark texture",
    "leaves clearly visible",
    "branching structure visible",
]

GEOGRAPHY = [
    "Pacific Northwest US Region",
    "Northeastern US Region"
    "Southeastern US Region",
    "Midwestern US Region",
    "Southwestern US Region",
    "Rocky Mountain Region"

]

URBAN_CONTEXT = [
    "residential street with sidewalks",
    "suburban neighborhood with houses",
    "parked cars along curbside",
    "city buildings in background",
    "urban sidewalk and pavement",
    "residential park"
]

NATURAL_CONTEXT = [
    "forest understory vegetation",
    "natural woodland habitat",
    "mixed deciduous forest",
    "grassy meadow edge",
    "riparian woodland",
]


# ---------------------------------------------------------------------
# Dataset normalization
# ---------------------------------------------------------------------

def normalize_dataset_type(dataset_type: str) -> str:
    if dataset_type is None:
        return "autoarborist"

    normalized = ALIASES.get(dataset_type.strip().lower(), dataset_type.strip().lower())
    if normalized not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset_type '{dataset_type}'. "
            f"Expected one of: {sorted(SUPPORTED_DATASETS)}"
        )
    return normalized


# ---------------------------------------------------------------------
# Dynamic Prompt Builder
# ---------------------------------------------------------------------

def _sample(rng: random.Random, pool: List[str]) -> str:
    return rng.choice(pool)


def build_prompt(
    dataset_type: str,
    genus: str,
    seed: Optional[int] = None,
) -> str:
    """
    Build a single diverse prompt using structured vocabulary sampling.
    """

    dataset_type = normalize_dataset_type(dataset_type)
    rng = random.Random(seed)

    season = _sample(rng, SEASONS)
    lighting = _sample(rng, LIGHTING)
    angle = _sample(rng, CAMERA_ANGLES)
    stage = _sample(rng, TREE_STAGES)
    framing = _sample(rng, FRAMING)
    geography = _sample(rng, GEOGRAPHY)

    if dataset_type == "inaturalist":
        context = _sample(rng, NATURAL_CONTEXT)

        return (
            f"A real-world iNaturalist field photograph of a {stage} {genus} tree "
            f"in {season}, in a {context}, captured in {lighting}, "
            f"with {framing}, photographed from an {angle}. "
            f"Image captured from the {geography}, documentary ecological style, realistic colors."
        )

    # autoarborist / street view
    context = _sample(rng, URBAN_CONTEXT)

    return (
        f"A street-level Google Street View style photograph of a {stage} {genus} tree "
        f"during {season}, located along a {context}, captured in {lighting}, "
        f"with {framing}, photographed from an {angle}. "
        f"Image captured from the {geography}, municipal inventory style, realistic urban environment."
    )


def generate_prompt_batch(
    dataset_type: str,
    genus: str,
    num_prompts: int,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Generate multiple diverse prompts.
    """

    rng = random.Random(seed)
    prompts = []

    for i in range(num_prompts):
        prompts.append(
            build_prompt(
                dataset_type=dataset_type,
                genus=genus,
                seed=rng.randint(0, 1_000_000),
            )
        )

    return prompts


def get_negative_prompt(dataset_type: str) -> str:
    _ = normalize_dataset_type(dataset_type)
    return (
        "illustration, drawing, painting, sketch, cartoon, anime, 3d render, cgi, "
        "studio lighting, product photography, surreal, abstract, "
        "text, watermark, logo, caption"
    )