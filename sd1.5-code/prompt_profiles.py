"""Dataset-specific prompt templates for SD1.5 training and generation."""

from typing import List


SUPPORTED_DATASETS = {"autoarborist", "inaturalist"}


ALIASES = {
    "aa": "autoarborist",
    "auto_arborist": "autoarborist",
    "autoarborist": "autoarborist",
    "inat": "inaturalist",
    "i_nat": "inaturalist",
    "inaturalist": "inaturalist",
}


def normalize_dataset_type(dataset_type: str) -> str:
    """Normalize dataset names/aliases to canonical values."""
    if dataset_type is None:
        return "autoarborist"

    normalized = ALIASES.get(dataset_type.strip().lower(), dataset_type.strip().lower())
    if normalized not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset_type '{dataset_type}'. "
            f"Expected one of: {sorted(SUPPORTED_DATASETS)}"
        )
    return normalized


def get_training_prompt(dataset_type: str, genus: str) -> str:
    """Return a single training prompt aligned to the dataset image style."""
    dataset_type = normalize_dataset_type(dataset_type)

    if dataset_type == "inaturalist":
        return (
            f"a real-world iNaturalist field photograph of a tree, genus {genus}, "
            "outdoors in natural light with diagnostic botanical features visible "
            "such as leaves, branching structure, bark, flowers, or fruit"
        )

    return (
        f"a street-level Google Street View style photograph of a tree, genus {genus}, "
        "seen in an urban or suburban roadside context with sidewalk, street, "
        "or nearby buildings"
    )


def get_generation_prompts(dataset_type: str, genus: str) -> List[str]:
    """Return diverse generation prompts aligned to the dataset domain."""
    dataset_type = normalize_dataset_type(dataset_type)

    if dataset_type == "inaturalist":
        return [
            (
                f"A real-world iNaturalist field photograph of a {genus} tree in a natural habitat, "
                "captured in daylight with visible leaves, branching habit, and bark texture. "
                "Unstaged ecological context."
            ),
            (
                f"A naturalistic observation photo of genus {genus} outdoors, showing diagnostic "
                "tree morphology, including canopy structure and trunk bark, with realistic field "
                "background vegetation."
            ),
            (
                f"A close but realistic field image of a {genus} tree with identifiable botanical "
                "traits such as leaves and bark, photographed like an iNaturalist submission in "
                "ambient outdoor lighting."
            ),
        ]

    return [
        (
            f"A street-level Google Street View style photograph of a mature {genus} tree on a "
            "residential street with sidewalk, parked cars, and nearby homes in daylight."
        ),
        (
            f"Wide-angle roadway scene featuring a {genus} tree growing beside an urban sidewalk, "
            "captured from a street-view perspective with buildings in the background."
        ),
        (
            f"Suburban streetscape photo of genus {genus}, with the tree centered near curbside, "
            "surrounded by pavement, lawn edges, and neighborhood houses in natural light."
        ),
    ]


def get_negative_prompt(dataset_type: str) -> str:
    """Negative prompts to keep output photorealistic and dataset-consistent."""
    _ = normalize_dataset_type(dataset_type)
    return (
        "Illustration, drawing, painting, sketch, cartoon, anime, 3d render, cgi, "
        "studio lighting, product photo, surreal, abstract, text, watermark, logo"
    )
