"""BodyM measurement column -> `physique_schemas` Region. Only 8 of 10 regions have a label;
`upper_back` and `lats` are geometric-only (back view) and sit outside the rho gate (ADR-024)."""

REGION_COLUMN = {
    "shoulders": "shoulder_breadth",
    "chest": "chest",
    "waist": "waist",
    "hips": "hip",
    "upper_arm": "bicep",
    "forearm": "forearm",
    "thigh": "thigh",
    "calf": "calf",
}
LABELLED_REGIONS = tuple(REGION_COLUMN)
