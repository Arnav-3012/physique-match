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
# ADR-033, single source: width regions need only the front view; side-dependent ones need a depth.
WIDTH_REGIONS = ("shoulders", "upper_arm", "forearm", "thigh", "calf")
SIDE_REGIONS = ("chest", "waist", "hips")
