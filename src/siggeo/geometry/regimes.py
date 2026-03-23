from __future__ import annotations


REGIME_ORDER = (
    "aligned",
    "inverse",
    "orthogonal",
    "asymmetric_ORF",
    "asymmetric_CRISPR",
    "inactive",
)


def assign_regime(
    *,
    orf_active: bool,
    crispr_active: bool,
    sign_z: float,
    delta_activity: float,
    sign_z_threshold: float,
    asymmetry_threshold: float,
) -> str:
    if not orf_active and not crispr_active:
        return "inactive"
    if orf_active and (not crispr_active or delta_activity > asymmetry_threshold):
        return "asymmetric_ORF"
    if crispr_active and (not orf_active or delta_activity < -asymmetry_threshold):
        return "asymmetric_CRISPR"
    if sign_z >= sign_z_threshold:
        return "aligned"
    if sign_z <= -sign_z_threshold:
        return "inverse"
    return "orthogonal"
