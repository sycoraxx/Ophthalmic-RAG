"""
ophthalmic_taxonomy.py — Hierarchical Ophthalmic Disease Taxonomy
─────────────────────────────────────────────────────────────────
Maps ophthalmic labels into clinically meaningful supergroups
for two-stage hierarchical classification.

Supports BOTH label formats:
  - Modality-prefixed (curated):  "OCT, drusen", "slit lamp, corneal ulcer"
  - Plain ICD-style (legacy):     "diabetic retinopathy", "corneal ulcer"

Each supergroup is tagged with an imaging **modality**:
  - "anterior"  → visible from slit-lamp / external photos
  - "posterior" → visible from fundus / retinal imaging
  - "both"      → relevant to either modality
  - "external"  → visible from external eye photos

Labels are auto-assigned to a supergroup via keyword matching.
Order of rules matters — more specific keywords are checked first.
Unmatched labels fall into UNCATEGORIZED_GROUP and are always included.
"""

from __future__ import annotations

# ─── Modality Definitions ─────────────────────────────────────────────────────
# Prompts use the EXACT imaging modality names EyeCLIP was pre-trained on
# (from the paper: CFP, OCT, FFA, ICGA, FAF, Slit Lamp, OUS, Specular
# Microscopy, External Eye Photography, Corneal Photography, RetCam).
# This maximises cosine-similarity separation despite the narrow dynamic
# range (~0.06) of the fine-tuned text encoder.

MODALITY_PROMPT_GROUPS: dict[str, list[str]] = {
    "posterior": [
        "color fundus photography",
        "fundus fluorescein angiography",
        "fundus autofluorescence",
        "indocyanine green angiography",
        "retinal imaging",
        "optical coherence tomography of the retina",
        "RetCam imaging of the retina",
    ],
    "anterior": [
        "slit lamp photography",
        "anterior segment photograph",
        "corneal photography",
        "specular microscopy",
        "anterior segment optical coherence tomography",
        "slit lamp biomicroscopy showing cornea and iris",
        "gonioscopy of the anterior chamber angle",
    ],
    "external": [
        "external eye photography",
        "clinical photograph of eyelids",
        "periocular photograph",
        "photograph of the face and eye",
        "external photograph of the eye and adnexa",
    ],
}

# Canonical modality names (order matches the prompt groups)
MODALITY_NAMES: list[str] = list(MODALITY_PROMPT_GROUPS.keys())

# Minimum modality confidence to apply filtering (below this, allow all groups).
# Lowered from 0.45 to 0.38 because EyeCLIP's fine-tuned text encoder has a
# narrow cosine-similarity dynamic range (~0.06 across modalities).
MODALITY_CONF_THRESHOLD: float = 0.38

# ─── Supergroup Definitions ───────────────────────────────────────────────────
# Each supergroup has a canonical name, keyword patterns, and a modality tag.
# A label is assigned to the FIRST supergroup whose keywords match.
# Keywords are checked case-insensitively against the label string.

SUPERGROUPS: list[dict] = [
    # ── 0. Normal / Healthy ──────────────────────────────────────────────────
    {
        "name": "Normal / Healthy",
        "modality": "both",
        "keywords": [
            "healthy", "normal", "no pathology",
            "arcus senilis",
        ],
    },

    # ── 1. Retinal Detachment & Breaks ───────────────────────────────────────
    # (Check BEFORE generic retinal to catch "retinal detachment" first)
    {
        "name": "Retinal Detachment & Breaks",
        "modality": "posterior",
        "keywords": [
            "retinal detach", "retinal break", "retinal tear",
            "retinal dialysis", "retinoschisis", "retinal hole",
            "round hole", "horseshoe tear", "giant retinal tear",
            "separation of retinal layers",
            "traction detachment",
        ],
    },

    # ── 2. Retinal Vascular ──────────────────────────────────────────────────
    {
        "name": "Retinal Vascular",
        "modality": "posterior",
        "keywords": [
            "retinal artery", "retinal vein", "retinal vascular",
            "dr, ", "dr ", "dme",
            "brvo", "crvo", "brao", "crao", "rop",
            "cnv", "pcv", "microaneurysm",
            "retinal hemorrhage", "retinal ischemia", "retinal micro-aneurysm",
            "retinal neovascularization", "retinal telangiect",
            "intraretinal fluid", "exudative vitreoretinopathy",
            "retinal vasculitis", "retinopathy of prematurity",
            "retrolental fibroplasia",
            "hypertensive retinopathy", "background retinopathy",
            "diabetic retinopathy", "proliferative retinopathy",
            "exudative retinopathy", "solar retinopathy",
            "vein occl", "artery occl",
            "retinal edema",
            "changes in retinal vascular",
            "intraretinal microvascular",
            "retinal vascula",
            "venous engorgement",
            "macroaneurysm",
            "sickle cell", "coats disease",
        ],
    },

    # ── 3. Retinal Degenerative & Dystrophic ─────────────────────────────────
    {
        "name": "Retinal Degenerative & Dystrophic",
        "modality": "posterior",
        "keywords": [
            "macular degeneration", "mclr degn", "macular degn",
            "amd,", "amd ", "retinitis pigmentosa", "cone dystrophy",
            "chrpe", "fevr", "pseudoxanthoma", "retinoblastoma",
            "apmppe", "macular pucker", "erm", "csr",
            "subretinal fluid", "telangiectas",
            "drusen", "macular cyst", "macular edema", "macular hole",
            "pigment epithelial detachment", "outer retinal tubulation",
            "fundus flavimaculatus",
            "puckering of macula", "cystoid macular",
            "toxic maculopathy", "macular keratitis",
            "macula scars", "central serous chorioretinopathy",
            "lattice degeneration", "paving stone degeneration",
            "microcystoid degeneration", "reticular degeneration",
            "pigmentary retinal dystrophy", "retinal dystrophy",
            "sensory retina", "pigment epithelium",
            "vitreoretinal dystrophy", "vitreoretinal degeneration",
            "vitreomacular adhesion",
            "retinal disorder", "age-related reticular",
            "retinal pigment epithelium",
            "angioid streaks",
            "peripheral retinal degeneration",
            "multiple defects of retina",
            "parasitic cyst of retina",
            "secondary pigmentary degeneration",
            "degenerative myopia", "myopic degeneration",
            "epiretinal membrane",
            "gyrate atrophy",
            "geographic atrophy", "vitreomacular",
            "stargardt", "best disease",
        ],
    },

    # ── 4. Glaucoma Spectrum ─────────────────────────────────────────────────
    {
        "name": "Glaucoma Spectrum",
        "modality": "both",
        "keywords": [
            "glaucoma", "glaucomatous", "ocular hypertension",
            "preglaucoma", "open-angle", "open angle",
            "angle-closure", "angle closure",
            "steroid responder", "aqueous misdirection",
            "plateau iris", "pigmentary glaucoma",
            "chamber angle", "goniosynechiae",
            "anatomical narrow angle",
            "increased episcleral venous",
        ],
    },

    # ── 5. Corneal ───────────────────────────────────────────────────────────
    {
        "name": "Corneal",
        "modality": "anterior",
        "keywords": [
            "cornea", "keratitis", "keratocon", "keratopathy",
            "keratoconjunctiv", "keratomalacia",
            "pterygium", "pseudopterygium",
            "descemetocele", "descemet",
            "bowman", "pannus",
            "photokeratitis", "interstitial keratitis",
            "sclerosing keratitis",
            "adherent leukoma",
            "corneal ulcer",
            "herpes zoster", "pellucid",
        ],
    },

    # ── 6. Lens & Cataract ───────────────────────────────────────────────────
    {
        "name": "Lens & Cataract",
        "modality": "anterior",
        "keywords": [
            "cataract", "lens", "aphakia", "soemmering",
            "pseudxf", "capsular", "capslr", "pseudophakia",
        ],
    },

    # ── 7. Uveal & Inflammatory ──────────────────────────────────────────────
    {
        "name": "Uveal & Inflammatory",
        "modality": "both",
        "keywords": [
            "uveitis", "iridocyclitis", "panuveitis",
            "endophthalmitis", "panophthalmitis",
            "scleritis", "scleromalacia",
            "episcleritis",
            "sympathetic uveitis", "vogt-koyanagi",
            "harada",
            "iris atrophy", "iridodialysis", "iridoschisis",
            "synechiae", "pupillary membrane",
            "floppy iris", "fuchs",
            "hyphema", "hypopyon",
            "adhesions of iris",
            "degeneration of iris", "degeneration of pupillary",
            "idio cysts of iris", "parastc cyst of iris",
            "posterior cyclitis",
            "inflammation of postprocedural bleb",
            "exudative cyst", "implant cyst", "parasitic cyst of iris",
            "iris and ciliary body",
            "ciliary body",
            "sclera", "scleral ectasia",
            "chalcosis", "siderosis",
            "cytomegalovirus", "iris nevus", "iris melanoma",
            "pseudoexfoliation", "pigment dispersion",
        ],
    },

    # ── 8. Eyelid & Adnexa ───────────────────────────────────────────────────
    {
        "name": "Eyelid & Adnexa",
        "modality": "external",
        "keywords": [
            "eyelid", "blephar", "ptosis",
            "ectropion", "entropion",
            "lagophthalmos",
            "chalazion", "hordeolum",
            "dermatochalasis", "dermatitis",
            "trichiasis", "madarosis",
            "xanthelasma", "xeroderma",
            "chloasma", "vitiligo",
            "elephantiasis", "hypertrichosis",
            "retained foreign body in",
            "inflammation of eyelid",
            "cysts of lower", "cysts of upper",
            "vascular anomal",
            "eczematous",
            "discoid lupus",
            "periocular",
            "allergic dermatitis",
            "meibomian", "preseptal",
        ],
    },

    # ── 9. Lacrimal System ───────────────────────────────────────────────────
    {
        "name": "Lacrimal System",
        "modality": "external",
        "keywords": [
            "lacrimal", "dacryoadenitis", "dacryocystitis",
            "dacryolith", "dacryops",
            "nasolacrimal", "dry eye",
            "epiphora", "lacrimation",
            "keratoconjunct sicca",
        ],
    },

    # ── 10. Orbital ──────────────────────────────────────────────────────────
    {
        "name": "Orbital",
        "modality": "external",
        "keywords": [
            "orbit", "exophthalmos", "enophthalmos",
            "cellulitis of bilateral orbits",
            "cellulitis of orbit",
            "orbital myositis", "tenonitis",
            "osteomyelitis",
            "thyroid eye disease",
            "displacement", "luxation of globe",
            "staphyloma",
            "globe rupture",
            "atrophy of globe",
            "degenerated conditions of globe",
            "degenerative disorder of globe",
            "disorder of globe",
            "hypotony", "flat anterior chamber",
            "primary hypotony",
            "retained (old)", "retain intraoc", "retain (old)",
            "orbital edema",
        ],
    },

    # ── 11. Strabismus & Motility ────────────────────────────────────────────
    {
        "name": "Strabismus & Motility",
        "modality": "external",
        "keywords": [
            "strabismus", "esotropia", "exotropia",
            "esophoria", "exophoria", "hypertropia",
            "heterophoria", "heterotropia",
            "cyclophoria", "cyclotropia",
            "amblyopia", "diplopia",
            "nystagmus",
            "ophthalmoplegia",
            "nerve palsy", "conjugate gaze",
            "convergence", "accommodation",
            "binocular", "monofixation",
            "saccadic", "eye movement",
            "cranial nerve",
            "brown's sheath",
            "duane",
            "abnormal innervation",
            "myopathy of extraocular",
        ],
    },

    # ── 12. Neuro-Ophthalmic ─────────────────────────────────────────────────
    {
        "name": "Neuro-Ophthalmic",
        "modality": "posterior",
        "keywords": [
            "optic neuritis", "optic neuropathy",
            "optic atrophy", "optic nerve",
            "optic disc", "optic papillitis",
            "papilledema", "pseudopapilledema",
            "morning glory", "coloboma",
            "foster-kennedy",
            "visual pathway", "visual cortex",
            "optic chiasm",
            "retrobulbar", "ischemic optic",
            "horner",
            "kearns-sayre",
            "argyll robertson",
            "tonic pupil", "miosis", "mydriasis",
            "pupillary abnorm", "anomaly of pupillary",
            "pupillary cyst",
        ],
    },

    # ── 13. Conjunctival ─────────────────────────────────────────────────────
    {
        "name": "Conjunctival",
        "modality": "anterior",
        "keywords": [
            "conjunctiv",
            "pinguecula", "pingueculitis",
            "pemphigoid", "stevens-johnson",
            "symblepharon",
            "ophthalmia nodosa",
            "vascular abnormalities of conjunctiva",
        ],
    },

    # ── 14. Vitreous ─────────────────────────────────────────────────────────
    {
        "name": "Vitreous",
        "modality": "posterior",
        "keywords": [
            "vitreous", "vitreal", "hemophthalmos",
            "asteroid hyalosis"
        ],
    },

    # ── 15. Choroidal ────────────────────────────────────────────────────────
    {
        "name": "Choroidal",
        "modality": "posterior",
        "keywords": [
            "choroid", "chorioret",
            "choroideremia",
            "multifoc placoid pigment",
        ],
    },

    # ── 16. Visual Function & Refractive ──────────────────────────────────
    {
        "name": "Visual Function & Refractive",
        "modality": "both",
        "keywords": [
            "blindness", "low vision", "visual loss",
            "visual disturbance", "visual discomfort",
            "visual distortion", "visual field",
            "scotoma", "field defect",
            "color vision", "achromatopsia",
            "night blindness", "day blindness",
            "glare sensitivity", "contrast sensitivity",
            "myopia", "hypermetropia", "presbyopia",
            "astigmatism", "anisometropia", "aniseikonia",
            "disorder of refraction",
            "leucocoria", "anisocoria",
            "psychophysical visual",
            "subjective visual",
            "deuteranomaly", "protanomaly", "tritanomaly",
            "sudden visual loss", "transient visual loss",
            "ocular pain",
        ],
    },

    # ── 17. Surgical & Traumatic Complications ───────────────────────────────
    {
        "name": "Surgical & Traumatic Complications",
        "modality": "both",
        "keywords": [
            "postproc", "intraop",
            "acc pnctr",
            "following cataract surgery",
            "fol cataract surgery",
            "postprocedural",
            "localized traumatic",
            "disorder of eye and adnexa",
            "disorders of eye and adnexa",
            "oth intraoperative",
            "oth postproc",
            "cyst of ora serrata",
            "cyst of pars plana",
            "cysts",
            "kayser-fleischer",
        ],
    },
]

NORMAL_GROUP = "Normal / Healthy"
UNCATEGORIZED_GROUP = "Uncategorized"

# ─── Modality Prefix → Segment Mapping ────────────────────────────────────────
# Maps the modality prefixes used in eye_diseases_curated.txt to segment types.
# This allows direct modality extraction from prefixed labels without keyword matching.
MODALITY_PREFIX_TO_SEGMENT: dict[str, str] = {
    "OCT":                  "posterior",
    "CFP":                  "posterior",
    "FFA":                  "posterior",
    "ICGA":                 "posterior",
    "FAF":                  "posterior",
    "OUS":                  "posterior",
    "RetCam":               "posterior",
    "slit lamp":            "anterior",
    "specular":             "anterior",
    "corneal photography":  "anterior",
    "external":             "external",
}

# All valid modality prefixes (ordered longest-first for greedy matching)
_SORTED_PREFIXES = sorted(MODALITY_PREFIX_TO_SEGMENT.keys(), key=len, reverse=True)


def parse_modality_prefix(label: str) -> tuple[str | None, str]:
    """
    Parse a modality-prefixed label into (modality_prefix, condition).

    Examples:
        "OCT, drusen"           → ("OCT", "drusen")
        "slit lamp, corneal ulcer" → ("slit lamp", "corneal ulcer")
        "diabetic retinopathy"  → (None, "diabetic retinopathy")
    """
    for prefix in _SORTED_PREFIXES:
        # Check for "prefix, condition" or "prefix,condition" format
        if label.startswith(prefix + ","):
            condition = label[len(prefix) + 1:].strip()
            return prefix, condition
    return None, label


def get_modality_from_label(label: str) -> str:
    """
    Get the segment modality directly from a modality-prefixed label.

    For prefixed labels ("OCT, drusen"), returns the segment type ("posterior").
    For unprefixed labels, falls back to supergroup keyword matching.

    Returns: "anterior", "posterior", "external", or "both"
    """
    prefix, _ = parse_modality_prefix(label)
    if prefix:
        return MODALITY_PREFIX_TO_SEGMENT.get(prefix, "both")

    # Fallback for unprefixed labels: use supergroup keyword matching
    sg_name = assign_label_to_supergroup(label)
    for group in SUPERGROUPS:
        if group["name"] == sg_name:
            return group.get("modality", "both")
    return "both"


# ─── Modality → Supergroup Mapping ────────────────────────────────────────────

def get_modality_allowed_groups(modality: str) -> set[str]:
    """
    Return the set of supergroup names allowed for a given imaging modality.
    "anterior" and "external" both map to anterior-compatible groups.
    "posterior" maps to posterior-compatible groups.
    Groups tagged "both" are always allowed.
    """
    if modality == "external":
        modality = "anterior"  # external photos show anterior structures

    allowed = set()
    for group in SUPERGROUPS:
        gmod = group.get("modality", "both")
        if gmod == "both" or gmod == modality:
            allowed.add(group["name"])
    # Uncategorized is always allowed
    allowed.add(UNCATEGORIZED_GROUP)
    return allowed


# ─── Label Assignment ─────────────────────────────────────────────────────────

def assign_label_to_supergroup(label: str) -> str:
    """
    Assign a single label to its supergroup by keyword matching.
    Handles both modality-prefixed ("OCT, drusen") and plain labels.
    Returns the supergroup name, or UNCATEGORIZED_GROUP if no match.
    """
    # Strip modality prefix if present, so keyword matching works on the condition
    _, condition = parse_modality_prefix(label)
    condition_lower = condition.lower()

    # Also check the full label (prefix included) for broader matching
    label_lower = label.lower()

    for group in SUPERGROUPS:
        for keyword in group["keywords"]:
            kw = keyword.lower()
            if kw in condition_lower or kw in label_lower:
                return group["name"]
    return UNCATEGORIZED_GROUP


def build_taxonomy(labels: list[str]) -> dict:
    """
    Assign all labels to supergroups and return a complete taxonomy dict.

    Returns:
        {
            "label_to_supergroup": {label: group_name, ...},
            "supergroup_to_indices": {group_name: [idx0, idx1, ...], ...},
            "supergroup_names": [group_name, ...],   # ordered, unique
        }
    """
    label_to_sg: dict[str, str] = {}
    sg_to_indices: dict[str, list[int]] = {}

    for idx, label in enumerate(labels):
        sg = assign_label_to_supergroup(label)
        label_to_sg[label] = sg
        sg_to_indices.setdefault(sg, []).append(idx)

    # Ordered list of supergroup names (preserve definition order, add Uncategorized if used)
    sg_names = [g["name"] for g in SUPERGROUPS if g["name"] in sg_to_indices]
    if UNCATEGORIZED_GROUP in sg_to_indices:
        sg_names.append(UNCATEGORIZED_GROUP)

    return {
        "label_to_supergroup": label_to_sg,
        "supergroup_to_indices": sg_to_indices,
        "supergroup_names": sg_names,
    }


# ─── CLI Preview ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """Quick preview: load the label file and print taxonomy assignments."""
    import os
    import sys

    # Default to curated labels; accept a path argument for legacy labels
    default_label_path = os.path.join(".", "data", "processed", "eye_diseases_curated.txt")
    label_path = sys.argv[1] if len(sys.argv) > 1 else default_label_path

    if not os.path.exists(label_path):
        print(f"Label file not found at {label_path}")
        exit(1)

    with open(label_path, "r") as f:
        labels = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    taxonomy = build_taxonomy(labels)

    print(f"\n{'='*60}")
    print(f"  OPHTHALMIC TAXONOMY — {len(labels)} labels → "
          f"{len(taxonomy['supergroup_names'])} supergroups")
    print(f"{'='*60}\n")

    for sg_name in taxonomy["supergroup_names"]:
        indices = taxonomy["supergroup_to_indices"][sg_name]
        print(f"  [{len(indices):3d}] {sg_name}")
        for idx in indices[:3]:
            prefix, cond = parse_modality_prefix(labels[idx])
            display = f"[{prefix}] {cond}" if prefix else labels[idx]
            print(f"         • {display}")
        if len(indices) > 3:
            print(f"         ... and {len(indices)-3} more")
        print()

    # Check coverage
    total_assigned = sum(len(v) for v in taxonomy["supergroup_to_indices"].values())
    print(f"  Coverage: {total_assigned}/{len(labels)} labels assigned")
    if UNCATEGORIZED_GROUP in taxonomy["supergroup_to_indices"]:
        uncategorized = taxonomy["supergroup_to_indices"][UNCATEGORIZED_GROUP]
        print(f"\n  ⚠ {len(uncategorized)} UNCATEGORIZED labels:")
        for idx in uncategorized:
            print(f"    • {labels[idx]}")
