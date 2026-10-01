"""
Bill layout profiles for the cash-reader project.

Coordinates are normalized from 0.0 to 1.0.

    x = left -> right
    y = top -> bottom

These profiles describe the FRONT of U.S. bills.

The back of the bill is handled separately by pipeline.py.

IMPORTANT:
These are prototype regions. As we test real camera images, the
coordinates can be refined for each denomination and bill design.
"""

from __future__ import annotations


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def make_region(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
):
    """
    Create a normalized image region.

    Coordinates are percentages represented as decimals.

    Example:

        make_region(0.05, 0.10, 0.40, 0.25)

    means:

        left   = 5%
        top    = 10%
        right  = 40%
        bottom = 25%
    """

    return [
        float(x1),
        float(y1),
        float(x2),
        float(y2),
    ]


def make_profile(
    name: str,
    denomination: str,
    serial_left,
    serial_right,
    series,
    fed_letter,
    fed_number,
    denomination_regions=None,
):
    """
    Build a bill profile.

    All regions are normalized coordinates.
    """

    return {
        "name": name,
        "denomination": str(denomination),

        # Front-side identifying fields.
        "serial_left": serial_left,
        "serial_right": serial_right,
        "series": series,
        "fed_letter": fed_letter,
        "fed_number": fed_number,

        # Additional denomination recognition regions.
        "denomination_regions": (
            denomination_regions
            if denomination_regions
            else []
        ),
    }


# =====================================================================
# $1 BILL
# =====================================================================

ONE_DOLLAR = make_profile(
    name="U.S. One Dollar",
    denomination="1",

    # Serial number on left side.
    serial_left=make_region(
        0.12,
        0.62,
        0.42,
        0.75,
    ),

    # Serial number on right side.
    serial_right=make_region(
        0.60,
        0.22,
        0.88,
        0.37,
    ),

    # Series information.
    series=make_region(
        0.64,
        0.75,
        0.72,
        0.89,
    ),

    # Federal Reserve letter.
    fed_letter=make_region(
        0.24,
        0.40,
        0.36,
        0.60,
    ),

    # Federal Reserve district number.
    fed_number=make_region(
        0.10,
        0.39,
        0.23,
        0.53,
    ),

    denomination_regions=[
        make_region(
            0.00,
            0.00,
            0.25,
            0.30,
        ),
        make_region(
            0.75,
            0.00,
            1.00,
            0.30,
        ),
        make_region(
            0.00,
            0.70,
            0.25,
            1.00,
        ),
        make_region(
            0.75,
            0.70,
            1.00,
            1.00,
        ),
    ],
)


# =====================================================================
# $5 BILL
# =====================================================================

FIVE_DOLLAR = make_profile(
    name="U.S. Five Dollar",
    denomination="5",

    serial_left=make_region(
        0.06,
        0.66,
        0.48,
        0.84,
    ),

    serial_right=make_region(
        0.52,
        0.16,
        0.94,
        0.34,
    ),

    series=make_region(
        0.22,
        0.52,
        0.78,
        0.76,
    ),

    fed_letter=make_region(
        0.18,
        0.15,
        0.40,
        0.38,
    ),

    fed_number=make_region(
        0.60,
        0.15,
        0.82,
        0.38,
    ),

    denomination_regions=[
        make_region(
            0.00,
            0.00,
            0.25,
            0.30,
        ),
        make_region(
            0.75,
            0.00,
            1.00,
            0.30,
        ),
        make_region(
            0.00,
            0.70,
            0.25,
            1.00,
        ),
        make_region(
            0.75,
            0.70,
            1.00,
            1.00,
        ),
    ],
)


# =====================================================================
# $10 BILL
# =====================================================================

TEN_DOLLAR = make_profile(
    name="U.S. Ten Dollar",
    denomination="10",

    serial_left=make_region(
        0.06,
        0.66,
        0.48,
        0.84,
    ),

    serial_right=make_region(
        0.52,
        0.16,
        0.94,
        0.34,
    ),

    series=make_region(
        0.22,
        0.52,
        0.78,
        0.76,
    ),

    fed_letter=make_region(
        0.18,
        0.15,
        0.40,
        0.38,
    ),

    fed_number=make_region(
        0.60,
        0.15,
        0.82,
        0.38,
    ),

    denomination_regions=[
        make_region(
            0.00,
            0.00,
            0.25,
            0.30,
        ),
        make_region(
            0.75,
            0.00,
            1.00,
            0.30,
        ),
        make_region(
            0.00,
            0.70,
            0.25,
            1.00,
        ),
        make_region(
            0.75,
            0.70,
            1.00,
            1.00,
        ),
    ],
)


# =====================================================================
# $20 BILL
# =====================================================================

TWENTY_DOLLAR = make_profile(
    name="U.S. Twenty Dollar",
    denomination="20",

    serial_left=make_region(
        0.06,
        0.66,
        0.48,
        0.84,
    ),

    serial_right=make_region(
        0.52,
        0.16,
        0.94,
        0.34,
    ),

    series=make_region(
        0.22,
        0.52,
        0.78,
        0.76,
    ),

    fed_letter=make_region(
        0.18,
        0.15,
        0.40,
        0.38,
    ),

    fed_number=make_region(
        0.60,
        0.15,
        0.82,
        0.38,
    ),

    denomination_regions=[
        make_region(
            0.00,
            0.00,
            0.25,
            0.30,
        ),
        make_region(
            0.75,
            0.00,
            1.00,
            0.30,
        ),
        make_region(
            0.00,
            0.70,
            0.25,
            1.00,
        ),
        make_region(
            0.75,
            0.70,
            1.00,
            1.00,
        ),
    ],
)


# =====================================================================
# $50 BILL
# =====================================================================

FIFTY_DOLLAR = make_profile(
    name="U.S. Fifty Dollar",
    denomination="50",

    serial_left=make_region(
        0.06,
        0.66,
        0.48,
        0.84,
    ),

    serial_right=make_region(
        0.52,
        0.16,
        0.94,
        0.34,
    ),

    series=make_region(
        0.22,
        0.52,
        0.78,
        0.76,
    ),

    fed_letter=make_region(
        0.18,
        0.15,
        0.40,
        0.38,
    ),

    fed_number=make_region(
        0.60,
        0.15,
        0.82,
        0.38,
    ),

    denomination_regions=[
        make_region(
            0.00,
            0.00,
            0.25,
            0.30,
        ),
        make_region(
            0.75,
            0.00,
            1.00,
            0.30,
        ),
        make_region(
            0.00,
            0.70,
            0.25,
            1.00,
        ),
        make_region(
            0.75,
            0.70,
            1.00,
            1.00,
        ),
    ],
)


# =====================================================================
# $100 BILL
# =====================================================================

ONE_HUNDRED_DOLLAR = make_profile(
    name="U.S. One Hundred Dollar",
    denomination="100",

    serial_left=make_region(
        0.06,
        0.66,
        0.48,
        0.84,
    ),

    serial_right=make_region(
        0.52,
        0.16,
        0.94,
        0.34,
    ),

    series=make_region(
        0.22,
        0.52,
        0.78,
        0.76,
    ),

    fed_letter=make_region(
        0.18,
        0.15,
        0.40,
        0.38,
    ),

    fed_number=make_region(
        0.60,
        0.15,
        0.82,
        0.38,
    ),

    denomination_regions=[
        make_region(
            0.00,
            0.00,
            0.25,
            0.30,
        ),
        make_region(
            0.75,
            0.00,
            1.00,
            0.30,
        ),
        make_region(
            0.00,
            0.70,
            0.25,
            1.00,
        ),
        make_region(
            0.75,
            0.70,
            1.00,
            1.00,
        ),
    ],
)


# =====================================================================
# PROFILE COLLECTION
# =====================================================================

PROFILES = {
    "1": ONE_DOLLAR,
    "5": FIVE_DOLLAR,
    "10": TEN_DOLLAR,
    "20": TWENTY_DOLLAR,
    "50": FIFTY_DOLLAR,
    "100": ONE_HUNDRED_DOLLAR,
}


# ---------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------

def get_profile(
    denomination,
):
    """
    Return the profile for a denomination.

    Examples:

        get_profile("20")
        get_profile(20)
        get_profile("$20")
    """

    if denomination is None:
        return None

    text = str(
        denomination
    ).strip().upper()

    text = text.replace(
        "$",
        "",
    )

    # Handle denomination words.
    word_to_number = {
        "ONE": "1",
        "FIVE": "5",
        "TEN": "10",
        "TWENTY": "20",
        "FIFTY": "50",
        "ONE HUNDRED": "100",
        "ONEHUNDRED": "100",
        "HUNDRED": "100",
    }

    if text in word_to_number:
        text = word_to_number[text]

    return PROFILES.get(
        text
    )


def all_profiles():
    """
    Return all supported bill profiles.
    """

    return PROFILES.copy()


def supported_denominations():
    """
    Return supported denominations as strings.
    """

    return list(
        PROFILES.keys()
    )


# ---------------------------------------------------------------------
# Optional aliases for older code
# ---------------------------------------------------------------------

BILL_PROFILES = PROFILES


__all__ = [
    "PROFILES",
    "BILL_PROFILES",
    "ONE_DOLLAR",
    "FIVE_DOLLAR",
    "TEN_DOLLAR",
    "TWENTY_DOLLAR",
    "FIFTY_DOLLAR",
    "ONE_HUNDRED_DOLLAR",
    "get_profile",
    "all_profiles",
    "supported_denominations",
]
