"""Multi-plan compliance rules: which standard plans govern each component type.

For each component category we store:
  - applicable_plans: list of (agency, plan_id_hint, search_query) tuples used
    to retrieve from the v3 1898-page index.
  - threshold: numeric requirement + comparison operator + unit.
  - drawing_label: how the value appears on the design drawing.

The agent (Planner) sees ONLY the drawing (pure visual inference). It must
identify component types from the visual content and decide which plans apply.
This registry is used at scoring time (ground truth) and to compose drawings.
"""
from __future__ import annotations

# Each component definition: required search-style content for ColPali retrieval,
# and the threshold the design value must meet.
COMPONENT_RULES = {
    "concrete_cover": {
        "search_queries": [
            "concrete cover requirements reinforced section",
            "minimum cover for reinforcing steel ACI",
        ],
        "applicable_plans_hint": ["ACI 318 / AASHTO LRFD"],
        "threshold": 2.0, "operator": ">=", "unit": "inches",
        "rule_name": "Min concrete cover (ACI 318/AASHTO LRFD)",
    },
    "rebar_grade": {
        "search_queries": [
            "reinforcing steel grade specification standard",
            "ASTM A615 reinforcement requirements",
        ],
        "applicable_plans_hint": ["ASTM A615", "AASHTO LRFD"],
        "threshold": 60, "operator": ">=", "unit": "ksi",
        "rule_name": "Min rebar yield strength (Grade 60)",
    },
    "concrete_class": {
        "search_queries": [
            "concrete class strength requirements bridge",
            "concrete mix design DOT specifications",
        ],
        "applicable_plans_hint": ["WYDOT 706", "Caltrans Sec 90"],
        "threshold": 4000, "operator": ">=", "unit": "psi",
        "rule_name": "Min compressive strength (Class A / 4000 psi)",
    },
    "stirrup_spacing": {
        "search_queries": [
            "stirrup spacing requirements beam reinforcement",
            "shear reinforcement maximum spacing ACI",
        ],
        "applicable_plans_hint": ["ACI 318"],
        "threshold": "d/2", "operator": "<=", "unit": "inches",
        "rule_name": "Max stirrup spacing = d/2 (ACI 318)",
    },
    "footing_depth": {
        "search_queries": [
            "post foundation minimum footing depth guardrail",
            "footing embedment requirements steel post",
        ],
        "applicable_plans_hint": ["WYDOT 606.05", "FDOT 515"],
        "threshold": 30, "operator": ">=", "unit": "inches",
        "rule_name": "Min footing depth (WYDOT 606.05 / FDOT 515)",
    },
    "anchor_bolt_embed": {
        "search_queries": [
            "anchor bolt embedment minimum sign post",
            "anchor bolt installation depth AASHTO",
        ],
        "applicable_plans_hint": ["AASHTO", "AZDOT C-25"],
        "threshold": 12, "operator": ">=", "unit": "inches",
        "rule_name": "Min anchor bolt embedment (AASHTO)",
    },
    "joint_seal_width": {
        "search_queries": [
            "joint seal width expansion joint pavement",
            "expansion joint seal specifications",
        ],
        "applicable_plans_hint": ["Caltrans P20", "WYDOT 706"],
        "threshold": 0.5, "operator": ">=", "unit": "inches",
        "rule_name": "Min joint seal width (Caltrans P20)",
    },
    "pavement_thickness": {
        "search_queries": [
            "pavement structural section thickness asphalt",
            "concrete pavement minimum thickness highway",
        ],
        "applicable_plans_hint": ["AASHTO Pavement Design", "FDOT 350"],
        "threshold": 6.0, "operator": ">=", "unit": "inches",
        "rule_name": "Min pavement structural thickness",
    },
    "bearing_pad_thickness": {
        "search_queries": [
            "elastomeric bearing pad thickness bridge",
            "neoprene bearing pad specifications AASHTO",
        ],
        "applicable_plans_hint": ["AASHTO LRFD 14.7", "Caltrans Sec 51"],
        "threshold": 1.0, "operator": ">=", "unit": "inches",
        "rule_name": "Min bearing pad thickness (AASHTO LRFD 14.7)",
    },
    "drainage_grate_open": {
        "search_queries": [
            "drainage grate opening size ADA bicycle",
            "inlet grate opening maximum dimension",
        ],
        "applicable_plans_hint": ["PROWAG / ADA", "FDOT 232"],
        "threshold": 4.0, "operator": "<=", "unit": "inches",
        "rule_name": "Max grate opening (PROWAG/ADA)",
    },
}


def get_rule(component_key):
    return COMPONENT_RULES[component_key]


def all_components():
    return list(COMPONENT_RULES.keys())
