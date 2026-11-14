from typing import Dict, List

SUPPORTED_MODULATIONS: Dict[str, int] = {
    "16QAM": 16,
    "64QAM": 64,
    "256QAM": 256,
}

MODULATION_ID_ORDER: List[str] = list(SUPPORTED_MODULATIONS.keys())
MOD_NAME_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(MODULATION_ID_ORDER)}
MOD_ID_TO_NAME: Dict[int, str] = {i: name for name, i in MOD_NAME_TO_ID.items()}

TRAIN_MODULATION_CHOICES: List[str] = ["16QAM", "64QAM"]
TEST_MODULATION_GROUPS: Dict[str, List[str]] = {
    "mod16": ["16QAM"],
    "mod64": ["64QAM"],
    "mixed": ["16QAM", "64QAM"],
    "heldout256": ["256QAM"],
}
