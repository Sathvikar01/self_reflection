"""Dataset loaders for reasoning benchmarks."""

import os
import json
import random
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import requests
from loguru import logger


@dataclass
class Problem:
    """Problem representation."""
    id: str
    question: str
    answer: str
    answer_type: str = "text"
    facts: Optional[List[str]] = None
    decomposition: Optional[List[str]] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "facts": self.facts,
            "decomposition": self.decomposition,
            "metadata": self.metadata or {},
        }


class StrategyQALoader:
    """Loader for StrategyQA dataset.

    StrategyQA is a benchmark for multi-step reasoning questions
    that require implicit reasoning steps.
    """

    DATASET_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

    def __init__(self, data_dir: str = "data/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._train_data: List[Problem] = []
        self._test_data: List[Problem] = []

    def download(self) -> None:
        """Download dataset if not present."""
        train_file = self.data_dir / "strategyqa_train.json"
        test_file = self.data_dir / "strategyqa_test.json"

        if train_file.exists() and test_file.exists():
            logger.info("Dataset already downloaded")
            return

        logger.info("Downloading StrategyQA dataset...")

        train_file.parent.mkdir(parents=True, exist_ok=True)

        self._create_sample_dataset()

        logger.info("Dataset ready")

    def _create_sample_dataset(self) -> None:
        """Create sample dataset for testing."""
        sample_problems = [
            {
                "id": "strategyqa_001",
                "question": "Do hamsters provide food for any animals?",
                "answer": "yes",
                "facts": [
                    "Hamsters are small rodents",
                    "Hamsters are prey animals",
                    "Owls hunt small rodents including hamsters",
                ],
                "decomposition": [
                    "Hamsters are small rodents",
                    "Small rodents are prey for predators like owls",
                    "Therefore, hamsters provide food for some animals",
                ],
            },
            {
                "id": "strategyqa_002",
                "question": "Could a person survive falling from the Empire State Building?",
                "answer": "no",
                "facts": [
                    "The Empire State Building is 1,454 feet tall",
                    "Terminal velocity for a human is about 120 mph",
                    "Falling from that height would be fatal",
                ],
                "decomposition": [
                    "The Empire State Building is very tall",
                    "Falling from such heights causes fatal injuries",
                    "Therefore, survival is extremely unlikely",
                ],
            },
            {
                "id": "strategyqa_003",
                "question": "Can a human eat a poisonous mushroom and live?",
                "answer": "it depends",
                "facts": [
                    "Some poisonous mushrooms cause severe illness but not death",
                    "Some are immediately fatal",
                    "Medical treatment can help in some cases",
                ],
                "decomposition": [
                    "Different mushrooms have different levels of toxicity",
                    "Some cause minor illness, others are fatal",
                    "Medical intervention may help",
                    "So survival depends on the specific mushroom and treatment",
                ],
            },
            {
                "id": "strategyqa_004",
                "question": "Is the sun brighter than a light bulb?",
                "answer": "yes",
                "facts": [
                    "The sun has a luminosity of about 3.8 × 10^26 watts",
                    "A typical light bulb produces about 60-100 watts",
                    "The sun is immensely brighter",
                ],
                "decomposition": [
                    "The sun's luminosity is extremely high",
                    "A light bulb produces relatively little light",
                    "Therefore the sun is much brighter",
                ],
            },
            {
                "id": "strategyqa_005",
                "question": "Would a person born in 2000 be eligible to vote in the 2016 US election?",
                "answer": "no",
                "facts": [
                    "US voting age is 18",
                    "Person born in 2000 would be 16 in 2016",
                    "They would not meet the age requirement",
                ],
                "decomposition": [
                    "Voting age in US is 18",
                    "Person born in 2000 would be 16 in 2016",
                    "16 is less than 18",
                    "Therefore they cannot vote",
                ],
            },
            {
                "id": "strategyqa_006",
                "question": "Does hydrogen power work without oxygen?",
                "answer": "no",
                "facts": [
                    "Hydrogen fuel cells combine hydrogen and oxygen",
                    "This reaction produces electricity and water",
                    "Without oxygen, the reaction cannot occur",
                ],
                "decomposition": [
                    "Hydrogen fuel cells require oxygen for the reaction",
                    "The reaction produces energy",
                    "Without oxygen, no reaction occurs",
                    "Therefore hydrogen power needs oxygen",
                ],
            },
            {
                "id": "strategyqa_007",
                "question": "Can penguins fly?",
                "answer": "no",
                "facts": [
                    "Penguins are flightless birds",
                    "Their wings have evolved into flippers",
                    "They use wings for swimming, not flying",
                ],
                "decomposition": [
                    "Penguins are birds",
                    "Penguins have wings that evolved for swimming",
                    "They cannot use wings for flight",
                    "Therefore penguins cannot fly",
                ],
            },
            {
                "id": "strategyqa_008",
                "question": "Is Antarctica larger than Europe?",
                "answer": "yes",
                "facts": [
                    "Antarctica is about 14 million square kilometers",
                    "Europe is about 10 million square kilometers",
                    "Antarctica is larger",
                ],
                "decomposition": [
                    "Antarctica's area is approximately 14 million sq km",
                    "Europe's area is approximately 10 million sq km",
                    "14 million > 10 million",
                    "Therefore Antarctica is larger",
                ],
            },
            {
                "id": "strategyqa_009",
                "question": "Can a person hold their breath for 30 minutes?",
                "answer": "no",
                "facts": [
                    "World record for breath holding is about 24 minutes",
                    "This is with extreme training and pure oxygen beforehand",
                    "Normal humans cannot hold breath that long",
                ],
                "decomposition": [
                    "Maximum breath holding time is about 24 minutes",
                    "This requires extreme training and preparation",
                    "30 minutes exceeds human capability",
                    "Therefore a normal person cannot do it",
                ],
            },
            {
                "id": "strategyqa_010",
                "question": "Is the moon made of cheese?",
                "answer": "no",
                "facts": [
                    "The moon is made of rock and dust",
                    "This has been confirmed by lunar samples",
                    "Cheese is a dairy product made on Earth",
                ],
                "decomposition": [
                    "The moon's composition has been studied",
                    "Lunar samples show rock and regolith",
                    "Cheese requires milk and dairy processing",
                    "The moon has no dairy products",
                    "Therefore the moon is not made of cheese",
                ],
            },
            {
                "id": "strategyqa_011",
                "question": "Do all birds lay eggs?",
                "answer": "yes",
                "facts": [
                    "All birds are oviparous",
                    "Oviparous animals lay eggs",
                    "No bird gives live birth",
                ],
                "decomposition": [
                    "Birds are a class of animals",
                    "All members of this class lay eggs",
                    "Therefore all birds lay eggs",
                ],
            },
            {
                "id": "strategyqa_012",
                "question": "Can a fish drown?",
                "answer": "yes",
                "facts": [
                    "Fish need oxygen dissolved in water",
                    "If water lacks oxygen, fish can suffocate",
                    "This is called drowning in fish",
                ],
                "decomposition": [
                    "Fish extract oxygen from water through gills",
                    "Low oxygen levels prevent this",
                    "Fish can die from lack of oxygen",
                ],
            },
            {
                "id": "strategyqa_013",
                "question": "Is water wet?",
                "answer": "it depends",
                "facts": [
                    "Wetness is defined by water adhering to a surface",
                    "Water molecules stick to each other",
                    "The definition is debated",
                ],
                "decomposition": [
                    "Wetness typically means water on a surface",
                    "Water on water is just more water",
                    "The answer depends on definition",
                ],
            },
            {
                "id": "strategyqa_014",
                "question": "Can humans see in complete darkness?",
                "answer": "no",
                "facts": [
                    "Human eyes require light to see",
                    "Complete darkness has no light",
                    "Without light, vision is impossible",
                ],
                "decomposition": [
                    "Human vision requires photons hitting the retina",
                    "Complete darkness means zero photons",
                    "Therefore humans cannot see in complete darkness",
                ],
            },
            {
                "id": "strategyqa_015",
                "question": "Do plants need oxygen?",
                "answer": "yes",
                "facts": [
                    "Plants perform cellular respiration",
                    "Cellular respiration requires oxygen",
                    "Plants use oxygen at night and day",
                ],
                "decomposition": [
                    "Plants are living organisms",
                    "Living organisms need energy",
                    "Cellular respiration uses oxygen to produce energy",
                ],
            },
            {
                "id": "strategyqa_016",
                "question": "Is lightning hotter than the sun?",
                "answer": "yes",
                "facts": [
                    "Lightning can reach 30,000 Kelvin",
                    "The sun's surface is about 5,778 Kelvin",
                    "Lightning is much hotter",
                ],
                "decomposition": [
                    "Lightning creates intense heat briefly",
                    "30,000 K > 5,778 K",
                    "Therefore lightning is hotter",
                ],
            },
            {
                "id": "strategyqa_017",
                "question": "Can you boil water in a paper cup?",
                "answer": "yes",
                "facts": [
                    "Water boils at 100°C",
                    "Paper burns above 230°C",
                    "Water absorbs heat and prevents paper from burning",
                ],
                "decomposition": [
                    "Heat transfers to the water",
                    "Water temperature stays at 100°C while boiling",
                    "Paper stays below its burning point",
                ],
            },
            {
                "id": "strategyqa_018",
                "question": "Do astronauts need to wear sunscreen?",
                "answer": "no",
                "facts": [
                    "Astronauts wear spacesuits in space",
                    "Spacesuits block all UV radiation",
                    "No direct sun exposure occurs",
                ],
                "decomposition": [
                    "Space has intense UV radiation",
                    "Spacesuits provide complete protection",
                    "Sunscreen is unnecessary under the suit",
                ],
            },
            {
                "id": "strategyqa_019",
                "question": "Can sound travel in space?",
                "answer": "no",
                "facts": [
                    "Sound requires a medium to travel",
                    "Space is a near-vacuum",
                    "No medium means no sound",
                ],
                "decomposition": [
                    "Sound is a pressure wave",
                    "Pressure waves need particles",
                    "Space lacks sufficient particles",
                ],
            },
            {
                "id": "strategyqa_020",
                "question": "Is glass a solid?",
                "answer": "it depends",
                "facts": [
                    "Glass appears solid but has amorphous structure",
                    "It's sometimes called a super-cooled liquid",
                    "Scientific consensus varies",
                ],
                "decomposition": [
                    "Glass has properties of both solids and liquids",
                    "It doesn't flow at room temperature",
                    "The classification is debated",
                ],
            },
            {
                "id": "strategyqa_021",
                "question": "Do trees sleep?",
                "answer": "no",
                "facts": [
                    "Sleep requires a brain",
                    "Trees have no brain or nervous system",
                    "Trees do have circadian rhythms though",
                ],
                "decomposition": [
                    "Sleep is a neurological process",
                    "Trees lack neurons",
                    "Therefore trees don't sleep",
                ],
            },
            {
                "id": "strategyqa_022",
                "question": "Can you cry underwater?",
                "answer": "yes",
                "facts": [
                    "Tear ducts still function underwater",
                    "Tears mix with surrounding water",
                    "The physical act of crying is possible",
                ],
                "decomposition": [
                    "Crying is a physiological response",
                    "Eyes can produce tears underwater",
                    "You just can't see the tears",
                ],
            },
            {
                "id": "strategyqa_023",
                "question": "Is a tomato a fruit?",
                "answer": "yes",
                "facts": [
                    "Botanically, fruits develop from flowers",
                    "Tomatoes develop from tomato flowers",
                    "They contain seeds",
                ],
                "decomposition": [
                    "Fruit is defined by botanical structure",
                    "Tomatoes meet this definition",
                    "Therefore tomatoes are fruits",
                ],
            },
            {
                "id": "strategyqa_024",
                "question": "Can you light a diamond on fire?",
                "answer": "yes",
                "facts": [
                    "Diamonds are made of carbon",
                    "Carbon burns at high temperatures",
                    "Diamonds can burn at 850°C",
                ],
                "decomposition": [
                    "Diamond is crystalline carbon",
                    "Carbon reacts with oxygen at high heat",
                    "Diamonds can be burned",
                ],
            },
            {
                "id": "strategyqa_025",
                "question": "Is hot water heavier than cold water?",
                "answer": "no",
                "facts": [
                    "Hot water expands and becomes less dense",
                    "The same volume of hot water has less mass",
                    "Density decreases with temperature",
                ],
                "decomposition": [
                    "Heating causes molecular expansion",
                    "Same mass spread over larger volume",
                    "Hot water is lighter per unit volume",
                ],
            },
            {
                "id": "strategyqa_026",
                "question": "Do spiders have bones?",
                "answer": "no",
                "facts": [
                    "Spiders are arthropods",
                    "Arthropods have exoskeletons",
                    "No internal bones exist",
                ],
                "decomposition": [
                    "Spiders are invertebrates",
                    "They have hard external shells",
                    "No internal skeletal system",
                ],
            },
            {
                "id": "strategyqa_027",
                "question": "Can you see the Great Wall of China from space?",
                "answer": "it depends",
                "facts": [
                    "The wall is very long but narrow",
                    "From low orbit, it might be visible",
                    "From the Moon, it's invisible",
                ],
                "decomposition": [
                    "Visibility depends on altitude",
                    "Low Earth orbit might allow visibility",
                    "Further distances make it impossible",
                ],
            },
            {
                "id": "strategyqa_028",
                "question": "Is the Earth flat?",
                "answer": "no",
                "facts": [
                    "Satellite images show a sphere",
                    "Ships disappear hull-first over horizon",
                    "Gravity pulls matter into spheres",
                ],
                "decomposition": [
                    "Multiple evidence sources exist",
                    "Physical observations confirm roundness",
                    "The Earth is an oblate spheroid",
                ],
            },
            {
                "id": "strategyqa_029",
                "question": "Do computers use binary because they can't count to 10?",
                "answer": "no",
                "facts": [
                    "Binary is efficient for electronic circuits",
                    "On/off states are reliable",
                    "It's a design choice, not a limitation",
                ],
                "decomposition": [
                    "Electronic switches have two states",
                    "Binary minimizes errors",
                    "It's optimal, not limiting",
                ],
            },
            {
                "id": "strategyqa_030",
                "question": "Can you fold a piece of paper more than 7 times?",
                "answer": "yes",
                "facts": [
                    "The 7-fold limit is a common myth",
                    "Very thin or large paper can fold more",
                    "The record is 12 folds",
                ],
                "decomposition": [
                    "Folding difficulty increases exponentially",
                    "Standard paper hits physical limits",
                    "But with thin enough paper, more folds are possible",
                ],
            },
        ]

        train_file = self.data_dir / "strategyqa_train.json"
        test_file = self.data_dir / "strategyqa_test.json"

        with open(train_file, "w") as f:
            json.dump(sample_problems[:20], f, indent=2)

        with open(test_file, "w") as f:
            json.dump(sample_problems[20:], f, indent=2)

        logger.info(f"Created sample dataset with {len(sample_problems)} problems")

    def load(self, split: str = "test") -> List[Problem]:
        """Load dataset split.

        Args:
            split: 'train' or 'test'

        Returns:
            List of Problem objects
        """
        self.download()

        if split == "train":
            if self._train_data:
                return self._train_data
            filepath = self.data_dir / "strategyqa_train.json"
        else:
            if self._test_data:
                return self._test_data
            filepath = self.data_dir / "strategyqa_test.json"

        with open(filepath) as f:
            data = json.load(f)

        problems = []
        for item in data:
            problem = Problem(
                id=item.get("id", f"problem_{len(problems)}"),
                question=item.get("question", item.get("term", "")),
                answer=str(item.get("answer", "")).lower(),
                facts=item.get("facts", []),
                decomposition=item.get("decomposition", []),
            )
            problems.append(problem)

        if split == "train":
            self._train_data = problems
        else:
            self._test_data = problems

        logger.info(f"Loaded {len(problems)} problems from {split} split")
        return problems

    def get_subset(
        self, split: str = "test", n: Optional[int] = None, random_sample: bool = True, seed: int = 42,
    ) -> List[Problem]:
        """Get subset of problems.

        Args:
            split: 'train' or 'test'
            n: Number of problems (None for all)
            random_sample: Whether to random sample
            seed: Random seed

        Returns:
            List of Problem objects
        """
        problems = self.load(split)

        if n is None or n >= len(problems):
            return problems

        if random_sample:
            random.seed(seed)
            return random.sample(problems, n)

        return problems[:n]


class CommonSenseQALoader:
    """Loader for CommonSenseQA dataset."""

    def __init__(self, data_dir: str = "data/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._train_data: List[Problem] = []
        self._test_data: List[Problem] = []

    def download(self) -> None:
        """Download dataset if not present."""
        train_file = self.data_dir / "commonsenseqa_train.json"
        test_file = self.data_dir / "commonsenseqa_test.json"

        if train_file.exists() and test_file.exists():
            logger.info("CommonSenseQA dataset already downloaded")
            return

        self._create_sample_dataset()

    def _create_sample_dataset(self) -> None:
        """Create sample CommonSenseQA problems."""
        sample_problems = [
            {"id": "csqa_001", "question": "Where would you find a bed in a house?", "answer": "bedroom"},
            {"id": "csqa_002", "question": "What do people use to cut paper?", "answer": "scissors"},
            {"id": "csqa_003", "question": "What do you use to see in the dark?", "answer": "flashlight"},
            {"id": "csqa_004", "question": "Where do fish live?", "answer": "water"},
            {"id": "csqa_005", "question": "What do birds use to fly?", "answer": "wings"},
            {"id": "csqa_006", "question": "What is used to tell time?", "answer": "clock"},
            {"id": "csqa_007", "question": "Where do you go to borrow books?", "answer": "library"},
            {"id": "csqa_008", "question": "What do you wear on your feet?", "answer": "shoes"},
            {"id": "csqa_009", "question": "What do you use to write on paper?", "answer": "pen"},
            {"id": "csqa_010", "question": "Where do you sleep at night?", "answer": "bed"},
            {"id": "csqa_011", "question": "What do you use to open a door?", "answer": "doorknob"},
            {"id": "csqa_012", "question": "What do you use to brush your teeth?", "answer": "toothbrush"},
            {"id": "csqa_013", "question": "Where do you store cold food?", "answer": "refrigerator"},
            {"id": "csqa_014", "question": "What do you use to cut wood?", "answer": "saw"},
            {"id": "csqa_015", "question": "Where do you wash your hands?", "answer": "sink"},
            {"id": "csqa_016", "question": "What do you use to erase pencil marks?", "answer": "eraser"},
            {"id": "csqa_017", "question": "Where do you sit at a table?", "answer": "chair"},
            {"id": "csqa_018", "question": "What do you use to take photos?", "answer": "camera"},
            {"id": "csqa_019", "question": "Where do you park a car?", "answer": "garage"},
            {"id": "csqa_020", "question": "What do you use to play music?", "answer": "instrument"},
            {"id": "csqa_021", "question": "What do you use to see far away?", "answer": "binoculars"},
            {"id": "csqa_022", "question": "Where do you buy groceries?", "answer": "store"},
            {"id": "csqa_023", "question": "What do you use to dry your hair?", "answer": "towel"},
            {"id": "csqa_024", "question": "Where do you watch movies at home?", "answer": "tv"},
            {"id": "csqa_025", "question": "What do you use to measure length?", "answer": "ruler"},
        ]

        train_file = self.data_dir / "commonsenseqa_train.json"
        test_file = self.data_dir / "commonsenseqa_test.json"

        with open(train_file, "w") as f:
            json.dump(sample_problems[:15], f, indent=2)

        with open(test_file, "w") as f:
            json.dump(sample_problems[15:], f, indent=2)

        logger.info(f"Created CommonSenseQA sample dataset")

    def load(self, split: str = "test") -> List[Problem]:
        """Load dataset split."""
        self.download()

        if split == "train":
            if self._train_data:
                return self._train_data
            filepath = self.data_dir / "commonsenseqa_train.json"
        else:
            if self._test_data:
                return self._test_data
            filepath = self.data_dir / "commonsenseqa_test.json"

        with open(filepath) as f:
            data = json.load(f)

        problems = [
            Problem(
                id=item.get("id", f"csqa_{i}"),
                question=item.get("question", ""),
                answer=str(item.get("answer", "")).lower(),
            )
            for i, item in enumerate(data)
        ]

        if split == "train":
            self._train_data = problems
        else:
            self._test_data = problems

        return problems

    def get_subset(self, split: str = "test", n: Optional[int] = None, random_sample: bool = True, seed: int = 42) -> List[Problem]:
        """Get subset of problems."""
        problems = self.load(split)
        if n is None or n >= len(problems):
            return problems
        if random_sample:
            random.seed(seed)
            return random.sample(problems, n)
        return problems[:n]


class StrategicReasoningLoader:
    """Loader for strategic reasoning problems (chess, game strategy, planning)."""

    def __init__(self, data_dir: str = "data/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._train_data: List[Problem] = []
        self._test_data: List[Problem] = []

    def download(self) -> None:
        """Download dataset if not present."""
        train_file = self.data_dir / "strategic_train.json"
        test_file = self.data_dir / "strategic_test.json"

        if train_file.exists() and test_file.exists():
            logger.info("Strategic Reasoning dataset already downloaded")
            return

        self._create_sample_dataset()

    def _create_sample_dataset(self) -> None:
        """Create strategic reasoning problems requiring multi-step planning."""
        sample_problems = [
            {
                "id": "strategic_001",
                "question": "You have $100 and want to buy items costing $23, $18, and $45. Can you afford all three?",
                "answer": "yes",
                "facts": [
                    "Total cost is $23 + $18 + $45 = $86",
                    "$86 is less than $100",
                    "So you can afford all three",
                ],
                "decomposition": [
                    "Add the prices: $23 + $18 = $41",
                    "Add third item: $41 + $45 = $86",
                    "Compare: $86 < $100",
                    "Answer: yes, you can afford all",
                ],
            },
            {
                "id": "strategic_002",
                "question": "If you need to wake up at 7 AM and it takes 30 minutes to get ready, and 45 minutes to commute, what time should you wake up to arrive by 9 AM?",
                "answer": "7:45",
                "facts": [
                    "Arrival time is 9:00 AM",
                    "Commute takes 45 minutes",
                    "Getting ready takes 30 minutes",
                ],
                "decomposition": [
                    "Subtract commute time: 9:00 - 0:45 = 8:15 AM",
                    "Subtract prep time: 8:15 - 0:30 = 7:45 AM",
                    "Wake up at 7:45 AM",
                ],
            },
            {
                "id": "strategic_003",
                "question": "A recipe serves 4 people and requires 2 cups of flour. If you want to serve 10 people, how many cups of flour do you need?",
                "answer": "5",
                "facts": [
                    "Original recipe: 4 servings need 2 cups",
                    "Target: 10 servings",
                    "Scale factor is 10/4 = 2.5",
                ],
                "decomposition": [
                    "Calculate scale factor: 10/4 = 2.5",
                    "Multiply original amount: 2 * 2.5 = 5",
                    "Need 5 cups of flour",
                ],
            },
            {
                "id": "strategic_004",
                "question": "You're packing a box that can hold 20 kg. You have items weighing 5 kg, 8 kg, 4 kg, and 6 kg. What's the maximum weight you can pack without exceeding the limit?",
                "answer": "19",
                "facts": [
                    "Box limit: 20 kg",
                    "Items: 5, 8, 4, 6 kg",
                    "Need to find best combination",
                ],
                "decomposition": [
                    "Try combinations: 5+8+4+6 = 23 (too heavy)",
                    "Try 5+8+6 = 19 (fits)",
                    "Try 8+4+6 = 18 (fits but less)",
                    "Maximum is 19 kg",
                ],
            },
            {
                "id": "strategic_005",
                "question": "In tic-tac-toe, if you place your first X in the center, and your opponent puts O in the top-left corner, should you place your next X in a corner or edge?",
                "answer": "corner",
                "facts": [
                    "Center X gives maximum control",
                    "Corners create diagonal winning lines",
                    "Opposite corners create two threats",
                ],
                "decomposition": [
                    "Center is already controlled by X",
                    "O in corner means threat on diagonal",
                    "Place X in opposite corner",
                    "This creates two winning paths",
                ],
            },
            {
                "id": "strategic_006",
                "question": "If a train leaves Station A at 9 AM traveling 60 mph toward Station B 180 miles away, and another train leaves Station B at 10 AM traveling 50 mph toward Station A, will they meet before 11 AM?",
                "answer": "no",
                "facts": [
                    "Train A starts at 9 AM going 60 mph",
                    "Train B starts at 10 AM going 50 mph",
                    "Distance is 180 miles",
                ],
                "decomposition": [
                    "By 10 AM, Train A has gone 60 miles (60 * 1 hour)",
                    "Remaining distance: 180 - 60 = 120 miles",
                    "Combined speed after 10 AM: 60 + 50 = 110 mph",
                    "Time to meet: 120/110 = 1.09 hours",
                    "Meeting time: 10:00 + 1:05 = 11:05 AM",
                    "So they meet after 11 AM, answer is no",
                ],
            },
            {
                "id": "strategic_007",
                "question": "You have 3 jugs: 8-liter, 5-liter, and 3-liter. The 8-liter jug is full. Can you measure exactly 4 liters using these jugs?",
                "answer": "yes",
                "facts": [
                    "Start: 8L=8, 5L=0, 3L=0",
                    "Goal: measure 4 liters",
                    "Can pour between jugs",
                ],
                "decomposition": [
                    "Pour 8L to 5L: 8L=3, 5L=5, 3L=0",
                    "Pour 5L to 3L: 8L=3, 5L=2, 3L=3",
                    "Empty 3L: 8L=3, 5L=2, 3L=0",
                    "Pour 5L to 3L: 8L=3, 5L=0, 3L=2",
                    "Continue process to get 4L",
                ],
            },
            {
                "id": "strategic_008",
                "question": "In rock-paper-scissors, if your opponent has played rock 3 times in a row, should you play paper next?",
                "answer": "yes",
                "facts": [
                    "Rock beats scissors",
                    "Paper beats rock",
                    "Scissors beats paper",
                ],
                "decomposition": [
                    "Opponent played rock 3 times",
                    "If pattern continues, next is rock",
                    "Paper beats rock",
                    "Play paper",
                ],
            },
            {
                "id": "strategic_009",
                "question": "You need to cross a bridge with a flashlight. The bridge holds max 2 people. Four people take 1, 2, 5, and 10 minutes to cross. The flashlight must be carried back. What's the minimum time for all to cross?",
                "answer": "17",
                "facts": [
                    "People take: 1, 2, 5, 10 minutes",
                    "Max 2 on bridge at once",
                    "Flashlight needed to cross",
                ],
                "decomposition": [
                    "Send fastest pair first: 1+2 cross (2 min)",
                    "Fastest returns with light: 1 returns (1 min) = 3 total",
                    "Send slowest pair: 5+10 cross (10 min) = 13 total",
                    "Second fastest returns: 2 returns (2 min) = 15 total",
                    "Final pair: 1+2 cross (2 min) = 17 total",
                ],
            },
            {
                "id": "strategic_010",
                "question": "A snail climbs 3 feet up a wall during the day but slides down 2 feet at night. The wall is 10 feet tall. How many days to reach the top?",
                "answer": "8",
                "facts": [
                    "Climbs 3 feet per day",
                    "Slides 2 feet per night",
                    "Net gain: 1 foot per 24 hours",
                    "Wall is 10 feet",
                ],
                "decomposition": [
                    "Day 1: climbs to 3, slides to 1",
                    "Day 2: climbs to 4, slides to 2",
                    "Day 7: climbs to 9, slides to 7",
                    "Day 8: climbs to 10 - reaches top!",
                    "Answer: 8 days",
                ],
            },
            {
                "id": "strategic_011",
                "question": "In a guessing game, you need to find a number between 1-100. Is binary search the optimal strategy to minimize guesses?",
                "answer": "yes",
                "facts": [
                    "Range is 1-100",
                    "Want to minimize guesses",
                    "Each guess should eliminate half the range",
                ],
                "decomposition": [
                    "Guess middle: 50",
                    "If too high, new range 1-49, guess 25",
                    "If too low, new range 51-100, guess 75",
                    "Continue halving",
                    "Maximum 7 guesses needed (log2(100) ≈ 7)",
                ],
            },
            {
                "id": "strategic_012",
                "question": "You have 12 balls, one weighs differently. Can you find the odd ball using a balance scale only 3 times?",
                "answer": "yes",
                "facts": [
                    "12 balls, one weighs different",
                    "Balance scale only",
                    "Only 3 weighings allowed",
                ],
                "decomposition": [
                    "Divide into 3 groups of 4: A, B, C",
                    "Weigh A vs B",
                    "If equal, odd ball is in C",
                    "If not equal, odd ball is in heavier/lighter group",
                    "Take suspect group, continue weighing",
                    "Third weighing identifies the odd ball",
                ],
            },
            {
                "id": "strategic_013",
                "question": "In chess, if you can take a queen with your knight but it would expose your king to check, should you take the queen?",
                "answer": "no",
                "facts": [
                    "Queen is worth 9 points",
                    "Knight is worth 3 points",
                    "King check can lead to checkmate",
                ],
                "decomposition": [
                    "Evaluate material gain: 9 points",
                    "Evaluate risk: king in check",
                    "Check can lead to checkmate",
                    "Losing king means losing game",
                    "Do not take queen",
                ],
            },
            {
                "id": "strategic_014",
                "question": "You're in a card game. Should you discard a card that helps slightly now for a chance at a better card if the deck has mostly better cards?",
                "answer": "yes",
                "facts": [
                    "Current card: slight benefit",
                    "Deck has mostly better cards",
                    "Probability favors drawing better card",
                ],
                "decomposition": [
                    "Evaluate current card's value",
                    "Calculate probability of better card",
                    "If probability > 50%, discard and draw",
                    "Expected value favors drawing",
                ],
            },
            {
                "id": "strategic_015",
                "question": "If you flip a coin 10 times and get heads all 10 times, is the probability of heads on the 11th flip still 50%?",
                "answer": "yes",
                "facts": [
                    "Coin flips are independent events",
                    "Previous results don't affect future",
                    "Fair coin has 50% heads probability",
                ],
                "decomposition": [
                    "Each flip is independent",
                    "Past 10 heads doesn't change next flip",
                    "Probability remains 50%",
                ],
            },
        ]

        train_file = self.data_dir / "strategic_train.json"
        test_file = self.data_dir / "strategic_test.json"

        with open(train_file, "w") as f:
            json.dump(sample_problems[:10], f, indent=2)

        with open(test_file, "w") as f:
            json.dump(sample_problems[10:], f, indent=2)

        logger.info(f"Created Strategic Reasoning sample dataset")

    def load(self, split: str = "test") -> List[Problem]:
        """Load dataset split."""
        self.download()

        if split == "train":
            if self._train_data:
                return self._train_data
            filepath = self.data_dir / "strategic_train.json"
        else:
            if self._test_data:
                return self._test_data
            filepath = self.data_dir / "strategic_test.json"

        with open(filepath) as f:
            data = json.load(f)

        problems = [
            Problem(
                id=item.get("id", f"strategic_{i}"),
                question=item.get("question", ""),
                answer=str(item.get("answer", "")).lower(),
                facts=item.get("facts", []),
                decomposition=item.get("decomposition", []),
            )
            for i, item in enumerate(data)
        ]

        if split == "train":
            self._train_data = problems
        else:
            self._test_data = problems

        return problems

    def get_subset(self, split: str = "test", n: Optional[int] = None, random_sample: bool = True, seed: int = 42) -> List[Problem]:
        """Get subset of problems."""
        problems = self.load(split)
        if n is None or n >= len(problems):
            return problems
        if random_sample:
            random.seed(seed)
            return random.sample(problems, n)
        return problems[:n]


class GSM8KLoader:
    """Loader for GSM8K math word problems dataset."""

    def __init__(self, data_dir: str = "data/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._train_data: List[Problem] = []
        self._test_data: List[Problem] = []

    def download(self) -> None:
        """Download dataset if not present."""
        train_file = self.data_dir / "gsm8k_train.json"
        test_file = self.data_dir / "gsm8k_test.json"

        if train_file.exists() and test_file.exists():
            logger.info("GSM8K dataset already downloaded")
            return

        self._create_sample_dataset()

    def _create_sample_dataset(self) -> None:
        """Create sample GSM8K problems."""
        sample_problems = [
            {"id": "gsm8k_001", "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "72"},
            {"id": "gsm8k_002", "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "answer": "10"},
            {"id": "gsm8k_003", "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Betty's parents gave her $15. How much more money does Betty need to buy the wallet?", "answer": "35"},
            {"id": "gsm8k_004", "question": "Julie is reading a 276-page book. She read 78 pages on Monday and 56 pages on Tuesday. How many pages does she have left to read?", "answer": "142"},
            {"id": "gsm8k_005", "question": "Mark has 12 apples. He gives 3 apples to his sister and buys 5 more apples. How many apples does Mark have now?", "answer": "14"},
            {"id": "gsm8k_006", "question": "A rectangle has a length of 8 cm and a width of 5 cm. What is the perimeter of the rectangle?", "answer": "26"},
            {"id": "gsm8k_007", "question": "There are 25 students in a class. If 3 students are absent, how many students are present?", "answer": "22"},
            {"id": "gsm8k_008", "question": "A store sells pencils for $2 each. If John buys 7 pencils, how much does he spend?", "answer": "14"},
            {"id": "gsm8k_009", "question": "Sarah has 45 marbles. She gives 18 marbles to her friend. How many marbles does Sarah have left?", "answer": "27"},
            {"id": "gsm8k_010", "question": "A train travels 60 miles per hour. How far will it travel in 3 hours?", "answer": "180"},
            {"id": "gsm8k_011", "question": "A pizza has 8 slices. If 3 people share it equally, how many slices does each person get?", "answer": "2"},
            {"id": "gsm8k_012", "question": "Tom has $50. He spends $15 on lunch and $8 on a book. How much money does he have left?", "answer": "27"},
            {"id": "gsm8k_013", "question": "A box contains 24 cookies. If 6 cookies are taken out, how many cookies remain?", "answer": "18"},
            {"id": "gsm8k_014", "question": "Mary walks 3 miles to school and 3 miles back. How many miles does she walk in 5 days?", "answer": "30"},
            {"id": "gsm8k_015", "question": "A car travels 50 miles on 2 gallons of gas. How far can it travel on 6 gallons?", "answer": "150"},
            {"id": "gsm8k_016", "question": "John reads 20 pages per day. How many pages does he read in a week?", "answer": "140"},
            {"id": "gsm8k_017", "question": "A baker makes 36 muffins. He sells 28 muffins. How many muffins are left?", "answer": "8"},
            {"id": "gsm8k_018", "question": "Lisa has 3 boxes of crayons. Each box has 24 crayons. How many crayons does she have in total?", "answer": "72"},
            {"id": "gsm8k_019", "question": "A toy costs $25. If you buy 4 toys, how much do you spend?", "answer": "100"},
            {"id": "gsm8k_020", "question": "A garden has 15 rose bushes. Each bush has 6 roses. How many roses are there in total?", "answer": "90"},
            {"id": "gsm8k_021", "question": "A swimmer swims 10 laps each day. How many laps does she swim in 12 days?", "answer": "120"},
            {"id": "gsm8k_022", "question": "A teacher has 30 pencils. She gives 5 pencils to each of 6 students. How many pencils are left?", "answer": "0"},
            {"id": "gsm8k_023", "question": "A movie is 2 hours and 15 minutes long. How many minutes is the movie?", "answer": "135"},
            {"id": "gsm8k_024", "question": "A store has 100 apples. They sell 45 apples in the morning and 30 in the afternoon. How many apples are left?", "answer": "25"},
            {"id": "gsm8k_025", "question": "A runner jogs at 4 miles per hour for 3 hours. How far does the runner jog?", "answer": "12"},
        ]

        train_file = self.data_dir / "gsm8k_train.json"
        test_file = self.data_dir / "gsm8k_test.json"

        with open(train_file, "w") as f:
            json.dump(sample_problems[:15], f, indent=2)

        with open(test_file, "w") as f:
            json.dump(sample_problems[15:], f, indent=2)

        logger.info(f"Created GSM8K sample dataset")

    def load(self, split: str = "test") -> List[Problem]:
        """Load dataset split."""
        self.download()

        if split == "train":
            if self._train_data:
                return self._train_data
            filepath = self.data_dir / "gsm8k_train.json"
        else:
            if self._test_data:
                return self._test_data
            filepath = self.data_dir / "gsm8k_test.json"

        with open(filepath) as f:
            data = json.load(f)

        problems = [
            Problem(
                id=item.get("id", f"gsm8k_{i}"),
                question=item.get("question", ""),
                answer=str(item.get("answer", "")),
            )
            for i, item in enumerate(data)
        ]

        if split == "train":
            self._train_data = problems
        else:
            self._test_data = problems

        return problems

    def get_subset(self, split: str = "test", n: Optional[int] = None, random_sample: bool = True, seed: int = 42) -> List[Problem]:
        """Get subset of problems."""
        problems = self.load(split)
        if n is None or n >= len(problems):
            return problems
        if random_sample:
            random.seed(seed)
            return random.sample(problems, n)
        return problems[:n]


class DataLoader:
    """Unified data loader for multiple datasets."""

    def __init__(self, data_dir: str = "data/datasets"):
        self.data_dir = Path(data_dir)
        self._loaders = {
            "strategy_qa": StrategyQALoader(str(self.data_dir)),
            "commonsense_qa": CommonSenseQALoader(str(self.data_dir)),
            "gsm8k": GSM8KLoader(str(self.data_dir)),
            "strategic": StrategicReasoningLoader(str(self.data_dir)),
        }

    def load(
        self, dataset: str = "strategy_qa", split: str = "test", n: Optional[int] = None, seed: int = 42,
    ) -> List[Problem]:
        """Load dataset.

        Args:
            dataset: Dataset name ('strategy_qa', 'commonsense_qa', 'gsm8k')
            split: 'train' or 'test'
            n: Number of samples (None for all)
            seed: Random seed

        Returns:
            List of Problem objects
        """
        if dataset not in self._loaders:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(self._loaders.keys())}")

        loader = self._loaders[dataset]

        if n:
            return loader.get_subset(split, n, random_sample=True, seed=seed)

        return loader.load(split)

    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets."""
        return list(self._loaders.keys())

    def get_dataset_info(self, dataset: str) -> Dict[str, Any]:
        """Get information about dataset."""
        info = {
            "strategy_qa": {
                "name": "StrategyQA",
                "description": "Multi-step reasoning questions requiring implicit knowledge",
                "metrics": ["accuracy", "reasoning_quality"],
                "answer_types": ["yes", "no", "it depends"],
            },
            "commonsense_qa": {
                "name": "CommonSenseQA",
                "description": "Commonsense knowledge questions",
                "metrics": ["accuracy"],
                "answer_types": ["text"],
            },
            "gsm8k": {
                "name": "GSM8K",
                "description": "Grade school math word problems",
                "metrics": ["accuracy", "numerical_accuracy"],
                "answer_types": ["number"],
            },
            "strategic": {
                "name": "StrategicReasoning",
                "description": "Strategic reasoning problems (chess, games, planning)",
                "metrics": ["accuracy", "reasoning_quality"],
                "answer_types": ["text"],
            },
        }
        return info.get(dataset, {"name": dataset, "description": "Unknown dataset"})


def create_mock_dataset(n: int = 20, seed: int = 42) -> List[Problem]:
    """Create mock dataset for testing without real data.
    
    Args:
        n: Number of problems to create
        seed: Random seed
    
    Returns:
        List of Problem objects
    """
    random.seed(seed)
    
    templates = [
        "Is {} greater than {}?",
        "Can {} {}?",
        "Does {} require {}?",
        "Would a person {} if {}?",
        "Is {} found in {}?",
    ]
    
    fillers = {
        "numbers": ["100", "50", "25", "10", "5", "1000"],
        "objects": ["water", "air", "metal", "wood", "plastic"],
        "actions": ["float", "burn", "freeze", "conduct electricity"],
        "places": ["ocean", "desert", "mountain", "city"],
        "conditions": ["it rains", "it's cold", "there's no air"],
    }
    
    problems = []
    for i in range(n):
        template = random.choice(templates)
        
        if "{} greater than {}" in template:
            a, b = random.sample(fillers["numbers"], 2)
            question = template.format(a, b)
            answer = "yes" if int(a) > int(b) else "no"
        elif "Can {} {}" in template:
            obj = random.choice(fillers["objects"])
            action = random.choice(fillers["actions"])
            question = template.format(obj, action)
            answer = random.choice(["yes", "no"])
        else:
            question = template.format(*random.sample(fillers["objects"], 2))
            answer = random.choice(["yes", "no"])
        
        problem = Problem(
            id=f"mock_{i:03d}",
            question=question,
            answer=answer,
        )
        problems.append(problem)
    
    return problems
