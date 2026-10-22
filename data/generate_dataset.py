"""Generate a reproducible mock e-commerce catalog for the vector recommender.

Run from the project root:
    python data/generate_dataset.py
"""

from __future__ import annotations 

import csv
import itertools
import random
from collections import Counter
from pathlib import Path

RANDOM_SEED = 42
ITEMS_PER_CATEGORY = 200
MIN_PRICE, MAX_PRICE = 10.00, 750.00
OUTPUT_PATH = Path(__file__).resolve().parent / "products.csv"
FIELDNAMES = ["product_id", "title", "description", "category", "price"]

# Each item type maps to ((min_price, max_price), [features]).
# Features are item-specific vocabulary; use_cases are category-wide vocabulary.
CATALOG: dict[str, dict] = {
    "Electronics": {
        "brands": ["Voltix", "Auralis", "Nexora", "Kinetek", "Lumio", "Brightwave"],
        "modifiers": ["Pro", "Ultra", "Compact", "Smart", "Elite", "Max"],
        "items": {
            "Wireless Headphones": ((60, 450), ["active noise cancellation", "a 40-hour battery life", "plush memory-foam ear cushions", "low-latency Bluetooth 5.3"]),
            "Mechanical Keyboard": ((50, 250), ["hot-swappable tactile switches", "per-key RGB backlighting", "a detachable USB-C cable", "PBT double-shot keycaps"]),
            "Gaming Mouse": ((25, 180), ["a 26,000 DPI optical sensor", "an ultra-lightweight 58-gram shell", "programmable side buttons", "a 1000 Hz polling rate"]),
            "4K Monitor": ((220, 750), ["a 27-inch IPS panel", "HDR10 color accuracy", "a 144 Hz refresh rate", "USB-C power delivery"]),
            "Power Bank": ((20, 120), ["a 20,000 mAh battery", "65W fast charging", "dual USB-C ports", "pass-through charging"]),
            "Smartwatch": ((90, 550), ["continuous heart-rate monitoring", "built-in GPS", "an always-on AMOLED display", "a seven-day battery life"]),
            "Bluetooth Speaker": ((30, 300), ["360-degree sound", "IP67 waterproofing", "deep bass radiators", "a 24-hour playtime"]),
        },
        "use_cases": [
            "Built for gamers, remote workers, and tech enthusiasts who want reliable wireless performance.",
            "Pairs seamlessly with laptops, phones, and tablets for an upgraded digital setup.",
            "Ideal for streaming, gaming, and video calls with crisp, responsive performance.",
            "A smart gadget upgrade for anyone who lives on their devices.",
        ],
    },
    "Fitness Equipment": {
        "brands": ["IronPeak", "FlexCore", "Stridewell", "Pulsefit", "Titanform", "Zenmotion"],
        "modifiers": ["Pro", "Elite", "Studio", "Home", "Performance", "Core"],
        "items": {
            "Yoga Mat": ((20, 140), ["a non-slip textured surface", "6mm high-density cushioning", "sweat-resistant natural rubber", "alignment guide markings"]),
            "Adjustable Dumbbell Set": ((150, 700), ["quick-select weight dials", "a 5-to-52.5-pound weight range", "a compact storage tray", "knurled steel handles"]),
            "Resistance Band Set": ((15, 70), ["five color-coded tension levels", "anti-snap latex layers", "a padded door anchor", "cushioned foam handles"]),
            "Foam Roller": ((15, 60), ["high-density EVA foam", "a deep-tissue grid texture", "a hollow core that resists deformation", "a travel-friendly length"]),
            "Kettlebell": ((25, 180), ["a single-piece cast-iron body", "a wide powder-coated handle", "a flat base for stability", "color-coded weight markings"]),
            "Folding Treadmill": ((350, 750), ["a quiet 2.5 HP motor", "a 12-level incline", "a cushioned running deck", "a fold-flat frame"]),
            "Weighted Jump Rope": ((10, 45), ["ball-bearing swivel handles", "an adjustable steel cable", "removable handle weights", "a tangle-free coating"]),
        },
        "use_cases": [
            "Designed for home workouts, strength training, and cardio sessions.",
            "Helps athletes build strength, improve mobility, and speed up muscle recovery.",
            "A durable training essential for gym-goers and fitness beginners alike.",
            "Supports HIIT, yoga, and conditioning routines in any workout space.",
        ],
    },
    "Kitchenware": {
        "brands": ["Hearthstone", "Copperleaf", "Savora", "Fornello", "Kitchara", "Olivewood"],
        "modifiers": ["Pro", "Classic", "Precision", "Signature", "Essential", "Deluxe"],
        "items": {
            "Air Fryer": ((60, 250), ["a 6-quart nonstick basket", "rapid 360-degree hot air circulation", "eight one-touch cooking presets", "a dishwasher-safe crisper tray"]),
            "Espresso Machine": ((150, 750), ["a 15-bar Italian pump", "a steam wand for milk frothing", "precise PID temperature control", "an integrated conical burr grinder"]),
            "Chef's Knife": ((25, 220), ["a forged high-carbon steel blade", "a full-tang ergonomic handle", "a razor-sharp 15-degree edge", "a balanced 8-inch profile"]),
            "Nonstick Frying Pan": ((20, 150), ["a PFOA-free ceramic coating", "an induction-ready base", "even heat distribution", "a stay-cool riveted handle"]),
            "Electric Kettle": ((25, 130), ["variable temperature settings", "rapid 1500W boiling", "a keep-warm function", "automatic shut-off"]),
            "High-Speed Blender": ((60, 500), ["a 1400W motor", "hardened stainless steel blades", "a 64-ounce BPA-free pitcher", "pulse and smoothie presets"]),
            "Dutch Oven": ((45, 380), ["an enameled cast-iron body", "a tight-fitting self-basting lid", "superior heat retention", "oven-safe construction up to 500 degrees"]),
        },
        "use_cases": [
            "Perfect for home cooks who want restaurant-quality results in their own kitchen.",
            "Makes meal prep, baking, and everyday cooking faster and easier.",
            "A countertop essential for healthy meals, morning coffee, and family dinners.",
            "Built to handle searing, simmering, and blending through years of daily cooking.",
        ],
    },
    "Office Supplies": {
        "brands": ["Deskwise", "Paperline", "Ergonomix", "Workform", "Quillmark", "Officely"],
        "modifiers": ["Pro", "Classic", "Executive", "Essential", "Studio", "Premium"],
        "items": {
            "Office Chair": ((120, 750), ["adjustable lumbar support", "a breathable mesh backrest", "4D adjustable armrests", "a synchro-tilt recline mechanism"]),
            "Standing Desk": ((250, 750), ["dual-motor height adjustment", "four programmable height presets", "a scratch-resistant desktop", "an anti-collision sensor"]),
            "Monitor Arm": ((40, 220), ["gas-spring height adjustment", "full 360-degree rotation", "integrated cable management", "a sturdy desk clamp mount"]),
            "LED Desk Lamp": ((20, 140), ["five dimmable brightness levels", "adjustable color temperature", "a flicker-free LED panel", "a built-in USB charging port"]),
            "Desk Organizer": ((15, 70), ["compartments for pens and supplies", "a bamboo finish", "a built-in phone stand", "a stackable modular design"]),
            "Dotted Notebook": ((10, 40), ["120gsm bleed-resistant paper", "a lay-flat binding", "a durable hardcover", "numbered pages with an index"]),
            "Document Scanner": ((90, 450), ["duplex scanning at 30 pages per minute", "an automatic document feeder", "searchable PDF output", "cloud upload over Wi-Fi"]),
        },
        "use_cases": [
            "Ideal for home offices, remote work setups, and productive workspaces.",
            "Helps professionals stay organized, comfortable, and focused through long workdays.",
            "A smart addition to any desk setup for writing, planning, and paperwork.",
            "Designed for ergonomic comfort and productivity during the workday.",
        ],
    },
    "Outdoor Gear": {
        "brands": ["Trailforge", "Northridge", "Cairnline", "Wildpine", "Ridgeback", "Summitry"],
        "modifiers": ["Trail", "Summit", "Expedition", "Alpine", "Ultralight", "Basecamp"],
        "items": {
            "Backpacking Tent": ((90, 600), ["a waterproof rainfly", "lightweight aluminum poles", "a two-door vestibule design", "taped seams for storm protection"]),
            "Sleeping Bag": ((40, 350), ["a 20-degree temperature rating", "water-resistant down insulation", "a draft-blocking hood", "a compact compression sack"]),
            "Hiking Backpack": ((60, 320), ["an adjustable torso length", "a ventilated mesh back panel", "a hydration reservoir sleeve", "a built-in rain cover"]),
            "Trekking Poles": ((30, 180), ["carbon fiber shafts", "quick-lock length adjustment", "cork grips", "shock-absorbing tips"]),
            "Camping Stove": ((25, 180), ["a push-button piezo igniter", "a wind-resistant burner", "a fast boil time", "fold-down pot supports"]),
            "Headlamp": ((15, 90), ["a 400-lumen beam", "a red night-vision mode", "a rechargeable battery", "an IPX7 waterproof rating"]),
            "Insulated Water Bottle": ((15, 60), ["double-wall vacuum insulation", "a leakproof lid", "24-hour cold retention", "a durable powder-coated finish"]),
        },
        "use_cases": [
            "Built for hiking, camping, and backpacking adventures in rugged terrain.",
            "A trail-ready essential for multi-day treks and weekend camping trips.",
            "Keeps outdoor explorers prepared for mountains, forests, and changing weather.",
            "Designed for backpackers who need durable, weather-ready gear in the wilderness.",
        ],
    },
}


def build_category_rows(rng: random.Random, category: str, spec: dict) -> list[dict]:
    """Sample unique (brand, modifier, item) combos and render one row per combo."""
    combos = list(itertools.product(spec["brands"], spec["modifiers"], spec["items"]))
    if len(combos) < ITEMS_PER_CATEGORY:
        raise ValueError(f"{category}: only {len(combos)} unique titles possible, need {ITEMS_PER_CATEGORY}")

    rows = []
    for brand, modifier, item in rng.sample(combos, ITEMS_PER_CATEGORY):
        (low, high), features = spec["items"][item]
        feature_a, feature_b = rng.sample(features, 2)
        use_case = rng.choice(spec["use_cases"])
        price = min(max(rng.uniform(low, high), MIN_PRICE), MAX_PRICE)

        rows.append({
            "title": f"{brand} {modifier} {item}",
            "description": f"The {brand} {modifier} {item} combines {feature_a} with {feature_b}. {use_case}",
            "category": category,
            "price": f"{price:.2f}",
        })
r

def validate(rows: list[dict]) -> None:
    """Fail loudly if the dataset breaks its contract (raise, not assert: asserts vanish under python -O)."""
    expected = ITEMS_PER_CATEGORY * len(CATALOG)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, got {len(rows)}")
    if [r["product_id"] for r in rows] != list(range(1, expected + 1)):
        raise ValueError("product_id must run 1..N with no gaps or duplicates")
    if len({r["title"] for r in rows}) != len(rows):
        raise ValueError("Duplicate titles found")
    for r in rows:
        if not MIN_PRICE <= float(r["price"]) <= MAX_PRICE:
            raise ValueError(f"Price out of range: {r}")


def main() -> None:
    rng = random.Random(RANDOM_SEED)  # local RNG: reproducible, no global state

    rows: list[dict] = []
    for category, spec in CATALOG.items():
        rows.extend(build_category_rows(rng, category, spec))

    rng.shuffle(rows)  # interleave categories so IDs carry no category signal
    for product_id, row in enumerate(rows, start=1):
        row["product_id"] = product_id

    validate(rows)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} products to {OUTPUT_PATH}")
    for category, count in sorted(Counter(r["category"] for r in rows).items()):
        print(f"  {category:<18} {count}")


if __name__ == "__main__":
    main()
