"""
LLM-based action-tag classifier for motion labels.

Batch-classifies multiple action names in a single LLM call, with
dual-layer caching (in-memory + single disk file).

Usage::

    from data_loaders.truebones.truebones_utils.motion_labels_llm import (
        classify_action_tags_batch,
        ACTION_TAGS,
    )
    # Without species context:
    results = classify_action_tags_batch(["WalkLoop", "ChargeAttack", "Idle"])
    # → {"WalkLoop": ["locomotion"], "ChargeAttack": ["locomotion", "attack"], "Idle": ["idle"]}

    # With species context (LLM receives species info, cache keyed by species):
    results = classify_action_tags_batch(
        ["WalkLoop", "ChargeAttack"],
        object_type="Horse",
    )
"""
from __future__ import annotations

import json
import os
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# Tag definitions
# ---------------------------------------------------------------------------

ACTION_TAGS: tuple[str, ...] = (
    "idle",
    "locomotion",
    "getup",
    "swim",
    "fly",
    "jump",
    "turn",
    "attack",
    "gethurt",
    "rest",
    "emote",
    "interact",
    "death",
    "fall",
    "unknown",
)

_TAG_DESCRIPTIONS: dict[str, str] = {
    "idle": "Stationary upright pose: standing, waiting, breathing, tpose, bindpose, stance, ready, "
            "steady. Body stays in place with no significant displacement. Alert / combat-ready "
            "readiness is still idle, not attack.",
    "locomotion": "Terrestrial displacement with surface contact: walk, run, trot, gallop, sprint, "
                  "pace, dash, sneak, stalk, crawl, march, step, stride, jog, canter, scamper, scurry, "
                  "back-up, retreat, slide-back, climb, scramble. Continuous or discrete movement "
                  "across the ground or up/along a surface (climbing counts as locomotion). "
                  "NOT in-place turning (use 'turn'); NOT getting up from a downed state (use 'getup').",
    "getup": "Recovering to a standing posture from a downed, fallen, prone, dead, or wounded state: "
             "get-up, rise, stand-up, recover-to-feet, struggle-up, die-getup, fall-getup. The "
             "transition FROM lying/fallen/downed/dead BACK TO standing. When a name pairs a "
             "downed-state token (die/dead/fall/down/ko/knocked) WITH a recovery token "
             "(getup/rise/stand/recover), it describes RECOVERING from that state → tag 'getup' "
             "ONLY, NOT 'death'/'fall'. Its own category, distinct from locomotion.",
    "swim": "Aquatic locomotion: swim, float(in water), slither(in water), wade. Movement through or on water.",
    "fly": "Aerial locomotion: fly, glide, soar, hover, flap-in-air, takeoff, take-off, land(from "
           "flight), landing, alight, descend, float-up, float-down, ride-wind, wind-ride. Body "
           "airborne, wings active — INCLUDING taking off into the air, gliding/soaring, and coming "
           "down to land or descending from flight. For a flying species (bird, bat, pterosaur, "
           "dragon) treat take-off, landing, and descent as 'fly', NOT 'jump'. Wing movement while "
           "standing on the ground is NOT fly (use 'emote' or 'interact').",
    "jump": "Discrete aerial maneuver by a GROUND creature: jump, hop, leap, pounce, spring, bound, "
            "vault, land(from a jump/leap), dive(lunge). A single explosive launch with a brief "
            "aerial phase, or the landing from such a jump. For a flying species, taking off and "
            "landing belong to 'fly', NOT 'jump'.",
    "turn": "Rotation or reorientation of the body: turn, spin, pivot, rotate, circle, strafe. "
            "Pair 'turn' with a movement medium ONLY when the turn travels through space: a ground "
            "turn/circle/strafe that covers distance = ['turn','locomotion'], a flier banking to land "
            "= ['turn','fly'], a swimmer circling = ['turn','swim']. An in-place rotation that stays "
            "put — a stationary Spin/Pivot/Turn — is just ['turn'], with NO movement tag. Add a "
            "further tag only when the name contains another action (CircleBite = "
            "['turn','locomotion','attack']). A directional suffix on a head/voice action (LookRight, "
            "RoarLeft) is facing, not a turn — that is 'emote', not ['turn',...].",
    "attack": "Offensive combat the creature DEALS to a target: bite, strike, slash, kick, punch, "
              "gore, headbutt, claw, whip, shoot, charge, chase, smash, slam, swipe, snap, rip, "
              "peck, sting, tail-whip, kill(offensive). Physical aggression directed at a target — "
              "the creature is the one delivering the blow. A name ending in 'Hit'/'Hurt'/'Impact'/"
              "'Shot' instead means the creature is RECEIVING the blow → use 'gethurt', not 'attack'. "
              "Also threat/intimidation displays with aggressive intent (bristling, baring fangs, "
              "angry/enraged posturing) — these usually also get 'emote'.",
    "gethurt": "Taking damage / receiving a blow: hurt, impact, knocked, stun, limp, struck, damaged, "
               "hit(received), shot(received). The creature is on the receiving end of combat — the "
               "patient side of attack. A name ending in '...Hit' (e.g. AntHit, BodyHit) denotes "
               "BEING hit → 'gethurt'. Use 'attack' for the agent delivering the blow "
               "(strike/bite/slash), 'gethurt' for the one suffering it.",
    "rest": "Low-posture stationary: sit, lie, sleep, crouch, kneel, lay, settle, hide. "
            "Body is close to the ground, not standing upright.",
    "emote": "Expression and communication: dance, play, taunt, celebrate, wave, beg, gesture, bark, "
             "howl, roar, growl, bleat, hiss, alert, warn, look, sniff, smell, bellow, coo, squawk, "
             "scream, rutting, ground-paw display, flinch, recoil, stagger, twitch, flip, buck, rear, "
             "defend, cower, stabilize. Broadcasting intent or emotion — whether voluntary (social) "
             "or involuntary (defensive/nervous reaction). Ground-based wing/flap displays belong "
             "here. Aggressive threat displays also get 'attack'.",
    "interact": "Interaction with self or environment: eat, feast, graze, drink, chew, swallow, "
                "catch-fish, bite-at-food, clean, preen, scratch(self), shake, shiver, yawn, itch, "
                "puke, stretch, tired, sick, wake-up, dig(self-grooming). Actions directed at the "
                "character's own body (self-maintenance) or consuming/interacting with objects.",
    "death": "Death sequence: die, dead, dying, collapse(final), knockout, defeat, expire. The "
             "character is rendered lifeless or is dying. A character performing a kill is 'attack', "
             "NOT 'death' — only use 'death' when the character itself dies/expires.",
    "fall": "Loss of balance: fall, falling, fallen, drop, tumble, slip. "
            "Character loses footing and descends uncontrollably.",
    "unknown": "Use ONLY when the action name is unrecognizable, garbled, or too ambiguous to map to "
               "any other tag with confidence — NOT a substitute for 'idle'.",
}

# ---------------------------------------------------------------------------
# Deterministic overrides
# ---------------------------------------------------------------------------
# A small set of idiosyncratic dataset names the LLM reliably mis-tags, where
# there is no clean general principle to lean on. These win over both cache and
# LLM and are resolved at read time (never sent to the LLM, never cached), so
# the system prompt can stay general instead of accumulating per-name patches.
# Keyed by exact CamelCase name; lookup normalizes (lowercase + strips trailing
# digits) so numbered variants like 'Squirly2' also match.
_OVERRIDE_TAGS_RAW: dict[str, list[str]] = {
    "Squirly": ["emote"],        # non-standard word for nervous fidgeting
    "Trottle": ["locomotion"],   # non-standard gait spelling
    "ChargedUp": ["idle"],       # readiness state, easily mis-read as attack
    "GroundFlap": ["emote"],     # ground wing display, easily mis-read as fly
    "TailWhip": ["attack"],      # tail strike, easily mis-read as emote
    "Specail": ["emote"],        # Camel, non-standard word for walking display
    "Wild1": ["emote"],          # Camel, semi-aggressive vocal display
    "Fancy": ["emote"],          # Pteranodon, dancing/courtship display
    "Flapergasted": ["emote"],   # non-standard word for being startled/flustered
    "EggTend": ["emote"],        # Raptor2, egg-tending behavior
    "Spin": ["turn"],            # no locomotion, just turn
    "ComeDown": ["fall"],        # Deer, dropping/collapsing down — mis-read as locomotion
    "FlyLoop": ["fly"],          # looping flight anim, no turn — LLM adds spurious 'turn' per-species
    "RunLoop": ["locomotion"],   # looping run anim, no turn — LLM adds spurious 'turn' (Horse)
    "WalkLoop": ["locomotion"],  # looping walk anim, no turn — spurious 'turn' (Spider)
    "GlideLoop": ["fly"],        # looping glide anim, no turn — spurious 'turn' (Eagle)
    "LookLeft": ["emote"],       # gaze/head turn = facing, not turn+locomotion (Trex)
    "LookRight": ["emote"],      # gaze/head turn = facing, not turn+locomotion (Trex)
    "RoarLeft": ["emote"],       # directional roar = facing, not turn+locomotion (Trex)
    "RoarRight": ["emote"],      # directional roar = facing, not turn+locomotion (Trex)
    "OutOfGround": ["getup"],    # Cricket/HermitCrab, emerging from ground to standing = getup
    "AntLeft": ["locomotion"],   # FireAnt moving left = directional locomotion, mis-read as emote
    "AntRight": ["locomotion"],  # FireAnt moving right = directional locomotion, mis-read as emote
    "FallGetUp": ["fall", "getup"],  # Raptor2, genuine fall-THEN-recover sequence (exception to rule 7)
    "Uppity": ["jump"],          # Roach, non-standard word for hopping up — LLM returns 'unknown'
    "Guns": ["emote"],           # Scorpion, raised pincers = threat display — LLM returns 'unknown'
    "HoofScrape": ["emote"],     # ground-paw display (per emote def), not self-maintenance
    "ScrapeHoof": ["emote"],     # same as HoofScrape, alternate name order (Camel)
    "FeetUp": ["emote"],         # Horse, rearing (front feet raised) display, mis-read as jump
    "Shake": ["interact"],       # body shake-off = self-maintenance; some species mis-tagged emote
    # NOTE: "Left"/"Right" are intentionally NOT overridden — they are species-dependent
    # (Roach Left = locomotion, Pigeon Left = emote/display), so a name-only key cannot
    # disambiguate. Roach_Left_810 is corrected directly in motion_metadata.json.
}


def _normalize_name_key(name: str) -> str:
    """Normalize an action name for override lookup: lowercase, drop trailing digits."""
    return re.sub(r"\d+$", "", str(name).strip().lower())


_OVERRIDE_TAGS: dict[str, list[str]] = {
    _normalize_name_key(name): tags for name, tags in _OVERRIDE_TAGS_RAW.items()
}


def _lookup_override(action_name: str) -> list[str] | None:
    tags = _OVERRIDE_TAGS.get(_normalize_name_key(action_name))
    return list(tags) if tags is not None else None

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

_CACHE_VERSION = "v10"

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "dataset", "truebones", "zoo", "truebones_processed", "cache",
)

_TAGS_CACHE_FILE = os.path.join(_CACHE_DIR, "action_tags_cache.json")
_LEGACY_CATEGORY_CACHE_FILE = os.path.join(_CACHE_DIR, "action_category_cache.json")

# In-memory cache: action_name → action tags. On first access the on-disk
# single-file cache is loaded into this dict; every LLM write flushes the
# whole dict back to disk.
_in_memory_cache: dict[str, list[str]] = {}
_cache_loaded_from_disk: bool = False

# Batch size for LLM calls — per-species upper bound.
# When regenerate_dataset_artifacts.py prefetches tags, it groups action names
# by object_type and calls classify_action_tags_batch once per species.
# If a single species has more than 50 uncached action names, this function
# further splits them into chunks of 50 for individual LLM requests.
_DEFAULT_BATCH_SIZE: int = 50

# Number of LLM batch requests to dispatch concurrently. Each batch is an
# independent chat completion, so when a species has more uncached names than
# one batch holds we fire several at once. Workers share the in-memory cache,
# the disk flush and the lazy client init, all guarded by the locks below.
_DEFAULT_MAX_CONCURRENCY: int = 4

# Guards mutation of _in_memory_cache + the disk flush from worker threads.
_cache_write_lock = threading.Lock()
# Guards the one-time lazy LLM client/model initialisation.
_client_init_lock = threading.Lock()
# Guards the one-time disk->memory cache load. Without it, concurrent species
# workers race: one flips _cache_loaded_from_disk and starts loading while the
# others see the flag already set, skip the load, and re-query an empty cache.
_cache_load_lock = threading.Lock()

# ---------------------------------------------------------------------------
# LLM client (lazy, follows retarget.py pattern)
# ---------------------------------------------------------------------------

_LLM_CLIENT = None
_LLM_MODEL: str | None = None


def _get_llm_client_and_model() -> tuple:
    """Return (client, model_name), initialising lazily on first call."""
    global _LLM_CLIENT, _LLM_MODEL
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT, _LLM_MODEL

    with _client_init_lock:
        # Re-check inside the lock: another thread may have initialised it
        # while we were waiting.
        if _LLM_CLIENT is not None:
            return _LLM_CLIENT, _LLM_MODEL
        return _init_llm_client_and_model()


def _init_llm_client_and_model() -> tuple:
    """Actually build the client/model. Must be called holding _client_init_lock."""
    global _LLM_CLIENT, _LLM_MODEL

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for LLM action classification. "
            "Install it with: pip install openai"
        )

    base_url = os.environ.get(
        "ACTION_LABEL_LLM_BASE_URL",
        os.environ.get("RETARGET_LLM_BASE_URL", "http://127.0.0.1:8066/v1"),
    )
    api_key = os.environ.get(
        "ACTION_LABEL_LLM_API_KEY",
        os.environ.get(
            "RETARGET_LLM_API_KEY",
            os.environ.get("OPENAI_API_KEY", ""),
        ),
    )
    _LLM_CLIENT = OpenAI(base_url=base_url, api_key=api_key or "sk-dummy")

    model_override = os.environ.get(
        "ACTION_LABEL_LLM_MODEL",
        os.environ.get("RETARGET_LLM_MODEL", ""),
    )
    if model_override:
        _LLM_MODEL = model_override
        print(f"[action_labels] LLM model set from env: {_LLM_MODEL}  endpoint: {base_url}")
    else:
        models = list(_LLM_CLIENT.models.list())
        if not models:
            raise RuntimeError(
                f"LLM endpoint {base_url} returned no models. "
                "Set ACTION_LABEL_LLM_MODEL to specify a model explicitly."
            )
        _LLM_MODEL = models[0].id
        print(f"[action_labels] LLM model auto-discovered: {_LLM_MODEL}  endpoint: {base_url}")

    return _LLM_CLIENT, _LLM_MODEL


# ---------------------------------------------------------------------------
# Single-file cache helpers
# ---------------------------------------------------------------------------

def _normalize_tag_list(raw_tags, valid_tags: set[str]) -> list[str]:
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        values = [raw_tags]
    elif isinstance(raw_tags, (list, tuple, set)):
        values = raw_tags
    else:
        values = [raw_tags]

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = str(value).strip().lower()
        if not tag or tag in seen or tag not in valid_tags:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized


def _load_cache_json(cache_path: str):
    if not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _ensure_cache_loaded() -> None:
    """Load the single-file cache into memory on first access (thread-safe).

    The flag is flipped only AFTER the load completes, under a lock, so
    concurrent species workers either do the load or block until it is done —
    none can observe a half-populated / empty cache and re-query the LLM.
    """
    global _cache_loaded_from_disk
    if _cache_loaded_from_disk:
        return
    with _cache_load_lock:
        # Re-check inside the lock: another thread may have finished loading
        # while we were waiting on it.
        if _cache_loaded_from_disk:
            return
        _load_cache_from_disk()
        _cache_loaded_from_disk = True


def _load_cache_from_disk() -> None:
    """Populate _in_memory_cache from disk. Must hold _cache_load_lock."""
    valid_tags = set(ACTION_TAGS)
    data = _load_cache_json(_TAGS_CACHE_FILE)
    if isinstance(data, dict) and data.get("_version") == _CACHE_VERSION:
        mapping = data.get("_mapping")
        if isinstance(mapping, dict):
            for key, value in mapping.items():
                tags = _normalize_tag_list(value, valid_tags)
                if tags:
                    _in_memory_cache[str(key)] = tags
        return

    # Best-effort migration for the legacy single-category cache.
    legacy = _load_cache_json(_LEGACY_CATEGORY_CACHE_FILE)
    if isinstance(legacy, dict):
        mapping = legacy.get("_mapping")
        if isinstance(mapping, dict):
            for key, value in mapping.items():
                tags = _normalize_tag_list(value, valid_tags)
                if tags:
                    _in_memory_cache[str(key)] = tags


def _flush_cache_to_disk() -> None:
    """Write the entire in-memory cache to the single JSON file."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    data = {
        "_version": _CACHE_VERSION,
        "_mapping": dict(sorted(_in_memory_cache.items())),
    }
    try:
        with open(_TAGS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _build_tag_list_text() -> str:
    lines: list[str] = []
    for tag in ACTION_TAGS:
        desc = _TAG_DESCRIPTIONS.get(tag, "")
        lines.append(f"- **{tag}**: {desc}")
    return "\n".join(lines)


def _build_system_message() -> str:
    tag_text = _build_tag_list_text()
    valid_list = ", ".join(f'"{c}"' for c in ACTION_TAGS)
    return (
        "You are an expert in character animation and motion classification. "
        "Given one or more action names (extracted from animation filenames, typically "
        "CamelCase or space-separated tokens describing what a 3D character is doing), "
        "classify each one into ONE OR MORE of the following action tags.\n\n"
        f"{tag_text}\n\n"
        "RULES (general principles — the tag descriptions above carry the vocabulary):\n"
        "1. Return every tag that clearly applies, but stay concise: most names need 1 tag, "
        "some legitimately need 2 or 3.\n"
        "2. Movement + intent compose into multiple tags. A run-and-bite name is "
        "['locomotion', 'attack']; a run-and-roar name is ['locomotion', 'emote'].\n"
        "3. Aggressive / threat displays broadcast both hostility and intent — give them BOTH "
        "'emote' and 'attack'.\n"
        "4. Combat vs feeding: a bite at a target is 'attack'; a bite at food is 'interact'.\n"
        "5. Agent vs patient: a character performing a kill is 'attack'; a character dying is 'death'. "
        "A name ending in 'Hit'/'Hurt'/'Impact'/'Shot' means the creature is BEING hit → 'gethurt', "
        "NOT 'attack' (e.g. AntHit, BodyHit → ['gethurt']).\n"
        "6. Turning that TRAVELS is movement; in-place rotation is not. Pair 'turn' with a movement "
        "tag ('locomotion'/'fly'/'swim') ONLY when the body actually moves through space — a circling "
        "run/strafe is ['turn','locomotion'], a flier banking to land is ['turn','fly'], CircleBite is "
        "['turn','locomotion','attack']. A stationary Spin/Pivot/Turn-in-place is just ['turn'] — do "
        "NOT add 'locomotion'. A directional suffix on a head or voice action is FACING, not travel: "
        "LookLeft/LookRight is ['emote'], RoarLeft/RoarRight is ['emote'] (no 'turn', no 'locomotion'). "
        "A '...Loop' suffix marks an animation loop and never adds 'turn' (FlyLoop is ['fly'], RunLoop "
        "is ['locomotion']). Getting up from a downed state is 'getup' not 'locomotion'.\n"
        "7. Downed+recovery compounds: when a name combines a downed-state token (die/dead/fall/down/"
        "ko) WITH a recovery token (getup/rise/stand/recover), it is RECOVERING → 'getup' ONLY, not "
        "'death'/'fall' (e.g. DieGetUp → ['getup']).\n"
        "8. Flying species: for birds/bats/pterosaurs/dragons, take-off, landing, descent, gliding "
        "and soaring are all 'fly' (NOT 'jump'); a 'CircleLand' for a flier is ['turn','fly'].\n"
        "9. Almost every named action of a recognized animal maps to a real tag. Reserve 'unknown' "
        "for genuinely garbled or meaningless tokens — never assign 'unknown' to an obvious action "
        "(FlyLoop, Strike, TakeOff, Landing, Glide…) or to a whole species' action set. Do NOT fall "
        "back to 'idle' either (idle is only for genuine stationary upright poses).\n"
        "10. Return arrays of unique lowercase tags from the valid list only.\n\n"
        f"Valid tags: [{valid_list}]\n\n"
        "Return ONLY valid JSON — no explanation, no markdown fences."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _cache_key(action_name: str, object_type: str | None) -> str:
    """Build a cache key from action name and optional object type."""
    if object_type:
        return f"{object_type}::{action_name}"
    return action_name


def classify_action_tags_batch(
    action_names: list[str],
    *,
    object_type: str | None = None,
    force_llm: bool = False,
    max_batch_size: int = _DEFAULT_BATCH_SIZE,
) -> dict[str, list[str]]:
    """Classify multiple action names into action tags using LLM with per-name caching.

    Names for a single species are classified serially in chunks of
    ``max_batch_size``. To classify several species concurrently, use
    :func:`classify_action_tags_by_species`.

    Args:
        action_names: List of action names to classify.
        object_type: Optional species/object type for context-aware classification.
            When provided, the LLM prompt includes the species info and the cache
            is keyed by ``(action_name, object_type)`` so the same action name
            can receive different tags for different species.
        force_llm: If True, bypass cache and force LLM call.
        max_batch_size: Maximum number of names per LLM call.
    """
    if not action_names:
        return {}

    unique_names = list(dict.fromkeys(action_names))
    result: dict[str, list[str]] = {}

    # Deterministic overrides win over both cache and LLM; they are never sent
    # to the LLM and never written to the cache.
    pending: list[str] = []
    for name in unique_names:
        override = _lookup_override(name)
        if override is not None:
            result[name] = override
            continue
        pending.append(name)

    uncached: list[str] = []
    if not force_llm:
        _ensure_cache_loaded()
        for name in pending:
            key = _cache_key(name, object_type)
            if key in _in_memory_cache:
                result[name] = list(_in_memory_cache[key])
                continue
            uncached.append(name)
    else:
        uncached = list(pending)

    if not uncached:
        return result

    for batch_start in range(0, len(uncached), max_batch_size):
        batch = uncached[batch_start:batch_start + max_batch_size]
        batch_results = _call_llm_batch(batch, object_type=object_type)
        result.update(batch_results)

    return result


def classify_action_tags_by_species(
    action_names_by_species: dict[str, list[str]],
    *,
    force_llm: bool = False,
    max_batch_size: int = _DEFAULT_BATCH_SIZE,
    max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
) -> dict[str, dict[str, list[str]]]:
    """Classify action names for several species, concurrently across species.

    Each species is processed by :func:`classify_action_tags_batch` (serial in
    chunks of ``max_batch_size`` internally); up to ``max_concurrency`` species
    are processed in parallel worker threads. The shared in-memory cache, the
    disk flush and the lazy client init are guarded by module-level locks, so
    parallel species are safe.

    Args:
        action_names_by_species: Mapping ``object_type -> [action_name, ...]``.
        force_llm: If True, bypass cache and force LLM calls.
        max_batch_size: Maximum number of names per LLM call.
        max_concurrency: Maximum number of species classified in parallel
            (default 4). Set to 1 to disable concurrency.

    Returns:
        Mapping ``object_type -> {action_name: [tag, ...]}``.
    """
    species_items = [
        (object_type, names)
        for object_type, names in action_names_by_species.items()
        if names
    ]
    if not species_items:
        return {}

    def _run(object_type: str, names: list[str]) -> dict[str, list[str]]:
        return classify_action_tags_batch(
            names,
            object_type=object_type,
            force_llm=force_llm,
            max_batch_size=max_batch_size,
        )

    if len(species_items) == 1 or max_concurrency <= 1:
        return {
            object_type: _run(object_type, names)
            for object_type, names in species_items
        }

    # Warm up the shared state once, before any worker starts, so the threads
    # share a single initialised client and a fully-loaded cache instead of
    # racing on first use.
    if not force_llm:
        _ensure_cache_loaded()
    _get_llm_client_and_model()
    workers = min(max_concurrency, len(species_items))
    results: dict[str, dict[str, list[str]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_species = {
            executor.submit(_run, object_type, names): object_type
            for object_type, names in species_items
        }
        for future in future_to_species:
            object_type = future_to_species[future]
            results[object_type] = future.result()
    return results


def classify_action_tags(
    action_name: str,
    *,
    object_type: str | None = None,
    force_llm: bool = False,
) -> list[str]:
    """Classify a single action name into action tags.

    Args:
        action_name: The action name to classify.
        object_type: Optional species/object type for context-aware classification.
        force_llm: If True, bypass cache and force LLM call.
    """
    results = classify_action_tags_batch(
        [action_name], object_type=object_type, force_llm=force_llm,
    )
    return results.get(action_name, ["unknown"])


def lookup_cached_tags(action_name: str, object_type: str | None = None) -> list[str] | None:
    """Return the resolved tags for *action_name* (override or cache), or None.

    Args:
        action_name: The action name to look up.
        object_type: Optional species/object type. When provided, looks up the
            species-specific cache entry.
    """
    override = _lookup_override(action_name)
    if override is not None:
        return override
    _ensure_cache_loaded()
    key = _cache_key(action_name, object_type)
    cached = _in_memory_cache.get(key)
    return list(cached) if cached is not None else None


# ---------------------------------------------------------------------------
# Internal: LLM call with caching
# ---------------------------------------------------------------------------

def _call_llm_batch(
    action_names: list[str],
    *,
    object_type: str | None = None,
) -> dict[str, list[str]]:
    """Call LLM to classify a batch of action names. Caches results per-name.

    Args:
        action_names: List of action names to classify.
        object_type: Optional species/object type. When provided, the prompt
            includes the species context and the cache is keyed by
            ``(action_name, object_type)``.
    """
    system_msg = _build_system_message()
    species_context = (
        f" (species: {object_type})" if object_type else ""
    )
    user_msg = (
        "Classify each of these action names into action tags"
        f"{species_context}:\n"
        + "\n".join(f"- {name}" for name in action_names)
        + '\n\nReturn JSON: {"action_name": ["tag1", "tag2"], ...}'
    )

    client, model = _get_llm_client_and_model()
    valid_tags = set(ACTION_TAGS)
    messages: list[dict] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    species_tag = f" species={object_type}" if object_type else ""
    print(f"[action_labels] Calling LLM for {len(action_names)} actions{species_tag}  model={model}")

    max_retries = 2
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"[action_labels] LLM retry {attempt}/{max_retries} after parse error")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=0,
            top_p=1.0,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = response.choices[0].message.content or ""

        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            stripped = stripped.rsplit("```", 1)[0]

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            last_exc = exc
            if attempt < max_retries:
                print(f"[action_labels] LLM parse error (attempt {attempt}): {exc}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response could not be parsed as JSON. Error: {exc}. "
                        "Please return ONLY valid JSON, nothing else."
                    ),
                })
                continue
            raise RuntimeError(
                f"LLM returned unparseable JSON after {max_retries + 1} attempts. "
                f"Last error: {exc}\nLast response:\n{raw}"
            ) from exc

        if not isinstance(parsed, dict):
            type_err = f"expected JSON object, got {type(parsed).__name__}"
            last_exc = ValueError(type_err)
            if attempt < max_retries:
                print(f"[action_labels] LLM wrong type (attempt {attempt}): {type_err}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response was valid JSON but had the wrong structure: {type_err}. "
                        'Return a JSON object {"action_name": ["tag1", "tag2"], ...}, nothing else.'
                    ),
                })
                continue
            raise RuntimeError(
                f"LLM returned wrong JSON type after {max_retries + 1} attempts: {type_err}\n"
                f"Last response:\n{raw}"
            ) from last_exc

        result: dict[str, list[str]] = {}
        invalid: list[str] = []
        for name in action_names:
            raw_tags = parsed.get(name)
            tags = _normalize_tag_list(raw_tags, valid_tags)
            if not tags:
                if raw_tags is not None:
                    invalid.append(f"'{name}' → {raw_tags!r}")
                result[name] = ["unknown"]
                continue

            if isinstance(raw_tags, (list, tuple)):
                invalid_items = [
                    str(value)
                    for value in raw_tags
                    if str(value).strip().lower() not in valid_tags
                ]
                if invalid_items:
                    invalid.append(f"'{name}' invalid tags {invalid_items!r}")
            elif raw_tags is not None and not isinstance(raw_tags, str):
                invalid.append(f"'{name}' → {raw_tags!r}")

            result[name] = tags

        if invalid:
            warnings.warn(
                f"[action_labels] LLM returned {len(invalid)} invalid tag payload(s), "
                f"normalizing/defaulting to ['unknown']: {', '.join(invalid[:5])}"
                + (" ..." if len(invalid) > 5 else ""),
                stacklevel=2,
            )

        with _cache_write_lock:
            for name, tags in result.items():
                key = _cache_key(name, object_type)
                _in_memory_cache[key] = list(tags)
            _flush_cache_to_disk()

        print(f"[action_labels] LLM result: {len(result)}/{len(action_names)} classified{species_tag}")
        return result

    raise RuntimeError(f"LLM action classification failed: {last_exc}")
