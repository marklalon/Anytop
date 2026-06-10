"""
LLM-based action-tag classifier for motion labels.

Batch-classifies multiple action names in a single LLM call, with
dual-layer caching (in-memory + single disk file).

Usage::

    from data_loaders.truebones.truebones_utils.motion_labels_llm import (
        classify_action_tags_batch,
        ACTION_TAGS,
    )
    results = classify_action_tags_batch(["WalkLoop", "ChargeAttack", "Idle"])
    # → {"WalkLoop": ["locomotion"], "ChargeAttack": ["locomotion", "attack"], "Idle": ["idle"]}
"""
from __future__ import annotations

import json
import os
import re
import warnings

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
    "reaction",
    "rest",
    "eat",
    "social",
    "self",
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
    "getup": "Recovering to a standing posture from a downed, fallen, prone, or wounded state: "
             "get-up, rise, stand-up, recover-to-feet, struggle-up. The transition FROM "
             "lying/fallen/downed BACK TO standing. Its own category, distinct from locomotion.",
    "swim": "Aquatic locomotion: swim, float, slither(in water), wade. Movement through or on water.",
    "fly": "Aerial locomotion: fly, glide, soar, hover, flap-in-air, takeoff. Body airborne, wings "
           "active. Wing movement while standing on the ground is NOT fly (use 'social' or 'self').",
    "jump": "Discrete aerial maneuver: jump, hop, leap, pounce, spring, bound, vault, land, dive. "
            "A single explosive launch with an aerial phase, or the landing from one.",
    "turn": "In-place rotation or reorientation: turn, spin, pivot, rotate, circle, strafe. Body "
            "reorients without significant forward displacement. In-place rotation is NOT locomotion.",
    "attack": "Offensive combat: bite, strike, slash, kick, punch, gore, headbutt, claw, whip, shoot, "
              "charge, chase, hit, smash, swipe, snap, rip, peck, sting, tail-whip, kill(offensive). "
              "Physical aggression directed at a target. Also threat/intimidation displays with "
              "aggressive intent (bristling, baring fangs, angry/enraged posturing) — these usually "
              "also get 'social'.",
    "reaction": "Physical response to stimulus: flinch, recoil, stagger, hurt, impact, knocked, stun, "
                "limp, twitch, flip, buck, rear, defend, cower, stabilize, hit(receiving), "
                "shot(receiving). Involuntary, reflexive, or defensive reaction, not self-initiated "
                "locomotion. Nervous/anxious fidgety motion belongs here, not idle.",
    "rest": "Low-posture stationary: sit, lie, sleep, crouch, kneel, lay, settle, hide. "
            "Body is close to the ground, not standing upright.",
    "eat": "Feeding behavior: eat, feast, graze, drink, chew, swallow, cud, catch-fish, bite-at-food. "
           "Head lowered to a food source, ingestion motions. A bite that targets food is eat, "
           "not attack.",
    "social": "Communication and display: dance, play, taunt, celebrate, wave, beg, gesture, bark, "
              "howl, roar, growl, bleat, hiss, alert, warn, look, sniff, smell, bellow, coo, squawk, "
              "scream, rutting, ground-paw display. Broadcasting intent or emotion to others. "
              "Ground-based (non-airborne) wing/flap displays belong here. Aggressive displays also "
              "get 'attack'.",
    "self": "Self-maintenance: clean, preen, scratch(self), shake, shiver, yawn, itch, puke, stretch, "
            "tired, sick, wake-up, dig(self-grooming). Actions directed at own body, not at others.",
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
    "Squirly": ["reaction"],     # non-standard word for nervous fidgeting
    "Trottle": ["locomotion"],   # non-standard gait spelling
    "ChargedUp": ["idle"],       # readiness state, easily mis-read as attack
    "GroundFlap": ["social"],    # ground wing display, easily mis-read as fly
    "TailWhip": ["attack"],      # tail strike, easily mis-read as social
    "Specail": ["social"],       # Camel, non-standard word for walking display
    "Wild1": ["social"],         # Camel, semi-aggressive vocal display
    "Fancy": ["social"],         # Pteranodon, dancing/courtship display
    "Flapergasted": ["reaction"],# non-standard word for being startled/flustered
    "EggTend": ["social"],       # Raptor2, egg-tending behavior
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

_CACHE_VERSION = "v8"

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

# Batch size for LLM calls
_DEFAULT_BATCH_SIZE: int = 50

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
    """Load the single-file cache into memory on first access."""
    global _cache_loaded_from_disk
    if _cache_loaded_from_disk:
        return
    _cache_loaded_from_disk = True

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
        "['locomotion', 'attack']; a run-and-roar name is ['locomotion', 'social'].\n"
        "3. Aggressive / threat displays broadcast both hostility and intent — give them BOTH "
        "'social' and 'attack'.\n"
        "4. Combat vs feeding: a bite at a target is 'attack'; a bite at food is 'eat'.\n"
        "5. Agent vs patient: a character performing a kill is 'attack'; a character dying is 'death'.\n"
        "6. Displacement vs in-place: in-place rotation is 'turn' not 'locomotion'; getting up from a "
        "downed state is 'getup' not 'locomotion'; only airborne movement is 'fly'.\n"
        "7. If the name is unrecognizable, garbled, or too ambiguous to map confidently, use "
        "['unknown'] — do NOT fall back to 'idle' (idle is only for genuine stationary upright poses).\n"
        "8. Return arrays of unique lowercase tags from the valid list only.\n\n"
        f"Valid tags: [{valid_list}]\n\n"
        "Return ONLY valid JSON — no explanation, no markdown fences."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_action_tags_batch(
    action_names: list[str],
    *,
    force_llm: bool = False,
    max_batch_size: int = _DEFAULT_BATCH_SIZE,
) -> dict[str, list[str]]:
    """Classify multiple action names into action tags using LLM with per-name caching."""
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
            if name in _in_memory_cache:
                result[name] = list(_in_memory_cache[name])
                continue
            uncached.append(name)
    else:
        uncached = list(pending)

    if not uncached:
        return result

    for batch_start in range(0, len(uncached), max_batch_size):
        batch = uncached[batch_start:batch_start + max_batch_size]
        batch_results = _call_llm_batch(batch)
        result.update(batch_results)

    return result


def classify_action_tags(
    action_name: str,
    *,
    force_llm: bool = False,
) -> list[str]:
    """Classify a single action name into action tags."""
    results = classify_action_tags_batch([action_name], force_llm=force_llm)
    return results.get(action_name, ["unknown"])


def lookup_cached_tags(action_name: str) -> list[str] | None:
    """Return the resolved tags for *action_name* (override or cache), or None."""
    override = _lookup_override(action_name)
    if override is not None:
        return override
    _ensure_cache_loaded()
    cached = _in_memory_cache.get(action_name)
    return list(cached) if cached is not None else None


# ---------------------------------------------------------------------------
# Internal: LLM call with caching
# ---------------------------------------------------------------------------

def _call_llm_batch(action_names: list[str]) -> dict[str, list[str]]:
    """Call LLM to classify a batch of action names. Caches results per-name."""
    system_msg = _build_system_message()
    user_msg = (
        "Classify each of these action names into action tags:\n"
        + "\n".join(f"- {name}" for name in action_names)
        + '\n\nReturn JSON: {"action_name": ["tag1", "tag2"], ...}'
    )

    client, model = _get_llm_client_and_model()
    valid_tags = set(ACTION_TAGS)
    messages: list[dict] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    print(f"[action_labels] Calling LLM for {len(action_names)} actions  model={model}")

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

        for name, tags in result.items():
            _in_memory_cache[name] = list(tags)
        _flush_cache_to_disk()

        print(f"[action_labels] LLM result: {len(result)}/{len(action_names)} classified")
        return result

    raise RuntimeError(f"LLM action classification failed: {last_exc}")
