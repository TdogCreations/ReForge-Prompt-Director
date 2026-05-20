# nai_char_prompt_builder.py
import re

# Match: "CHAR:", "char:", "CHAR :", "ChAr   :" etc.
_CHAR_RE = re.compile(r"(?i)\bchar\s*:")

def _clean_commas(text: str) -> str:
    if not text:
        return ""
    t = str(text).strip()

    # turn newlines into commas so prompts don't get weird formatting
    t = re.sub(r"[\r\n]+", ", ", t)

    # normalize commas/spaces
    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r"(,\s*){2,}", ", ", t)
    t = re.sub(r"\s{2,}", " ", t)

    return t.strip(" ,")

def _normalize_char_tokens(text: str, token: str = "CHAR:") -> str:
    """Convert any 'char :' variants to exactly 'CHAR:' (or the provided token)."""
    if not text:
        return ""
    return _CHAR_RE.sub(token, str(text))

def inject_global_before_chars(base_prompt: str, inject_text: str, mode: str = "Append", token: str = "CHAR:") -> str:
    base_prompt = _normalize_char_tokens(base_prompt or "", token=token)
    inject_text = _clean_commas(inject_text or "")
    if not inject_text:
        return base_prompt

    m = re.search(re.escape(token), base_prompt)
    if not m:
        # No CHAR blocks exist -> normal inject behavior
        if mode == "Replace":
            return _clean_commas(inject_text)
        if mode == "Prepend":
            return _clean_commas(f"{inject_text}, {base_prompt}")
        return _clean_commas(f"{base_prompt}, {inject_text}")

    idx = m.start()
    pre = base_prompt[:idx].strip()
    chars = base_prompt[idx:].lstrip()  # starts with "CHAR:"

    if mode == "Replace":
        new_pre = inject_text
    elif mode == "Prepend":
        new_pre = f"{inject_text}, {pre}" if pre else inject_text
    else:  # Append
        new_pre = f"{pre}, {inject_text}" if pre else inject_text

    new_pre = _clean_commas(new_pre)
    if new_pre:
        # Keep CHAR blocks visually separate
        return f"{new_pre}\n{chars}".strip()
    return chars

def merge_chars_into_prompt(prompt: str, char_lists, token: str = "CHAR:", max_chars: int = 6, replace_indices=None, **_ignored_kwargs) -> str:
    prompt = _normalize_char_tokens(prompt or "", token=token)
    if not char_lists:
        return prompt

    # Split into global + blocks
    parts = prompt.split(token)
    global_part = parts[0].rstrip()
    blocks = [p.strip() for p in parts[1:]]

    def _traits(lst):
        return [str(t).strip() for t in (lst or []) if str(t).strip()]

    # Decide whether we're doing targeted replacement
    use_replace = False
    if replace_indices is None:
        use_replace = False
    elif isinstance(replace_indices, int):
        use_replace = True
        replace_indices = [replace_indices]
    else:
        # set/list/tuple/etc
        try:
            replace_indices = list(replace_indices)
            use_replace = len(replace_indices) > 0
        except Exception:
            use_replace = False
            replace_indices = None

    if use_replace:
        # Sort indices so mapping is stable (sets are unordered)
        idxs = []
        for i in replace_indices:
            try:
                idxs.append(int(i))
            except Exception:
                pass
        idxs = sorted(set([i for i in idxs if 0 <= i < max_chars]))

        # Ensure enough blocks
        for idx in idxs:
            while len(blocks) <= idx:
                blocks.append("")

        # Replace sequentially: char_lists[0] -> idxs[0], char_lists[1] -> idxs[1], ...
        for j, idx in enumerate(idxs):
            if j >= len(char_lists):
                break
            traits = _traits(char_lists[j])
            if traits:
                blocks[idx] = ", ".join(dict.fromkeys(traits))

    else:
        # Normal merge/append behavior into slot i
        for i in range(min(len(char_lists), max_chars)):
            traits = _traits(char_lists[i])
            if not traits:
                continue

            while len(blocks) <= i:
                blocks.append("")

            old_tags = [t.strip() for t in re.split(r"[,\n]+", blocks[i]) if t.strip()]
            merged = list(dict.fromkeys(old_tags + traits))
            blocks[i] = ", ".join(merged)

        # If we have more lists than existing blocks, append new blocks (up to max_chars)
        for i in range(len(blocks), min(len(char_lists), max_chars)):
            traits = _traits(char_lists[i])
            if traits:
                blocks.append(", ".join(dict.fromkeys(traits)))

    # Rebuild prompt (each CHAR block on a new line so you can SEE them)
    out = global_part.strip()
    for b in blocks[:max_chars]:
        b = (b or "").strip()
        if b:
            out = (out + "\n" if out else "") + f"{token} {b}"

    return out.strip()
