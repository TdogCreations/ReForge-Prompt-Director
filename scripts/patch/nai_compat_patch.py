# nai_compat_patch.py
# ReForge / A1111 NAI compat: ensure NAI reads the *post-injection* prompts (p.all_prompts)
# Works by wrapping NAI entrypoints that receive (p, prompts, ...)

import importlib
import traceback

_PATCHED = False


def _resolve_base_index(p, prompts):
    """Find where this batch's prompts live inside p.all_prompts (handles ReForge recreating p)."""
    try:
        allp = getattr(p, "all_prompts", None)
        if isinstance(allp, list) and isinstance(prompts, (list, tuple)) and len(prompts) > 0:
            bsz = len(prompts)
            limit = len(allp) - bsz
            for i in range(0, max(-1, limit) + 1):
                if allp[i:i + bsz] == list(prompts):
                    return i
    except Exception:
        pass

    # fallback
    try:
        bi = getattr(p, "batch_index", None)
        if isinstance(bi, int) and isinstance(prompts, (list, tuple)) and len(prompts) > 0:
            return bi * len(prompts)
    except Exception:
        pass

    return 0


def _maybe_sync_prompts_from_all_prompts(p, prompts):
    """If p.all_prompts exists, overwrite prompts[] with the authoritative values from p.all_prompts."""
    if not isinstance(prompts, list):
        return False

    allp = getattr(p, "all_prompts", None)
    if not isinstance(allp, list) or not allp:
        return False

    bsz = len(prompts)
    if bsz <= 0:
        return False

    base = _resolve_base_index(p, prompts)
    changed = False

    for i in range(bsz):
        gi = base + i
        if 0 <= gi < len(allp):
            ap = allp[gi]
            if isinstance(ap, str) and ap and ap != prompts[i]:
                prompts[i] = ap
                changed = True

    if changed:
        try:
            print(f"✅ [NAI PATCH] Synced prompts from p.all_prompts (base_index={base}, batch={bsz})")
        except Exception:
            pass

    return changed


def _wrap_entrypoint(mod, fn_name):
    """Wrap a module-level function like nai_api_processing.process_batch(p, prompts, ...)"""
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        return False

    if getattr(fn, "_nai_patch_wrapped", False):
        return True

    def wrapper(*args, **kwargs):
        # Try to locate p and prompts from args/kwargs
        p = kwargs.get("p", None)
        prompts = kwargs.get("prompts", None)

        # Heuristic: p is often first arg
        if p is None and len(args) >= 1:
            p = args[0]

        # Find list[str] in args if prompts not provided
        if prompts is None:
            for a in args:
                if isinstance(a, list) and (not a or isinstance(a[0], str)):
                    prompts = a
                    break

        try:
            if p is not None and prompts is not None:
                _maybe_sync_prompts_from_all_prompts(p, prompts)
        except Exception:
            print("⚠️ [NAI PATCH] Sync attempt failed (module fn):")
            traceback.print_exc()

        return fn(*args, **kwargs)

    wrapper._nai_patch_wrapped = True
    setattr(mod, fn_name, wrapper)
    return True


def _wrap_class_method(mod, cls_name, method_name):
    """Wrap a class method like NAIGENScriptBase.begin_request(self, p, prompts, ...)"""
    cls = getattr(mod, cls_name, None)
    if cls is None:
        return False

    original = getattr(cls, method_name, None)
    if not callable(original):
        return False

    if getattr(original, "_nai_patch_wrapped", False):
        return True

    def wrapper(self, *args, **kwargs):
        # Try to locate p and prompts from args/kwargs
        p = kwargs.get("p", None)
        prompts = kwargs.get("prompts", None)

        # Heuristic: first positional arg often p
        if p is None and len(args) >= 1:
            p = args[0]

        # Find list[str] in args if prompts not provided
        if prompts is None:
            for a in args:
                if isinstance(a, list) and (not a or isinstance(a[0], str)):
                    prompts = a
                    break

        try:
            if p is not None and prompts is not None:
                _maybe_sync_prompts_from_all_prompts(p, prompts)
        except Exception:
            print("⚠️ [NAI PATCH] Sync attempt failed (class method):")
            traceback.print_exc()

        return original(self, *args, **kwargs)

    wrapper._nai_patch_wrapped = True
    setattr(cls, method_name, wrapper)
    return True


def apply_nai_patch():
    """
    Patch NAI entrypoints so NAI uses prompts after JoyCaption/WD14 injection.
    Idempotent: safe to call repeatedly. The done-flag lives on modules.shared
    because each caller re-execs this file (spec_from_file_location), which resets
    the module-level _PATCHED to False every time. modules.shared is a true
    singleton for the webui process, so it actually survives.
    """
    global _PATCHED

    try:
        from modules import shared as _shared
    except Exception:
        _shared = None

    if _PATCHED or (_shared is not None and getattr(_shared, "_nai_compat_patch_done", False)):
        return True

    wrapped_any = False

    # 1) Try module-level processing functions (some forks)
    try:
        mod = importlib.import_module("nai_api_gen.nai_api_processing")
        candidates = [
            "process",
            "process_batch",
            "process_images",
            "run",
            "run_batch",
            "txt2img",
            "img2img",
            "nai_process",
            "nai_process_batch",
            "generate",
            "generate_batch",
            "create_payload",
            "build_payload",
            "make_payload",
            "request",
        ]

        for name in candidates:
            try:
                if _wrap_entrypoint(mod, name):
                    print(f"✅ [NAI PATCH] Wrapped entrypoint: nai_api_processing.{name}")
                    wrapped_any = True
            except Exception:
                pass

        if not wrapped_any:
            print("⚠️ [NAI PATCH] Loaded nai_api_processing, but found no known entrypoints to wrap.")
    except Exception as e:
        print(f"⚠️ [NAI PATCH] Could not import nai_api_gen.nai_api_processing: {e}")

    # 2) Wrap the script class method your install uses (your log shows NAIGENScriptBase.begin_request)
    try:
        mod_script = importlib.import_module("nai_api_gen.nai_api_script")
        if _wrap_class_method(mod_script, "NAIGENScriptBase", "begin_request"):
            print("✅ [NAI PATCH] Wrapped class method: NAIGENScriptBase.begin_request")
            wrapped_any = True
    except Exception as e:
        print(f"⚠️ [NAI PATCH] Could not import nai_api_gen.nai_api_script: {e}")

    if wrapped_any:
        _PATCHED = True
        if _shared is not None:
            _shared._nai_compat_patch_done = True
        print("✅ [NAI PATCH] NAI compat patch applied.")
        return True

    print("⚠️ [NAI PATCH] Loaded NAI modules, but found no entrypoints to wrap.")
    return False
