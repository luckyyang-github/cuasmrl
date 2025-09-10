#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def ensure_triton_on_path():
    """Prefer in-repo Triton if installed Triton lacks required APIs.

    - If Triton isn't installed: prepend ./python to sys.path.
    - If Triton is installed but misses `compiler.make_launcher.make_stub`:
      force-prepend ./python and reload to use the vendored Triton.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    vendor_path = os.path.join(repo_root, "python")

    def _use_vendor():
        if os.path.isdir(vendor_path):
            # Purge any previously-imported Triton modules
            for k in list(sys.modules.keys()):
                if k == "triton" or k.startswith("triton."):
                    sys.modules.pop(k, None)
            # Prepend vendor path to import it first
            if vendor_path not in sys.path:
                sys.path.insert(0, vendor_path)
            # Import to validate availability
            import triton  # noqa: F401
            return True
        return False

    try:
        import triton  # noqa: F401
        # Check for required API in current Triton
        try:
            from triton.compiler import make_launcher as _ml  # noqa: F401
            getattr(_ml, "make_stub")
            return  # Suitable Triton already available
        except Exception:
            if _use_vendor():
                return
            # Fall through to raise a helpful error below
    except Exception:
        if _use_vendor():
            return
    # If we reach here, Triton is missing or incompatible and no vendor copy
    raise ImportError(
        "Triton not found or incompatible (missing make_stub), and no vendored copy at ./python."
    )


def load_sass(path: str) -> List[str]:
    with open(path, "r") as f:
        text = f.read()
    # Normalize to a list of lines (CuAsmParser expects iterable of lines)
    return text.splitlines()


def detect_kernel_name_from_sass(sass_lines: List[str]) -> Optional[str]:
    # Heuristic: pick the first line containing '.text.' and take suffix as symbol
    for line in sass_lines:
        if ".text." in line:
            # Examples: '.text._Z8mykernelPi', '.text.triton__something'
            idx = line.find(".text.")
            suffix = line[idx + len(".text."):].strip()
            # Avoid empty
            if suffix:
                # In some dumps, the label might be alone on a line; this is fine.
                return suffix
    return None


def parse_signature(sig: str) -> Dict[int, str]:
    # e.g. "*fp32,*fp32,i32" -> {0:"*fp32",1:"*fp32",2:"i32"}
    parts = [p.strip() for p in sig.split(",") if p.strip()]
    return {i: p for i, p in enumerate(parts)}


def torch_dtype_from_triton_ty(ty: str):
    import torch
    base = ty.lstrip("*")
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "f32": torch.float32,
        "fp64": torch.float64,
        "i8": torch.int8,
        "i16": torch.int16,
        "i32": torch.int32,
        "i64": torch.int64,
        # Unsigned -> pick a sane corresponding torch dtype
        "u32": torch.int64,
        "u64": torch.int64,
    }
    if base not in mapping:
        raise ValueError(f"Unsupported Triton type in signature: {ty}")
    return mapping[base]


def build_args_from_json(args_json_path: str, signature: Dict[int, str]) -> List[Any]:
    import torch
    with open(args_json_path, "r") as f:
        data = json.load(f)
    # Expect a list aligned to signature indices
    if not isinstance(data, list):
        raise ValueError("args-json must be a list of argument specs")
    if len(data) != len(signature):
        raise ValueError(
            f"args count ({len(data)}) does not match signature arity ({len(signature)})"
        )

    args: List[Any] = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for i, spec in enumerate(data):
        if not isinstance(spec, dict):
            raise ValueError("Each arg spec must be an object")
        ty = signature[i]
        is_ptr = ty.startswith("*")
        if is_ptr:
            # Expect shape; init can be rand/zeros
            shape = spec.get("shape")
            if not shape:
                raise ValueError(f"Pointer arg {i} requires 'shape' in args-json")
            dtype = torch_dtype_from_triton_ty(ty)
            init = spec.get("init", "rand")
            if init == "zeros":
                arg = torch.zeros(shape, dtype=dtype, device=device)
            elif init == "ones":
                arg = torch.ones(shape, dtype=dtype, device=device)
            else:
                arg = torch.randn(shape, dtype=dtype, device=device)
            args.append(arg)
        else:
            # Scalar. Respect provided 'value' or default 0
            val = spec.get("value", 0)
            args.append(val)
    return args


@dataclass
class DRLConfig:
    # Environment
    env_id: str = "cuasmenv-v0"
    n_tests: int = 32
    verbose: int = 1
    horizon: Optional[int] = None
    normalize_reward: int = 0
    seed: int = 0

    # PPO hyperparameters
    num_env: int = 1
    num_steps: int = 1500  # >1000 to trigger saving in first iteration
    minibatch_size: int = 64
    update_epochs: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_coef: float = 0.2
    clip_vloss: int = 1
    norm_adv: int = 1
    anneal_lr: int = 1
    target_kl: Optional[float] = None
    num_iterations: int = 1  # keep short by default
    gpu: int = 1
    agent: str = "ppo"

    # IO / logging
    default_out_path: str = "runs"
    save_dir: str = "exp"
    log: int = 1

    # Driver flags
    train: int = 1
    total_flops: Optional[float] = None


def assemble_sass_to_cubin(sass_lines: List[str]) -> bytes:
    try:
        from CuAsm.CuAsmParser import CuAsmParser
    except Exception as e:
        raise RuntimeError(
            "CuAssembler (CuAsm) not found. Please clone CuAssembler and set PYTHONPATH as in README."
        ) from e
    cap = CuAsmParser()
    cap.parse_from_buffer(sass_lines)
    cubin = cap.dump_cubin()
    return cubin


def disassemble_cubin_to_sass(cubin_bytes: bytes) -> str:
    # Use CuAsm's CubinFile to dump SASS as text
    from CuAsm.CubinFile import CubinFile
    with tempfile.NamedTemporaryFile(mode="wb", delete=True) as tf:
        tf.write(cubin_bytes)
        tf.flush()
        tf.seek(0)
        time.sleep(0.2)
        cf = CubinFile(tf.name)
        text1, _ = cf.dump_sass()  # text1: full SASS buffer
        return text1.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Run RL scheduling search starting from your SASS/cubin and output optimized SASS and speedup."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sass", type=str, help="Path to input SASS (.cuasm) file")
    src.add_argument("--cubin", type=str, help="Path to input cubin file")
    parser.add_argument("--kernel-name",
                        type=str,
                        default=None,
                        help="Kernel symbol name (auto-detect from SASS if omitted)")
    parser.add_argument("--signature",
                        type=str,
                        required=True,
                        help="Comma-separated Triton types, e.g. '*fp32,*fp32,i32'")
    parser.add_argument("--args-json",
                        type=str,
                        required=True,
                        help="JSON file describing runtime args in order")
    parser.add_argument("--grid",
                        type=str,
                        default="1,1,1",
                        help="Launch grid as gx,gy,gz")
    parser.add_argument("--num-warps",
                        type=int,
                        default=4,
                        help="Warps per block (blockDimX = 32*num_warps)")
    parser.add_argument("--shared",
                        type=int,
                        default=0,
                        help="Dynamic shared memory bytes")
    parser.add_argument("--ret-ptr",
                        type=int,
                        required=True,
                        help="Index of output pointer argument for correctness check")
    parser.add_argument("--out-dir",
                        type=str,
                        default="runs",
                        help="Base output dir for logs/checkpoints")
    parser.add_argument("--save-dir",
                        type=str,
                        default="exp",
                        help="Subdir for this run")
    parser.add_argument("--n-tests",
                        type=int,
                        default=32,
                        help="Number of random test samples per verification")
    parser.add_argument("--num-steps",
                        type=int,
                        default=1500,
                        help="PPO steps per iteration (set >1000 to trigger saving)")
    parser.add_argument("--num-iter",
                        type=int,
                        default=1,
                        help="Number of PPO iterations (epochs)")
    parser.add_argument("--total-flops",
                        type=float,
                        default=None,
                        help="If set, report performance in TFLOPS instead of -ms")
    parser.add_argument("--optimized-out",
                        type=str,
                        default=None,
                        help="Path to write optimized SASS (.cuasm); default within save-dir")

    args = parser.parse_args()

    ensure_triton_on_path()

    # Late imports after path setup
    import torch  # noqa: F401
    from triton.compiler.make_launcher import make_stub
    from cuasmrl.compiler import CompiledKernel as FGKCompiledKernel
    from cuasmrl.drl import run_drl

    # Prepare initial cubin
    if args.sass:
        sass_lines = load_sass(args.sass)
        kernel_name = args.kernel_name or detect_kernel_name_from_sass(sass_lines)
        if not kernel_name:
            raise RuntimeError(
                "Cannot auto-detect kernel name from SASS. Please pass --kernel-name."
            )
        cubin = assemble_sass_to_cubin(sass_lines)
    else:
        # Provided a cubin directly
        if not os.path.isfile(args.cubin):
            raise FileNotFoundError(args.cubin)
        with open(args.cubin, "rb") as f:
            cubin = f.read()
        kernel_name = args.kernel_name
        if not kernel_name:
            raise RuntimeError(
                "--kernel-name is required when starting from --cubin"
            )

    # Build launcher stub
    signature = parse_signature(args.signature)
    constants: Dict[int, Any] = {}
    ids = {
        "ids_of_tensormaps": (),
        "ids_of_folded_args": (),
        "ids_of_const_exprs": (),
    }
    so_path = make_stub(kernel_name, signature, constants, ids)

    # Metadata and asm
    metadata = {
        "name": kernel_name,
        "shared": int(args.shared),
        "num_warps": int(args.num_warps),
        "num_ctas": 1,
        "num_stages": 1,
        "clusterDims": [1, 1, 1],
        "constants": {},
        "device_type": "cuda",
    }
    asm = {"cubin": cubin}

    # Instantiate runtime args
    runtime_args = build_args_from_json(args.args_json, signature)

    # Grid
    try:
        gx, gy, gz = [int(x) for x in args.grid.split(",")]
    except Exception:
        raise ValueError("--grid must be formatted as 'gx,gy,gz'")

    # Compose DRL config
    drl_cfg = DRLConfig(
        n_tests=args.n_tests,
        default_out_path=args.out_dir,
        save_dir=args.save_dir,
        num_steps=args.num_steps,
        num_iterations=args.num_iter,
        total_flops=args.total_flops,
    )

    # Build kernel wrapper and run RL
    bin = FGKCompiledKernel(so_path, metadata, asm)
    ret_ptr = int(args.ret_ptr)
    sig_key: Tuple[Any, ...] = tuple()  # not used by verification path
    non_constexpr_arg_values = list(runtime_args)

    print("[INFO] Starting RL scheduling search...")
    run_drl(
        bin,
        so_path,
        metadata,
        asm,
        ret_ptr,
        runtime_args,
        sig_key,
        non_constexpr_arg_values,
        gx,
        gy,
        gz,
        None,  # stream (None -> default)
        None,
        None,
        drl_cfg,
    )

    # After training completes, scan save_dir for results
    save_path = os.path.join(args.out_dir, args.save_dir)
    best_pkl = None
    best_impr = None
    best_final = None
    best_init = None
    best_cubin = None
    if os.path.isdir(save_path):
        for fn in os.listdir(save_path):
            if not fn.endswith(".pkl"):
                continue
            full = os.path.join(save_path, fn)
            try:
                with open(full, "rb") as f:
                    obj = json.loads("null")  # placeholder to keep structure
                # Use pickle only when file exists; defer import lazily
                import pickle  # noqa: WPS433
                with open(full, "rb") as f:
                    data = pickle.load(f)
                final_perf = data.get("final_perf")
                init_perf = data.get("init_perf")
                cubin_bytes = data.get("cubin")
                if final_perf is None or init_perf is None or cubin_bytes is None:
                    continue
                # Improvement ratio (same as selection.py)
                try:
                    improvement = (final_perf - init_perf) / init_perf
                except ZeroDivisionError:
                    improvement = None
                if best_impr is None or (improvement is not None and improvement > best_impr):
                    best_impr = improvement
                    best_final = final_perf
                    best_init = init_perf
                    best_pkl = full
                    best_cubin = cubin_bytes
            except Exception:
                continue

    if best_cubin is None:
        print("[WARN] No optimized kernels were saved. Try increasing --num-steps/--num-iter.")
        sys.exit(0)

    # Disassemble optimized cubin to SASS and write out
    optimized_sass = disassemble_cubin_to_sass(best_cubin)
    out_path = args.optimized_out or os.path.join(save_path, "optimized.cuasm")
    with open(out_path, "w") as f:
        f.write(optimized_sass)

    print("[RESULT]")
    print(f"optimized_sass_path: {out_path}")
    if best_impr is not None:
        pct = best_impr * 100.0
        print(f"performance_improvement: {pct:.2f}% (final={best_final:.4f}, init={best_init:.4f})")
    else:
        print(f"final_perf: {best_final}, init_perf: {best_init} (improvement N/A)")


if __name__ == "__main__":
    main()
