import sys, os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
src = open('AlienX-mhd.py').read().replace('if __name__ == "__main__":\n    main()', 'if False:\n    main()')
exec(compile(src, 'AlienX-mhd.py', 'exec'), globals())

which = sys.argv[1] if len(sys.argv) > 1 else "5"
fns = {"5": run_gate5, "6": run_gate6}
passed, data = fns[which]()
print(f"GATE {which}: {'PASS' if passed else 'FAIL'}", data)
raise SystemExit(0 if passed else 1)