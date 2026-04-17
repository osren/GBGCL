# gen_joblist.py
import itertools, argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Computers,Photo,Co.CS,Co.Physics")
    ap.add_argument("--quity", default="homo,detach")
    ap.add_argument("--sim", default="dot,cos")
    ap.add_argument("--alpha", default="0.3,0.7")
    ap.add_argument("--stage", default="A", help="A(粗筛)/B(精训)")
    ap.add_argument("--out", default="joblist.txt")
    args = ap.parse_args()

    ds = [s.strip() for s in args.datasets.split(",")]
    qs = [s.strip() for s in args.quity.split(",")]
    ss = [s.strip() for s in args.sim.split(",")]
    al = [s.strip() for s in args.alpha.split(",")]

    with open(args.out, "w") as f:
        for d, q, s, a in itertools.product(ds, qs, ss, al):
            f.write(f"{d} {q} {s} {a} {args.stage}\n")
    print(f"[SAVE] {args.out}")

if __name__ == "__main__":
    main()
