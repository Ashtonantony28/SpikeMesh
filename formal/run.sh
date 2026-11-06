#!/usr/bin/env bash
# SpikeMesh SG3 formal runs: Yosys -> SMT2 -> yosys-smtbmc (SymbiYosys is not
# available in the container; this is the same flow sby's smtbmc engine uses).
#   bmc  : bounded model check from reset, depth N
#   prove: k-induction (unbounded proof); the base case is the BMC run
# Usage: formal/run.sh [quick]      (needs yosys, yosys-smtbmc, boolector or z3)
set -u
cd "$(dirname "$0")/.."
OUT=formal/build
mkdir -p "$OUT"
SOLVER=${SOLVER:-boolector}
command -v "$SOLVER" >/dev/null || SOLVER=z3
QUICK=${1:-}
pass=0; fail=0
SUMMARY=()

smt2() {  # name top params... (param as NAME=VAL)
    local name=$1 top=$2; shift 2
    local files=$FILES chp=""
    for p in "$@"; do chp+="chparam -set ${p%%=*} ${p#*=} $top; "; done
    yosys -q -l "$OUT/$name.yosys.log" -p "read_verilog -sv -formal $files; $chp prep -top $top; \
        async2sync; dffunmap; write_smt2 -wires $OUT/$name.smt2" >/dev/null 2>&1 \
        || { echo "yosys failed for $name (see $OUT/$name.yosys.log)"; return 1; }
}

check() {  # name mode depth
    local name=$1 mode=$2 depth=$3 args
    if [ "$mode" = prove ]; then args="-i"; else args=""; fi
    local t0=$(date +%s)
    if timeout 1200 yosys-smtbmc -s "$SOLVER" --presat $args -t "$depth" "$OUT/$name.smt2" \
            > "$OUT/$name.$mode.log" 2>&1 && grep -q "Status: PASSED" "$OUT/$name.$mode.log"; then
        r=PASS; pass=$((pass+1))
    else
        r=FAIL; fail=$((fail+1))
    fi
    local msg; msg=$(printf "%-28s %-6s depth %-3s %s (%ss)" "$name" "$mode" "$depth" "$r" $(( $(date +%s) - t0 )))
    echo "$msg"; SUMMARY+=("$msg")
}

# ---------------- router ----------------
FILES="rtl/noc/noc_fifo.sv rtl/noc/router.sv"
RCFG=("0 0 3 3" "1 1 3 3" "2 2 3 3" "0 1 2 2" "1 0 2 2")
[ "$QUICK" = quick ] && RCFG=("1 1 3 3" "0 0 2 2")
for c in "${RCFG[@]}"; do
    set -- $c
    for d in 2 4; do
        n="router_x$1y$2_m$3x$4_d$d"
        smt2 "$n" router X=$1 Y=$2 MESH_X=$3 MESH_Y=$4 FIFO_DEPTH=$d || { fail=$((fail+1)); continue; }
        check "$n" bmc $(( d == 2 ? 20 : 14 ))
        check "$n" prove $(( d == 2 ? 4 : 6 ))
    done
done

# ---------------- tick controller ----------------
FILES="rtl/top/tick_ctrl.sv"
for q in 4 6; do
    n="tick_ctrl_q$q"
    smt2 "$n" tick_ctrl NC=4 QUIESCE_CYCLES=$q START_GUARD=2 || { fail=$((fail+1)); continue; }
    check "$n" bmc 30
    check "$n" prove $(( q + 4 ))
done

echo "----"
printf '%s\n' "${SUMMARY[@]}" > "$OUT/summary.txt"
echo "formal: $pass passed, $fail failed (logs in $OUT/)"
[ "$fail" -eq 0 ]
