#!/usr/bin/env bash
# SpikeMesh SG3 formal checks with Yosys's built-in SAT engine (`sat`).
# SymbiYosys is not installable here and yosys-smtbmc with z3 4.8 / boolector
# 1.5 is too slow / too old for this design, so we use:
#   prove : sat -tempinduct            k-induction => UNBOUNDED proof
#   bmc   : sat -tempinduct-baseonly   bounded check from reset, depth N
#   mutant: a deliberately broken copy that MUST fail (non-vacuity check)
# Reset is forced in step 1 (assertions are checked from step 2 on).
# Usage: formal/run.sh            (about 10-15 minutes)
#        formal/run.sh quick      (one router config + tick_ctrl)
#        SKIP_ROUTER_PROVE=1 formal/run.sh   (skip the six router proofs)
# Failing runs: formal/build/<name>.log; formal/sat_cex.py <log> lists the
# failing assertion lines when rerun with -show-all.
set -u
cd "$(dirname "$0")/.."
OUT=formal/build
mkdir -p "$OUT"
QUICK=${1:-}
pass=0; fail=0
[ "${SKIP_ROUTER_PROVE:-0}" = 1 ] || : > "$OUT/summary.txt"

# run <name> <expect PASS|FAIL> <mode prove|bmc> <steps> <top> <"files"> <"defines"> <params...>
run() {
    local name=$1 expect=$2 mode=$3 steps=$4 top=$5 files=$6 defs=$7; shift 7
    local chp="" t0 res opts
    for p in "$@"; do chp+=" -set ${p%%=*} ${p#*=}"; done
    if [ "$mode" = prove ]; then opts="-tempinduct"; else opts="-tempinduct -tempinduct-baseonly"; fi
    t0=$(date +%s)
    timeout 900 yosys -l "$OUT/$name.log" -q -p "read_verilog -sv -formal $defs $files; \
        ${chp:+chparam$chp $top;} prep -top $top; memory_map; opt -fast; flatten; opt_clean; \
        sat $opts -prove-asserts -set-assumes -set-init-zero -set-at 1 rst_n 0 -seq 1 -maxsteps $steps -verify" \
        >/dev/null 2>&1
    if [ $? -eq 0 ]; then res=PASS; else res=FAIL; fi
    local ok=no
    [ "$res" = "$expect" ] && ok=yes
    if [ $ok = yes ]; then pass=$((pass+1)); else fail=$((fail+1)); fi
    local what="$mode"
    [ "$mode" = prove ] && what="prove (k<=$steps, unbounded)"
    [ "$mode" = bmc ] && what="bmc depth $steps"
    local msg; msg=$(printf "%-34s %-26s %s (expected %s)%s  %ss" "$name" "$what" "$res" "$expect" \
        "$([ $ok = yes ] || echo '  <<< UNEXPECTED')" $(( $(date +%s) - t0 )))
    echo "$msg" | tee -a "$OUT/summary.txt"
}

RF="rtl/noc/noc_fifo.sv rtl/noc/router.sv"
TF="rtl/top/tick_ctrl.sv"

# ---------------- router: all properties, FIFO_DEPTH 2 ----------------
RCFG=("0 0 3 3" "1 1 3 3" "2 2 3 3" "1 0 3 3" "0 1 2 2" "0 0 1 1")
[ "$QUICK" = quick ] && RCFG=("1 1 3 3")
[ "${SKIP_ROUTER_PROVE:-0}" = 1 ] && RCFG=()
for c in "${RCFG[@]}"; do
    set -- $c
    run "router_x$1y$2_m$3x$4_d2" PASS prove 3 router "$RF" "" X=$1 Y=$2 MESH_X=$3 MESH_Y=$4 FIFO_DEPTH=2
done
# FIFO_DEPTH 4 is not proven here: its 1-step induction did not finish in
# 900 s with the `sat` engine. The properties are depth-generic and
# FIFO_DEPTH 4 is covered by simulation (tests/noc).
if [ "$QUICK" != quick ]; then
    run "router_x1y1_m3x3_d2_bmc" PASS bmc 4 router "$RF" "" X=1 Y=1 MESH_X=3 MESH_Y=3 FIFO_DEPTH=2
fi

# ---------------- tick controller ----------------
for q in 4 6; do
    run "tick_ctrl_q$q" PASS prove $((q + 4)) tick_ctrl "$TF" "" NC=4 QUIESCE_CYCLES=$q START_GUARD=2
    run "tick_ctrl_q${q}_bmc" PASS bmc 30 tick_ctrl "$TF" "" NC=4 QUIESCE_CYCLES=$q START_GUARD=2
done

# ---------------- mutants (must FAIL) ----------------
M=$OUT/mut; mkdir -p "$M"
# M1: the round-robin index bug found in simulation (3-bit rr+k wraps at 8)
sed -e "s/sum = {1'b0, rr\[o\]} + 4'(k);/sum = 4'(3'(rr[o] + 3'(k)));/" rtl/noc/router.sv > "$M/router_m1.sv"
run "mutant_rr_wrap" FAIL prove 3 router "rtl/noc/noc_fifo.sv $M/router_m1.sv" "" X=1 Y=1 MESH_X=3 MESH_Y=3 FIFO_DEPTH=2
# M2: credits returned one per two pops (lost credit)
sed -e "s/credit_out <= pop\[3:0\];/credit_out <= pop[3:0] \& {4{drop_cnt[0]}};/" rtl/noc/router.sv > "$M/router_m2.sv"
run "mutant_lost_credit" FAIL prove 3 router "rtl/noc/noc_fifo.sv $M/router_m2.sv" "" X=1 Y=1 MESH_X=3 MESH_Y=3 FIFO_DEPTH=2
# M3: YX instead of XY order
sed -e "s/end else if ({28'd0, dx} > X) route = 3'(PE);/end else if ({28'd0, dy} > Y) route = 3'(PN); else if ({28'd0, dx} > X) route = 3'(PE);/" \
    rtl/noc/router.sv > "$M/router_m3.sv"
run "mutant_yx_routing" FAIL prove 3 router "rtl/noc/noc_fifo.sv $M/router_m3.sv" "" X=1 Y=1 MESH_X=3 MESH_Y=3 FIFO_DEPTH=2
# M4: quiescence ignores network traffic
sed -e "s/busy_r <= net_busy | core_any;/busy_r <= core_any;/" rtl/top/tick_ctrl.sv > "$M/tick_m4.sv"
run "mutant_quiesce_ignores_net" FAIL prove 8 tick_ctrl "$M/tick_m4.sv" "" NC=4 QUIESCE_CYCLES=4 START_GUARD=2
# M5: step_done one cycle early (does not wait for busy cores)
sed -e "s/assign step_done = (state == S_WAIT_UPD) \&\& (guard == '0) \&\& !core_any;/assign step_done = (state == S_WAIT_UPD) \&\& (guard == '0);/" \
    rtl/top/tick_ctrl.sv > "$M/tick_m5.sv"
run "mutant_done_ignores_busy" FAIL prove 8 tick_ctrl "$M/tick_m5.sv" "" NC=4 QUIESCE_CYCLES=4 START_GUARD=2

echo "----"
echo "formal: $pass as expected, $fail unexpected (summary: $OUT/summary.txt)"
[ "$fail" -eq 0 ]
