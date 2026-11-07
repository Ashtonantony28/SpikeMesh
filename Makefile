PY ?= python3
.PHONY: test test-golden test-train test-learn test-core test-noc test-compiler test-system test-fpga lint
test: test-golden test-compiler test-train test-learn test-core test-noc test-system test-fpga
test-golden:   ; $(PY) -m pytest -q tests/golden
test-compiler: ; $(PY) -m pytest -q tests/compiler
test-train:    ; $(PY) -m pytest -q tests/train
test-core:     ; $(PY) -m pytest -q tests/core
test-noc:      ; $(PY) -m pytest -q tests/noc
test-system:   ; $(PY) -m pytest -q tests/system
test-learn:    ; $(PY) -m pytest -q tests/learn
test-fpga:     ; $(PY) -m pytest -q tests/fpga
lint:          ; verilator --lint-only -Wall -Irtl/core -Irtl/noc -Irtl/top rtl/top/spikemesh_top.sv
