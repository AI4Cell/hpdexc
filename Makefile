PYTHON?=.venv/bin/python
BUILD_DIR?=build

.PHONY: all build debug rebuild clean test wheel install debug-test

all: build

build:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release -DPython3_ROOT_DIR=.venv -DPython3_EXECUTABLE=$(PYTHON) .. && $(MAKE) -j

debug:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Debug -DPython3_ROOT_DIR=.venv -DPython3_EXECUTABLE=$(PYTHON) .. && $(MAKE) -j

rebuild: clean build

debug-rebuild: clean debug

clean:
	rm -rf $(BUILD_DIR)
	find src/python/hpdex/backen -name "*.so" -delete || true

test: build
	PYTHONPATH=src/python uv run $(PYTHON) -c "import hpdex as hp, numpy as np, scipy.sparse as sp; print('ping:', hp.ping()); A=sp.csc_matrix(np.eye(3)); ref=np.array([1,0,0],dtype=np.uint8); tar=np.array([0,1,1],dtype=np.uint8); print(hp.mwu_v1_stats_csc(A,ref,tar,calc_tie=False))"

debug-test: debug
	@echo "=== 调试模式测试 ==="
	@echo "使用LLDB调试Python脚本..."
	@echo "运行: lldb --source debug_complete.lldb $(PYTHON) t.py"
	PYTHONPATH=src/python lldb --source debug_complete.lldb $(PYTHON) t.py

wheel:
	uv build

install: wheel
	uv pip install dist/*.whl --force-reinstall


test_rank_sum:
	g++ -std=c++17 -o test_rank_sum src/cpp/test_main.cpp && ./test_rank_sum