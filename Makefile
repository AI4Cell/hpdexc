# ========== 基础配置 ==========
PYTHON     ?= .venv/bin/python
BUILD_DIR  ?= build
CMAKE      ?= cmake
GENERATOR  ?= Ninja
BUILD_TYPE ?= Release

# ========== 参数转换 ==========
# 把 args 列表（空格分隔）变成 cmake -D 参数
# 例如 args="multithread vqsort_static"
# 会生成 -DHPDEXC_MULTITHREAD=ON -DHPDEXC_VQSORT_STATIC=ON
DEFINES = $(foreach a,$(args),-DHPDEXC_$(shell echo $(a) | tr '[:lower:]' '[:upper:]')=ON)

# ========== 伪目标 ==========
.PHONY: all build rebuild clean test wheel install git push pull debug-test

# 默认目标
all: build

# ========== Git 操作 ==========
git:
	@git status

push:
	@git add .
	@git commit -m "$(msg)"
	@git push origin

pull:
	@git pull

# ========== 构建相关 ==========
build:
	@echo "===> Configuring (BUILD_TYPE=$(BUILD_TYPE), DEFINES=$(DEFINES))"
	$(CMAKE) -S . -B $(BUILD_DIR) -G "$(GENERATOR)" -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(DEFINES)
	@echo "===> Building"
	$(CMAKE) --build $(BUILD_DIR) -j

clean:
	@echo "===> Removing $(BUILD_DIR)"
	rm -rf $(BUILD_DIR)

rebuild: clean build

# ========== Python 相关 ==========
test:
	@echo "===> Running pytest"
	$(PYTHON) -m pytest -q

wheel:
	@echo "===> Building wheel"
	$(PYTHON) -m build

install:
	@echo "===> Installing package"
	$(PYTHON) -m pip install -e .

# Debug 专用测试
debug-test: BUILD_TYPE=Debug
debug-test: rebuild test
