.PHONY: all clean uninstall install test

all:
	@grep -Ee '^[a-z].*:' Makefile | cut -d: -f1 | grep -vF all

clean:
	rm -rf build/ dist/ *.egg-info/ expm/*.c expm/*.so expm/*.h expm/__pycache__ benchmarks/__pycache__

uninstall: clean
	@echo pip uninstalling expm
	$(shell pip uninstall -y expm >/dev/null 2>/dev/null)
	$(shell pip uninstall -y expm >/dev/null 2>/dev/null)
	$(shell pip uninstall -y expm >/dev/null 2>/dev/null)

install: uninstall
	python setup.py install

benchmark:
	python benchmarks/benchmarks.py

test:
	python test/test_*.py

