TEST_REQUIREMENTS=nose coverage
 
all: install
install:
	pip install -i https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/+simple --upgrade .
test: clean install_tox
	tox -v
test-gpfs: clean install_tox
	tox -v -e py27-gpfs
install_tox:
	pip install tox
clean:
	@find . -name "*.pyc" -exec rm -rf {} \;
	@find . -name "*.png" -exec rm -rf {} \;
install_test_requirements:
	pip install -q $(TEST_REQUIREMENTS) --upgrade
doc: install
	pip install -q sphinx sphinx-autobuild sphinx_rtd_theme -I
	sphinx-apidoc -o docs/source bglibpy
	cd docs; $(MAKE) clean; $(MAKE) html
docpdf:                                                                         
	pip install sphinx sphinx-autobuild -I
	cd docs; $(MAKE) clean; $(MAKE) latexpdf
docopen: doc
	open docs/build/html/index.html
devpi:
	rm -rf dist
	python setup.py sdist
	upload2repo -t python -r dev -f `ls dist/bglibpy-*.tar.gz` 

