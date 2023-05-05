all: install
install:
	pip install -i https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/+simple --upgrade .
test: clean install_tox
	tox -v
install_tox:
	pip install tox
clean:
	@find . -name "*.pyc" -exec rm -rf {} \;
	rm -rf *.png

	rm -f .coverage
	rm -f *.coverage
	rm -f coverage.xml
	rm -f *.nrndat
