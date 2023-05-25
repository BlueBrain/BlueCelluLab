clean:
	@find . -name "*.pyc" -exec rm -rf {} \;
	rm -rf *.png

	rm -f .coverage
	rm -f *.coverage
	rm -f coverage.xml
	rm -f *.nrndat
