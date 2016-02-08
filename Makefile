test:
	nosetests wpca

doctest:
	nosetests --with-doctest wpca

test-coverage:
	nosetests --with-coverage --cover-package=wpca

test-coverage-html:
	nosetests --with-coverage --cover-html --cover-package=wpca
