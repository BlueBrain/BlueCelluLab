NEURON	{
	SUFFIX caout
	USEION ca READ ica WRITE cao
	RANGE trans, fhspace
}

UNITS	{
	(mV) = (millivolt)
	(mA) = (milliamp)
	FARADAY = (faraday) (coulombs)
	(molar) = (1/liter)
	(mM) = (millimolar)
}

PARAMETER	{
	cabath = 2 (mM)
	fhspace = 300 (angstrom)
	trans = 50 (ms)
}

ASSIGNED	{ica (mA/cm2)}

STATE	{cao (mM) }

BREAKPOINT	{ SOLVE state METHOD cnexp }

DERIVATIVE state	{
	cao' = ica*(1e8)/(fhspace*FARADAY) + (cabath-cao)/trans
}
