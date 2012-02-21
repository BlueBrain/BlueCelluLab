NEURON	{

	SUFFIX  capump

	USEION ca READ ica WRITE cai

	RANGE decay, conversion

	GLOBAL base_concentration

}



UNITS	{

	(mV) = (millivolt)

	(mA) = (milliamp)

	FARADAY = (faraday) (coulombs)

	(molar) = (1/liter)

	(mM) = (millimolar)

}



PARAMETER	{

	conversion = 10

	decay = 1500 (ms)

	base_concentration = 2e-4(mM)

}



ASSIGNED	{ica (mA/cm2)}



STATE	{cai (mM) }



INITIAL{

	cai = base_concentration

}



BREAKPOINT	{ SOLVE state METHOD cnexp }



DERIVATIVE state	{

	cai' = -(ica*conversion/FARADAY) - (cai-base_concentration)/decay

}


