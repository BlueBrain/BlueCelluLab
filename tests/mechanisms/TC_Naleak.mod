TITLE Leak potassium current
: From Amarillo et al., 2014

NEURON {
	SUFFIX TC_Naleak
	:USEION  na READ ena WRITE ina 
	USEION na WRITE ina :Read ena from the modfile
	RANGE g, i_rec, ena
}

:CONSTANT {
	:Q10 = 3 (1) : To check, recordings at room temperature
:}

UNITS {
	(mA) = (milliamp)
	:(uA) = (microamp)
	(mV) = (millivolt)
	(S) = (siemens)
}

PARAMETER {
	ena = 0		(mV)
	g = 3.0e-6	(S/cm2)	<0,1e9>
}

ASSIGNED {
	v	(mV)
	ina	(mA/cm2)
	:qt (1)
	i_rec
}


BREAKPOINT {
	ina = g*(v - ena)
	i_rec = ina
}







