:Comment :
:Reference : :		Voltage-gated K+ channels in layer 5 neocortical pyramidal neurones from young rats:subtypes and gradients,Korngreen and Sakmann, J. Physiology, 2000

NEURON	{
	SUFFIX Kfast
	USEION k READ ek WRITE ik
	RANGE gKfastbar, gKfast, ik 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gKfastbar = 0.00001 (S/cm2) 
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gKfast	(S/cm2)
	mInf
	mTau
	hInf
	hTau
}

STATE	{ 
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gKfast = gKfastbar*m*h
	ik = gKfast*(v-ek)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
	h' = (hInf-h)/hTau
}

INITIAL{
	rates()
	m = mInf
	h = hInf
}

PROCEDURE rates(){
	UNITSOFF
		mInf =  1/(1 + exp(-(v+47)/29))
		mTau =  (0.34+0.92*exp(-((v+71)/59)^2))
		hInf =  1/(1 + exp(-(v+56)/-10))
		hTau =  (8+49*exp(-((v+73)/23)^2))
	UNITSON
}
