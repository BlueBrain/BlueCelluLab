:Comment :
:Reference : :		Molecular basis of functional diversity of voltage-gated potassium channels in mammalian brain,  Stuehmer W., Ruppersberg J.P., Schroerter K.H., Sakmann B.; EMBO J. 8:3235-3244(1989).

NEURON	{
	SUFFIX Kv1_4
	USEION k READ ek WRITE ik
	RANGE gKv1_4bar, gKv1_4, ik 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gKv1_4bar = 0.00001 (S/cm2) 
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gKv1_4	(S/cm2)
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
	gKv1_4 = gKv1_4bar*m*h
	ik = gKv1_4*(v-ek)
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
		mInf =  1.0000/(1+ exp((v - -21.7000)/-16.9000))
		mTau =  3.0000
		hInf =  1.0000/(1+ exp((v - -73.6000)/12.8000))
		hTau =  119.0000
	UNITSON
}
