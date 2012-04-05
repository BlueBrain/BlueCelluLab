:Comment :
:Reference : :		Effects of Charbdotoxin on K+ channel(Kv1.2) deactivation and inactivation kinetics, European Journal of Pharmacology, 314 (1996), 357-364

NEURON	{
	SUFFIX Kv1_2
	USEION k READ ek WRITE ik
	RANGE gKv1_2bar, gKv1_2, ik 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gKv1_2bar = 0.00001 (S/cm2) 
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gKv1_2	(S/cm2)
	hInf
	hTau
	mInf
	mTau
}

STATE	{ 
	h
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gKv1_2 = gKv1_2bar*h*m
	ik = gKv1_2*(v-ek)
}

DERIVATIVE states	{
	rates()
	h' = (hInf-h)/hTau
	m' = (mInf-m)/mTau
}

INITIAL{
	rates()
	h = hInf
	m = mInf
}

PROCEDURE rates(){
	UNITSOFF
		hInf =  1.0000/(1+ exp((v - -22.0000)/11.3943))
		hTau =  15000.0000/(1+ exp((v - -46.5600)/-44.1479))
		mInf =  1.0000/(1+ exp((v - -21.0000)/-11.3943))
		mTau =  150.0000/(1+ exp((v - -67.5600)/34.1479))
	UNITSON
}
