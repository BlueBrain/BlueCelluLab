:Comment :
:Reference : :		Science 244:221-224 (1989)

NEURON	{
	SUFFIX Kv1_1
	USEION k READ ek WRITE ik
	RANGE gKv1_1bar, gKv1_1, ik 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gKv1_1bar = 0.00001 (S/cm2) 
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gKv1_1	(S/cm2)
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
	gKv1_1 = gKv1_1bar*m*h*h
	ik = gKv1_1*(v-ek)
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
		mInf =  1.0000/(1+ exp((v - -30.5000)/-11.3943))
		mTau =  30.0000/(1+ exp((v - -76.5600)/26.1479))
		hInf =  1.0000/(1+ exp((v - -30.0000)/27.3943))
		hTau =  15000.0000/(1+ exp((v - -160.5600)/-100.0000))
	UNITSON
}
