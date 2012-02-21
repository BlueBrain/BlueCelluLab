:Comment :
:Reference : :		Cloning and expression of a human voltage-gated potassium channel. A novel member of the RCK potassium channel family, The EMBO JOURNAL, vol. 9, no.6, 1749-1756,(1990)

NEURON	{
	SUFFIX Kv1_6
	USEION k READ ek WRITE ik
	RANGE gKv1_6bar, gKv1_6, ik 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gKv1_6bar = 0.00001 (S/cm2) 
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gKv1_6	(S/cm2)
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
	gKv1_6 = gKv1_6bar*m*h
	ik = gKv1_6*(v-ek)
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
		mInf =  1/(1+exp(((v -(-20.800))/(-8.100))))
		mTau =  30.000/(1+exp(((v -(-46.560))/(44.140))))
		hInf =  1/(1+exp(((v -(-22.000))/(11.390))))
		hTau =  5000.000/(1+exp(((v -(-46.560))/(-44.140))))
	UNITSON
}
