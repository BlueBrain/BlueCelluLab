:Ih current for thalamic interneurons
: Based on data and model in Halnes et al., 2011, PLOS Comp. Bio.

NEURON	{
	SUFFIX IN_ih
	NONSPECIFIC_CURRENT ih
	RANGE gh_max, g_h, i_rec
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gh_max = 2.2e-5(S/cm2)
	e_h =  -45.0 (mV)
        celsius (degC)
	q10 = 4	: Santoro et al., 2000
}

ASSIGNED	{
	v	(mV)
	ih	(mA/cm2)
	g_h	(S/cm2)
	mInf
	mTau
	tcorr		:Add temperature correction
	i_rec
}

STATE	{
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	g_h = gh_max*m
	ih = g_h*(v-e_h)
	i_rec = ih
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
}

INITIAL{
	rates()
	m = mInf
	:tcorr = q10^((celsius-34)/10)  :EI: Recording temp. 36 C Halnes, 2011, close to sim (34 C) -> no temperature correction
}

UNITSOFF
PROCEDURE rates(){
        v = v + 0
        mInf = 1/(1+exp((v+96)/10)) : Halnes et al., 2011, ModelDB 140229
        mTau = exp((v+250)/30.7) / ( 1 + exp((v+78.8)/5.78)): Halnes et al., 2011, ModelDB 140229
        v = v - 0
}
UNITSON
