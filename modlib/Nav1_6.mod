:[$URL: https://bbpteam.epfl.ch/svn/analysis/trunk/IonChannel/xmlTomod/CreateMOD.c $]
:[$Revision: 349 $]
:[$Date: 2007-05-08 16:31:38 +0200 (Tue, 08 May 2007) $]
:[$Author: rajnish $]
:Comment :
:Reference : :		Goldin, J. of Neuroscience 1998

NEURON	{
	SUFFIX Nav1_6
	USEION na READ ena WRITE ina
	RANGE gNav1_6bar, gNav1_6, ina 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gNav1_6bar = 0.00001 (S/cm2) 
	 
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gNav1_6	(S/cm2)
	mInf
	mTau
}

STATE	{ 
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gNav1_6 = gNav1_6bar*m
	ina = gNav1_6*(v-ena)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
}

INITIAL{
	rates()
	m = mInf
}

PROCEDURE rates(){
	UNITSOFF
		mInf = 1.0000/(1+ exp(-0.03937*4.2*(v - -17.000)))
		mTau = 1
	UNITSON
}
