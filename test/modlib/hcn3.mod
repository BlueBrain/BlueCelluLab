:[$URL: https://bbpteam.epfl.ch/svn/analysis/trunk/IonChannel/xmlTomod/CreateMOD.c $]
:[$Revision: 347 $]
:[$Date: 2007-05-07 23:30:07 +0200 (Mon, 07 May 2007) $]
:[$Author: rajnish $]
:Comment :
:Reference :Moosmang et al. Eur. J. Biochem. 268, 1646-1652 (2001)

NEURON	{
	SUFFIX hcn3
	NONSPECIFIC_CURRENT ihcn
	RANGE ghcn3bar, ghcn3, ihcn
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	ghcn3bar = 0.00001 (S/cm2) 
	ehcn = -45.0 (mV)
}

ASSIGNED	{
	v	(mV)
	ihcn	(mA/cm2)
	ghcn3	(S/cm2)
	mInf
	mTau
}

STATE	{ 
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	ghcn3 = ghcn3bar*m
	ihcn = ghcn3*(v-ehcn)
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
		mInf = 1.0000/(1+exp((v- -96)/8.6))
		mTau = 265.0000
	UNITSON
}
