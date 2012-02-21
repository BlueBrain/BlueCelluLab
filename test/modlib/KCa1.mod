:[$URL: https://bbpteam.epfl.ch/svn/analysis/trunk/IonChannel/xmlTomod/CreateMOD.c $]
:[$Revision: 349 $]
:[$Date: 2007-05-08 16:31:38 +0200 (Tue, 08 May 2007) $]
:[$Author: rajnish $]
:Comment :
:Reference : :		E. Moczydlowski, R Latorre (1983) J. Gen.Physiol 82, 511-542

NEURON	{
	SUFFIX KCa1_0237
    USEION ca READ cai
	USEION k READ ek WRITE ik
	RANGE gKCa1bar, gKCa1, ik, BBiD 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
    FARADAY = (faraday)  (kilocoulombs)
	R = (k-mole) (joule/degC)
}

PARAMETER	{
	gKCa1bar = 0.00001 (S/cm2) 
    cai		(mM) : 1e-3
    ca      (mM)
	BBiD = 237 
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gKCa1	(S/cm2)
	mInf
	mTau
}

STATE	{ 
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gKCa1 = gKCa1bar*m
	ik = gKCa1*(v-ek)
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
		mInf = 0.48/(1 + (0.18*exp(-2*0.84*FARADAY*v/R/(273.15 + celsius)))/ca)
		mTau = 0.28/(1 + ca/(0.011*exp(-2*FARADAY*v/R/(273.15 + celsius))))
	UNITSON
}
