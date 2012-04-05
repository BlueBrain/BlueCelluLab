NEURON	{
	SUFFIX HHO
	USEION na READ ena WRITE ina
	RANGE gHHObar, gHHO,O,C,I,K,J,TMP
	RANGE SHFT,tau_a,k_a
	RANGE tau_ci,k_ci,tau_i,tau_a_beta

}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gHHObar = 0.00684 (S/cm2) 
	SHFT
	TMP
	tau_i = 0.5
	K = 1000
	J = 3.2
	tau_a = 0.3
	k_a = 3
	tau_ci = 30
	k_ci = 4
	tau_a_beta = 0.3
	
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gHHO	(S/cm2)
}

STATE	{ 
	O
	C
	I
	H
}

BREAKPOINT	{
	TMP = O	
	SOLVE scheme METHOD sparse
	:SOLVE deriv METHOD derivimplicit
	gHHO = gHHObar*O
	ina = gHHO*(v-ena)
}

DERIVATIVE deriv{
	O' = alpha_a(v+SHFT+K*J*O)*(H-O) - ( 1/tau_i + beta_a(v+SHFT+K*J*O) )*O
	H' = alpha_ci(v+SHFT)*(1-H) - beta_ci(v+SHFT)*(H-O) - O/tau_i
}

KINETIC scheme{
	~C <-> O (alpha_a(v + SHFT + K*J*TMP),beta_a(v + SHFT + K*J*TMP))
	~I <-> C (alpha_ci(v+ SHFT ),beta_ci(v+ SHFT ))
	~O <-> I (1/tau_i,0)
	CONSERVE C + O + I = 1	
}

FUNCTION alpha_a(vv){
	alpha_a = 1/(1+exp(-(vv--35)/k_a))/tau_a
}

FUNCTION beta_a(vv){
	beta_a 	= 1/(1+exp( (vv--35)/k_a))/tau_a_beta
}
	
FUNCTION alpha_ci(vv){
	alpha_ci = 1/(1+exp(-(vv--60)/k_ci))/tau_ci
}

FUNCTION beta_ci(vv){
	beta_ci  = 1/(1+exp( (vv--60)/k_ci))/tau_ci
}



INITIAL{
	SOLVE scheme STEADYSTATE sparse
	O = 0
	H = 1
}

