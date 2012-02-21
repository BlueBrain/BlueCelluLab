
COMMENT
	Written by Albert Gidon & Leora Menhaim (2004).
ENDCOMMENT

UNITS {
 	(mA) = (milliamp)
 	(mV) = (millivolt)
	(S) = (siemens)		
}

DEFINE EX 1
DEFINE IN 0

NEURON {
	SUFFIX noise
	NONSPECIFIC_CURRENT i
	RANGE iNoise
	RANGE gauss_white_noise
	RANGE Noisetau
 }
	
PARAMETER {
        sqrt200
        Noisetau =2000 (ms)
}
 
ASSIGNED {
        v (mV)
        i (mA/cm2)
        iNoise(mA/cm2)
		
		gauss_white_noise
}


BREAKPOINT {
        SOLVE states METHOD after_cvode
		:-------------------------
		:Fellous et. al. 2003 
		:the function normrand doesn't work here well 
		:	I don't know why, but it works well from states()
		:gauss_white_noise = normrand(0,1)
		iNoise = sqrt200*gauss_white_noise/Noisetau
		i = iNoise
		:-------------------------
}
 

INITIAL {
	:Noisetau = 3e4:1e6*(3e-6)/(1/1e4) : (ms)
	sqrt200 = sqrt(200) :(mV*ms0.5)
}

PROCEDURE states (){  
	:-------------------------------------------------
	:Synaptic background noise controls the input/output 
	:	characteristics of single cells in an in vitro model of 
	:	in vivo activity. 
	: Fellous JM, Rudolph M, Destexhe A, Sejnowski TJ.
	:			Neuroscience. 2003;122(3):811-29. 
	:BUG - I don't know why but this line need to be here.
	: and isn't working in BREAKPOINT
	gauss_white_noise = normrand(0,1): (ms0.5*S/cm2)
}
