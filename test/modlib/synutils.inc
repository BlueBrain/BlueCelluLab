NEURON{}

VERBATIM 
extern double* vector_vec(); 
extern int vector_capacity(); 
extern void* vector_arg(); 
ENDVERBATIM      
 

PARAMETER{
	space_synutils	
	use_play_synutils
}


PROCEDURE play() { 
	use_play_synutils = 1
	VERBATIM 
	void** vv; 
	vv = (void**)(&space_synutils); 
	*vv = (void*)0; 
	if (ifarg(1)) { 
	 *vv = vector_arg(1); 
	} 
	ENDVERBATIM 
} 

PROCEDURE remove_play(){
	use_play_synutils = 0
}

INITIAL{LOCAL etime
VERBATIM
	void* vv1; int size; double *px; int ii;
 	vv1 = *((void**)(&space_synutils)); 
	if (vv1 && use_play_synutils) { 
	 	size  = vector_capacity(vv1);
		px = vector_vec(vv1); 
		for(ii=0;ii<size;ii++){
			_letime = px[ii];
			ENDVERBATIM 
				net_send (etime) 
			VERBATIM
		} 
	}
ENDVERBATIM
}