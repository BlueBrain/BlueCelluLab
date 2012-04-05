COMMENT
This object is designed to accept a pointer from a caller and use that to instantiate an array
for the storage of neuron addresses.  For example, if I want access to synapse parameters,
I would pass in a pointer (triple pointer?) and after an initial array is allocated, the passed in
pointer would be set to refer to the mem block.  Then bglib would pass in the memory addresses that
need to be inserted into that array.
ENDCOMMENT

NEURON {
	POINT_PROCESS MemoryAccess
	POINTER ptr
}

PARAMETER {
}

ASSIGNED {
	ptr
}

VERBATIM

extern double* hoc_pgetarg(int iarg);
extern double* getarg(int iarg);
extern char* gargstr(int iarg);
extern int hoc_is_str_arg(int iarg);
extern int nrnmpi_numprocs;
extern int nrnmpi_myid;
extern int ifarg(int iarg);
extern double chkarg(int iarg, double low, double high);

typedef struct {
    // do I need a file ptr?  I don't want to write anything out
	//FILE* file_;
    
    /**
     * Pointer to array of pointers referenceing neuron varables (such as voltage or conductances)
     */
	double** ptrs_; /* list of pointers to hoc variables */
    
    /**
     * In the event that each line needs some padding up front (e.g. for a tag), this stores
     * the offset into line where actual data can start to be written
     */
	//int n_; /* line_ to line_+n contains a constant tag */
    
    /**
     * The number of pointers stored in the vector
     */
	int np_;
    
    /**
     * Vector capacity
     */
	int psize_;
    
    //char* line_; /* for construction of a line to be printed */
    /**
     * space available for writing the data to a string for final output
     */
	//int linesize_;
} Info;

#define INFOCAST Info** ip = (Info**)(&(_p_ptr))

ENDVERBATIM

CONSTRUCTOR {
VERBATIM {
	INFOCAST;
	Info* info = (Info*)hoc_Emalloc(sizeof(Info)); hoc_malchk();
	//info->file_ = (FILE*)0;
	info->psize_ = 10;
	info->ptrs_ = (double**)hoc_Ecalloc(info->psize_, sizeof(double*)); hoc_malchk();
	//info->linesize_ = 200;
	//info->line_ = (char*)hoc_Ecalloc(info->linesize_, sizeof(char)); hoc_malchk();
	info->np_ = 0;
	//info->n_ = 0;
	*ip = info;
	
    //fprintf( stderr, "The truth now - %x %d\n", (int) &(info->np_), info->np_ );
    //check for argument containing memory address, set to reference the array
}
ENDVERBATIM
}

DESTRUCTOR {
VERBATIM {
	INFOCAST; Info* info = *ip;
	free(info);
}
ENDVERBATIM
}

PROCEDURE reveal() { : char* string char* string2
VERBATIM { INFOCAST; Info* info = *ip;
    char *helper1 = gargstr(1);
    char *helper2 = gargstr(2);
    
    //fprintf( stderr, "received %s and %s\n", helper1, helper2 );
    //fprintf( stderr, "What are you? %d\n", info->np_ );
    
    double**** access = NULL;

    //I want to parse this character string into a hex address.  How do I control whether I get
    // a 64 vs a 32 bit value? can I just put it directly into my variable?
    
    //fprintf(stderr, "extract the address\n" );
    sscanf( helper1, "%x", &access );
    
    //if I were to print access now, it should contain the address of neurodamus's object
    //if I were to dereference access now, it should contain zero
    //fprintf( stderr, "maintained address? %x\n", (int) access );
    //fprintf( stderr, "is it NULL? %x\n", (int) *access );
    
    //now, derefernce access and make it point to the array reference (not the array itself,
    // since that will potentially be reallocated as more variables are added)
    
    //fprintf( stderr, "dereference and assign address\n" );
    *access = &(info->ptrs_);
    
    int **access2;
    sscanf( helper2, "%x", &access2 );
    *access2 = &(info->np_);
    
    //fill in value to check it
    //*(*access2) = 5;
    //fprintf(stderr, "I filled in %d %d %d\n", *access2, info->np_, *(*access2) );
}
ENDVERBATIM
}

PROCEDURE nothing() { : double* pd
VERBATIM { INFOCAST; Info* info = *ip;
	//++info->np_;
    fprintf( stderr, "incremented to %d\n", info->np_ );
	/* allow 20 chars per double */
}
ENDVERBATIM
}

PROCEDURE addvar() { : double* pd
VERBATIM { INFOCAST; Info* info = *ip;
	if (info->np_ >= info->psize_) {
		info->psize_ *= 2;
		info->ptrs_ = (double**) hoc_Erealloc(info->ptrs_, info->psize_*sizeof(double*)); hoc_malchk();
	}
	info->ptrs_[info->np_] = hoc_pgetarg(1);
    //fprintf( stderr, "stored %d into %d (value = %lf)\n", info->ptrs_[info->np_], info->ptrs_, *info->ptrs_[info->np_] );
	++info->np_;
    //fprintf( stderr, "incremented to %d\n", info->np_ );
	/* allow 20 chars per double */
}
ENDVERBATIM
}

PROCEDURE prdata() {
VERBATIM { INFOCAST; Info* info = *ip;
}
ENDVERBATIM
}

FUNCTION prstr() {
VERBATIM { INFOCAST; Info* info = *ip;
}
ENDVERBATIM
}

FUNCTION wopen() {
VERBATIM { INFOCAST; Info* info = *ip;
}
ENDVERBATIM
}

FUNCTION aopen() {
VERBATIM { INFOCAST; Info* info = *ip;
}
ENDVERBATIM
}

PROCEDURE close() {
VERBATIM { INFOCAST; Info* info = *ip;
}
ENDVERBATIM
}

