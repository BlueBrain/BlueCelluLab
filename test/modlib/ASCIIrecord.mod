COMMENT
If the local variable step method is used then the only variables that should
be added are variables of the cell in which this FileRecord
has been instantiated.
ENDCOMMENT

NEURON {
    POINT_PROCESS ASCIIRecord
    POINTER ptr
    RANGE rank : rank of processor
    RANGE Dt
	RANGE tstart
	RANGE tstop
	RANGE rpts
}

PARAMETER {
    rank = 0
	Dt = .1 (ms)
	tstart = 0 (ms)
	tstop  = 0 (ms)
	rpts = 0
}

ASSIGNED {
    ptr
}

INITIAL {
	writeMeta()
    consolidateTiming()
    net_send(tstart, 1)
}


NET_RECEIVE(w) {
    recdata()
    if (t<tstop) {
        net_send(Dt, 1)
    }
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
	void* nextReport_;
	int handle_;
	char neuronName_[256];
	char rptName_[256];
    char path_[512];
	char fn_[256];
	int tstep_;
	int tsteps_;
	double tstart_;
	double tstop_;
	double Dt_;
    double** ptrs_; /* list of pointers to hoc variables */
    float* buf_; 
    float* map_; 
    int np_;
    int nm_;
    int tp_; /* temporary indicator of passed variable pointers */
    int mp_; /* temporary indicator of passed variable pointers */

	/* for ASCII version */
	char* line_;    /* this buffers the values of one timestep*/
	FILE* ascfile_; /* file pointer to ascii file*/
} Info;


#define INFOCAST Info** ip = (Info**)(&(_p_ptr))

#define dp double*

extern void nrn_register_recalc_ptr_callback(void (*f)());
extern Point_process* ob2pntproc(Object*);
extern double* nrn_recalc_ptr(double*);

static void recalc_ptr_callback() {
	Symbol* sym;
	hoc_List* instances;
	hoc_Item* q;
	/*printf("ASCIIrecord.mod recalc_ptr_callback\n");*/
	/* hoc has a list of the ASCIIRecord instances */
	sym = hoc_lookup("ASCIIRecord");
	instances = sym->u.template->olist;
	ITERATE(q, instances) {
		Info* InfoPtr;
		int i;
		Point_process* pnt;
		Object* o = OBJ(q);
		/*printf("callback for %s\n", hoc_object_name(o));*/
		pnt = ob2pntproc(o);
		_ppvar = pnt->_prop->dparam;
		INFOCAST;
                for (InfoPtr = *ip; InfoPtr != 0; InfoPtr = (Info*) InfoPtr->nextReport_)
                   for (i=0; i < InfoPtr->np_; ++i) 
                      InfoPtr->ptrs_[i]= nrn_recalc_ptr(InfoPtr->ptrs_[i]);

	}
}

ENDVERBATIM

CONSTRUCTOR { : double - loc of point process, int rank, string path, string filename
VERBATIM {
	static int first = 1;
	if (first) {
		first = 0;
		nrn_register_recalc_ptr_callback(recalc_ptr_callback);
	}
        if (ifarg(2)) {
                rank = (int) *getarg(2);
        }

	if (ifarg(4) && (hoc_is_str_arg(3)) && (hoc_is_str_arg(4))) {
		INFOCAST;
        	Info* info = 0;

		info = (Info*)hoc_Emalloc(sizeof(Info)); hoc_malchk(); 
		info->nextReport_ = 0;
		info->neuronName_[0]= 0;
		info->rptName_[0]= 0;
		info->handle_ = -1;
//		info->neuron_ = 0;
		info->tstart_ = 0;
		info->tstop_ = 0;
		info->Dt_ = 0;
//		info->rpt_ = 0;
	    info->ptrs_ = 0;
	    info->np_ = 0;
	    info->nm_ = 0;
		info->tp_ = 0;
		info->mp_ = 0;
		info->tstep_ = 0;
		info->tsteps_ = 0;
		info->line_ = 0;
		info->map_ = 0;
		
		*ip = info;
		sprintf((*ip)->path_, "%s", gargstr(3));
		sprintf((*ip)->fn_, "%s", gargstr(4));
	}

}
ENDVERBATIM
}

DESTRUCTOR {
VERBATIM { INFOCAST; Info* info = *ip; 
	for (info = *ip; info != 0; info = (Info*) info->nextReport_) {
		if (info->ascfile_) {
			fflush(info->ascfile_);
			fclose(info->ascfile_);
			info->ascfile_ = (FILE*)0;
		}
	}
}
ENDVERBATIM
}



FUNCTION newReport() { : string neuronname, string setname, double vars, double tstart, double tstop, double dt, string unit
VERBATIM { 
        INFOCAST; 
	Info* iPtr; 
	Info* info;
	for (iPtr = *ip; iPtr->nextReport_ != 0; iPtr = (Info*) iPtr->nextReport_) {}
	int newRpt = 0;
	int trial = 0;
	int fileOpen = 0;
	char fn[1024];
		
	if (iPtr->handle_ == -1) {
		if (hoc_is_str_arg(1)) sprintf(iPtr->neuronName_, "%s", gargstr(1));
		else sprintf(iPtr->neuronName_, "neuron");

		if (hoc_is_str_arg(2)) sprintf(iPtr->rptName_, "%s", gargstr(2));
		else sprintf(iPtr->rptName_, "report");

		sprintf(fn, "%s/%s.%s", iPtr->path_, iPtr->fn_, iPtr->rptName_);
		iPtr->ascfile_ = fopen(fn, "wb");
		fileOpen = iPtr->ascfile_ ? 1 : 0;
		while ((!fileOpen) && (trial < 20)) {
			iPtr->ascfile_ = fopen(fn, "wb");
			fileOpen = iPtr->ascfile_ ? 1 : 0; 
			trial += 1;
		}
		iPtr->handle_ = 0;
		
	} 
	// there already is a report --> need to create a new info struct 
	else {
		newRpt = 1;
//        	Info* info = 0;
		info = (Info*)hoc_Emalloc(sizeof(Info)); hoc_malchk(); 
		info->handle_ = -1;
		info->nextReport_ = 0;
		info->neuronName_[0]= 0;
		info->rptName_[0]= 0;
//		info->neuron_ = 0;
		info->tstart_ = 0;
		info->tstop_ = 0;
		info->Dt_ = 0;
//		info->rpt_ = 0;
	        info->ptrs_ = 0;
	        info->np_ = 0;
	        info->nm_ = 0;
		info->tp_ = 0;
		info->mp_ = 0;
		info->tstep_ = 0;
		info->tsteps_ = 0;
		info->line_ = 0;
		info->map_ = 0;
		
		if (hoc_is_str_arg(1)) sprintf(info->neuronName_, "%s", gargstr(1));
		else sprintf(info->neuronName_, "neuron");

		if (hoc_is_str_arg(2)) sprintf(info->rptName_, "%s", gargstr(2));
		else sprintf(info->rptName_, "report");

		sprintf(info->fn_, "%s", (*ip)->fn_);
		sprintf(info->path_, "%s", (*ip)->path_);
		
		sprintf(fn, "%s/%s.%s", info->path_, info->fn_, info->rptName_);
		info->ascfile_ = fopen(fn, "wb");
		fileOpen = info->ascfile_ ? 1 : 0;
		while ((!fileOpen) && (trial < 20)) {
			info->ascfile_ = fopen(fn, "wb");
			fileOpen = info->ascfile_ ? 1 : 0; 
			trial += 1;
		}	
		iPtr=info;
		
	}
	if (!fileOpen) {
		printf("[ASCIIrecord] Rank %d: Error! couldn't open file %s!\n", rank, fn);
		fflush(stdout);
		return;
	}

	char tmp[256];
	char unitStr[5];
		
	if (ifarg(3)) {
		iPtr->np_   = (int) *getarg(3);
	        iPtr->ptrs_ = (double**)hoc_Ecalloc(iPtr->np_, sizeof(double*)); hoc_malchk();
	        iPtr->buf_  = (float*)hoc_Ecalloc(iPtr->np_, sizeof(float)); hoc_malchk();
	}

	if (ifarg(4) && ifarg(5) && ifarg(6)) {
		iPtr->tstart_   = *getarg(4);
		iPtr->tstop_    = *getarg(5);
		iPtr->Dt_       = *getarg(6); 
//		printf("tstart = %g\n", info->tstart_);
//		printf("tstop = %g\n", info->tstop_);
//		printf("Dt = %g\n", info->Dt_);
		iPtr->tsteps_   = (int) (((iPtr->tstop_-iPtr->tstart_)/iPtr->Dt_)+.5);
//		printf("steps = %d\n", info->tsteps_);
		iPtr->tsteps_ += 1;
//		printf("steps = %d\n", info->tsteps_);
	}

	if (hoc_is_str_arg(7)) {
		sprintf(unitStr, "%s", gargstr(7));
	} else {
		sprintf(unitStr, "xx");
	}
		
	tstart = iPtr->tstart_;
	tstop  = iPtr->tstop_;
	Dt = iPtr->Dt_;
		
	/* initset can only be called once in case of ASCII reporting */

	if ((iPtr->ascfile_) && (iPtr->line_==0)) {        
		sprintf(tmp, "# neuron = %s\n", iPtr->neuronName_); fputs(tmp, iPtr->ascfile_);
		sprintf(tmp, "# report = %s\n", iPtr->rptName_); fputs(tmp, iPtr->ascfile_);
		sprintf(tmp, "# tstart = %g\n", iPtr->tstart_); fputs(tmp, iPtr->ascfile_);
		sprintf(tmp, "# tstop  = %g\n", iPtr->tstop_); fputs(tmp, iPtr->ascfile_);
		sprintf(tmp, "# Dt     = %g\n", iPtr->Dt_); fputs(tmp, iPtr->ascfile_);
		sprintf(tmp, "# tunit  = ms\n"); fputs(tmp, iPtr->ascfile_);
		sprintf(tmp, "# dunit  = %s\n", unitStr); fputs(tmp, iPtr->ascfile_);
		sprintf(tmp, "# rank   = %d\n", rank); fputs(tmp, iPtr->ascfile_);
	
		// allow 20 chars per double value. 
		iPtr->line_ = (char*)hoc_Ecalloc(iPtr->np_*20, sizeof(char)); hoc_malchk();
	}

//	if (!newRpt) *ip = info;
	if (newRpt == 1) {
		Info* tPtr; int hd = 1;
		for (tPtr = *ip; tPtr->nextReport_ != 0; tPtr = (Info*) tPtr->nextReport_, hd++) {}
		tPtr->nextReport_ = iPtr;
		iPtr->handle_ = hd;
	}
	rpts += 1;

	return iPtr->handle_;
}
ENDVERBATIM
}

FUNCTION newMapping() { : double rptHd, string mapping
VERBATIM { 
//printf("newMapping\n");
	INFOCAST; Info* iPtr = 0; Info* info = 0;
	char tmp[256];
	if (ifarg(1)) {
		for (iPtr = *ip; iPtr!= 0 && iPtr->handle_ != (int) *getarg(1); iPtr = (Info*) iPtr->nextReport_) {}
		if (iPtr == 0) printf("ERROR: given handle does not correspond to report!\n");
		else info=iPtr;
	}

	if (hoc_is_str_arg(2)) {
		if (strncmp(gargstr(2), "point", 5) == 0) {
		        info->map_  = (float*)hoc_Ecalloc(info->np_*3, sizeof(float)); hoc_malchk();
			info->nm_ = 3*info->np_;
		}
		else if (strncmp(gargstr(2), "compartment", 11) == 0) {
		        info->map_  = (float*)hoc_Ecalloc(info->np_, sizeof(float)); hoc_malchk();
			info->nm_ = info->np_;
		}	
	}

	if (info->ascfile_) {        

		if (strncmp(gargstr(2), "point", 5) == 0) {
		}
		else if (strncmp(gargstr(2), "compartment", 11) == 0) {
			int sec, soma, axon, basal, apic;
			sec = soma = axon = basal = apic = 0;

			if (ifarg(3)) {
				sec = (int) *getarg(3);
			}
			if (ifarg(4)) {
				soma = (int) *getarg(4);
			}
			if (ifarg(5)) {
				axon = (int) *getarg(5);
			}
			if (ifarg(6)) {
				basal = (int) *getarg(6);
			}
			if (ifarg(7)) {
				apic = (int) *getarg(7);
			}

			sprintf(tmp, "#\n"); fputs(tmp, info->ascfile_);
			sprintf(tmp, "# type      = compartment\n"); fputs(tmp, info->ascfile_);
			sprintf(tmp, "# totalSecs = %d\n", sec); fputs(tmp, info->ascfile_);
			sprintf(tmp, "# somaSecs  = %d\n", soma); fputs(tmp, info->ascfile_);
			sprintf(tmp, "# axonSecs  = %d\n", axon); fputs(tmp, info->ascfile_);
			sprintf(tmp, "# basalSecs = %d\n", basal); fputs(tmp, info->ascfile_);
			sprintf(tmp, "# apicSecs  = %d\n", apic); fputs(tmp, info->ascfile_);

		}
	}
	
}
ENDVERBATIM
}



PROCEDURE addvar() { : int rptHD, double* pd
VERBATIM { INFOCAST; Info* info = 0; Info* iPtr = 0;
//printf("addVar\n");
	if (ifarg(1)) {
		for (iPtr = *ip; iPtr!= 0 && iPtr->handle_ != (int) *getarg(1); iPtr = (Info*) iPtr->nextReport_) {}
		if (iPtr == 0) printf("ERROR: given handle does not correspond to report!\n");
		else info=iPtr;
	}

        if (info->tp_ < info->np_) {
	        info->ptrs_[info->tp_] = hoc_pgetarg(2);
//		if (ifarg(3)) {
//			info->map_[info->tp_] = (float) *getarg(3);
//		}
        	++(info->tp_);
        }
}
ENDVERBATIM
}

PROCEDURE addmapping() { : int rptHD, double var1, double var2, double var3
VERBATIM { INFOCAST; Info* info = 0; Info* iPtr = 0;
//printf("addMapping\n");
	if (ifarg(1)) {
		for (iPtr = *ip; iPtr!= 0 && iPtr->handle_ != (int) *getarg(1); iPtr = (Info*) iPtr->nextReport_) {}
		if (iPtr == 0) printf("ERROR: given handle does not correspond to report!\n");
		else info=iPtr;
	}

//	printf("getarg(2) = %g\n", *getarg(2));
        if (info->mp_ < info->np_) {
		if (ifarg(2)) {
			info->map_[info->mp_] = (float) *getarg(2);
//			printf("info->map = %g\n", info->map_[info->mp_]);
		}
		if (ifarg(3)) {
			info->map_[info->mp_+info->np_] = (float) *getarg(3);
		}
		if (ifarg(4)) {
			info->map_[info->mp_+info->np_*2] = (float) *getarg(4);
		}
        	++info->mp_;
        }
}
ENDVERBATIM
}

PROCEDURE recdata() {
VERBATIM { INFOCAST; Info* info = *ip;
	for (info = *ip; info != 0; info = (Info*) info->nextReport_) {
		if ((t >= info->tstart_) && (t <= info->tstop_)) {
			if (info->ascfile_) {
				int i, n;
				n = 0;
				for (i=0; i < info->tp_; i++) {
					n += sprintf(info->line_ + n, " %g", *info->ptrs_[i]);
				}
				
				sprintf(info->line_ + n, "\n");
					fputs(info->line_, info->ascfile_);
        		}
        	}
		++info->tstep_;		
        }	
}
ENDVERBATIM
}

PROCEDURE writeMeta() {
VERBATIM { INFOCAST; Info* info = *ip;
//printf("writeMeta()\n");
	char tmp[256];
	for (info = *ip; info != 0; info = (Info*) info->nextReport_) {
	        if (info->map_) {
			if (info->ascfile_) {
				int i = 0;
				sprintf(tmp, "# mapping  = "); fputs(tmp, info->ascfile_);
				for (i = 0; i< info->np_; i++) {
					sprintf(tmp, "%g ", info->map_[i]);	
					fputs(tmp, info->ascfile_);
				}
				sprintf(tmp, "\n"); fputs(tmp, info->ascfile_);
			}
	        }
	}
}
ENDVERBATIM
}

: currently, consolidateTiming() has a simple logic:
: 1. go through all reports and get minimum start time and maximum stop time
: 2. check whether all reports have same Dt
: 3. check whether the start and stop times are consistent with common Dt
PROCEDURE consolidateTiming() {
VERBATIM { INFOCAST; Info* info = *ip;
//printf("consolidateTiming()\n");
        double tmin = tstart; // values of last report!
        double tmax = tstop; // values of last report!
	double myeps=1e-10;
	double commonDt = Dt;
//printf("tmin=%g\n", tmin);
//printf("tmax=%g\n", tmax);
//printf("Dt=%g\n", Dt);
	for (info = *ip; info != 0; info = (Info*) info->nextReport_) {
		if (info->tstart_ < tmin) tmin = info->tstart_;
		if (info->tstop_ > tmax) tmax = info->tstop_;
                if (info->Dt_ != Dt) {
                	if (rank == 0) printf("[ASCIIrecord] Warning: Dt is not the same throughout reports! Setting Dt to %g\n", Dt);
                	info->Dt_ = Dt;
                }
        }
//printf("tmin=%g\n", tmin);
//printf("tmax=%g\n", tmax);
//printf("Dt=%g\n", Dt);

	for (info = *ip; info != 0; info = (Info*) info->nextReport_) {
		int steps2start = (int)((info->tstart_-tmin)/Dt+.5);
		double dsteps2start = (info->tstart_-tmin)/Dt;
		if (abs(dsteps2start - (double)(steps2start)) > myeps) {
			info->tstart_ = tmin + steps2start*Dt;
                	if (rank == 0) printf("[ASCIIrecord] Warning: Adjusting reporting start time to %g\n", info->tstart_);
		}
		int steps2stop = (int)((info->tstop_-tmin)/Dt+.5);
		double dsteps2stop = (info->tstop_-tmin)/Dt;
		if (abs(dsteps2stop - (double)(steps2stop)) > myeps) {
			info->tstop_ = tmin + steps2stop*Dt;
                	if (rank == 0) printf("[ASCIIrecord] Warning: Adjusting reporting stop time to %g\n", info->tstop_);
		}			
        }

	tstart = tmin;
	tstop = tmax;

//printf("tstart_=%g\n", info->tstart_);
//printf("tstop_=%g\n", info->tstop_);
//printf("Dt=%g\n", Dt);

/*
	phase* firstphase = (phase*)hoc_Emalloc(sizeof(phase)); hoc_malchk();
	firstphase->time = 0;
	firstphase->step = 0;
	firstphase->next = 0;
	int interval = 0;
	for (info = *ip; info != 0; info = (Info*) info->nextReport_) {
		phase* p = (phase*)hoc_Emalloc(sizeof(phase)); hoc_malchk();
		if (interval == 0) {
			p->time = info->tstart;
			p->step = info->Dt;
			firstphase->next = p;
			p = (phase*)hoc_Emalloc(sizeof(phase)); hoc_malchk();
			p->time = info->tstop;
			p->step = 0;
			p->next = 0;
			firstphase->next->next = p;
		} else {
			phase* pptr; phase* npptr = 0;
			for(pptr = firstphase; (pptr->next != 0) && (info->tstart < pptr->next->time); pptr = pptr->next) {}
			
			// the intervals are disjoint and interval is the last
			if (pptr->next == 0)) {
				p->time = info->tstart;
				p->step = info->Dt;
				pptr->next = p;
				p = (phase*)hoc_Emalloc(sizeof(phase)); hoc_malchk();
				p->time = info->tstop;
				p->step = 0;
				p->next = 0;
				pptr->next->next = p;
			} else {
                                npptr = pptr->next;
				p->time = info->tstart;
				// choose smallest timestep - need to check for consitstency 
                                if (info->Dt < pptr->step) p->step = info->Dt;
				else p->step = pptr->step;
				p->next = npptr;
				pptr = p;
                                p = (phase*)hoc_Emalloc(sizeof(phase)); hoc_malchk();

				double Dt_back = 0;
				// check whether intermediate intervals have correct time step 
				for (; (pptr->next != 0) && (info->tstop < pptr->next->time); pptr = pptr->next) {
					Dt_back = pptr->step;
                                	if (info->Dt < pptr->step) pptr->step = info->Dt;
				}
                                // interval ends is at end 
                                if (pptr->next == 0) {
                                         p->time = info->tstop;
                                         p->step = 0;
                                         p->next = 0;
                                         pptr->next = p;
                                } else {
					p->time = info->tstop;
					p->step = Dt_back;
					p->next = pptr->next;
					pptr->next = p
				}
 
			}
		}
		interval++;
	}
	
 	delete[] timesteps;
	delete[] times;
*/
}
ENDVERBATIM
}
