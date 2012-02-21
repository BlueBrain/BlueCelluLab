NEURON {
	SUFFIX nothing
}

VERBATIM
extern char* gargstr();
extern char** hoc_pgargstr();
extern void hoc_assign_str();
extern double chkarg();
ENDVERBATIM

: like File.scanstr
: icp = util_scanstr(instr, icp, outstr)
: return -1 if at end of instr and outstr is empty
: no checking for valid icp
FUNCTION util_scanstr() {
VERBATIM
{
	int ibegin, iend, i, flag;
	char *instr, **outstr, *cp, cend;
	instr = gargstr(1);
	ibegin = (int)chkarg(2, 0., 1e9);
	outstr = hoc_pgargstr(3);

	flag = 0;
	for (cp = instr+ibegin; *cp; ++cp, ++ibegin) {
		if (*cp != ' ' && *cp != '\n' && *cp != '\t') {
			flag = 1;
			break;
		}
	}
	if (flag == 0) {
		return -1.;
	}

	for (iend=ibegin; *cp; ++cp, ++iend) {
		if (*cp == ' ' || *cp == '\n' || *cp == '\t') {
			break;
		}
	}

	cend = instr[iend];
	instr[iend] = '\0';
	hoc_assign_str(outstr, instr + ibegin);
	instr[iend] = cend;
	return (double)iend;
}
ENDVERBATIM
}

: util_right(instr, icp, outstr)
: no checking for valid icp
PROCEDURE util_right() {
VERBATIM
{
	int icp, i, flag;
	char* instr, **outstr;
	instr = gargstr(1);
	icp = (int)chkarg(2, 0., 1e9);
	outstr = hoc_pgargstr(3);

	hoc_assign_str(outstr, instr+icp);
}
ENDVERBATIM
}

FUNCTION util_strhash() {
VERBATIM
{
	int i, j, k, h, n;
	char* s;
	s = gargstr(1);
	n = (int)chkarg(2, 1., 1e9);
	j = strlen(s);
	h = 0;
	for (i = 1; i < 5 && j >= i; ++i) {
		h *= 10;
		h += s[j - i];
	}

	return (double)(h%n);
}
ENDVERBATIM
}
