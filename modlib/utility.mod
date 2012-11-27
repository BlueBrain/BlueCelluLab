COMMENT
/**
 * @file utility.mod
 * @brief Collection of functions to give hoc access to extended functionality
 * @author king
 * @date 2009-06-12
 * @remark Copyright © BBP/EPFL 2005-2011; All rights reserved. Do not distribute without further notice.
 */
ENDCOMMENT

NEURON {
	SUFFIX nothing
}

VERBATIM

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

extern char* gargstr();
extern char** hoc_pgargstr();
extern void hoc_assign_str();
extern double chkarg();
extern errno;

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

FUNCTION checkDirectory() {
VERBATIM
    char* dirName = gargstr(1);
    
    struct stat st;
    if ( stat(dirName, &st) == 0) {
        if( !S_ISDIR(st.st_mode) ) {
            fprintf( stderr, "%s does not name a directory.\n", dirName );
            return -1;
        }
        return 0;
    }
    else if( errno == ENOENT ) {
        fprintf( stdout, "Directory %s does not exist.  Creating...\n", dirName );
        int res = mkdir( dirName, 0777 );
        if( res < 0 ) {
            fprintf( stderr, "Failed to create directory %s.\n", dirName );
            return -1;
        }
        return 0;
    }
ENDVERBATIM
}
