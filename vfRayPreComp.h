/**
 *   Visible Faces Ray Casting (VF-Ray)
 *
 * C Ansi only code -- January, 2008
 *
 */

/**
 *   VF-Ray Pre-Computation
 *
 * C header.
 *
 */

/// ----------------------------------   Definitions   ------------------------------------

#ifndef _VFRAYPRECOMP_H_
#define _VFRAYPRECOMP_H_

extern "C" {
#include <stdio.h>
#include <math.h>
#include <assert.h>
}

#include <iostream>
#include <fstream>

#include <vector>
#include <set>

#define MOD4(x,y)           ((x+y)%4) ///< module operation

/// Object-space to Image-space linear transformation
/// @arg obj object coordinate
/// @arg res resolution coordinate
/// @return image coordinate
#define OBJ_TO_IMG(obj,res)       ((uint)(( res - 1 ) * ( obj + 1 ) / 2.0) )

/// Image-space to Object-space linear transformation
/// @arg img image coordinate
/// @arg res resolution coordinate
/// @return object coordinate
#define IMG_TO_OBJ(img,res)       ( ( img * 2 ) / (real)(res - 1) - 1.0 )

using std::vector;
using std::set;
using std::less;
using std::ifstream;
using std::ofstream;
using std::cout;
using std::cerr;
using std::endl;

typedef unsigned int uint;
typedef unsigned char uchar;

typedef float real; ///< MUST BE THE SAME IN MAIN AND KERNEL FILES

/// Pair incident tetrahedra per vertex
typedef struct _incident {
	vector < uint > tetId;
	set < uint, less<uint> > vertId;
} incident;

/*
uint numVerts, numTets;

real *h_vertList;
uint  *h_tetList;

static incident *incidVert; ///< tetrahedra and vertices incident in each vertex

uint *h_conTet, *h_extFaces;

real *h_transferFunction;

uint numExtFaces;

real maxEdgeLength;
*/

extern uint numVerts, numTets;

extern real *h_vertList;
extern uint *h_tetList;
extern uint *h_extFaces;

extern uint *h_conTet;
extern real *h_transferFunction;

extern uint numExtFaces;

extern real h_maxEdgeLength, h_maxZ, h_minZ;

/// ----------------------------------   Prototypes   ------------------------------------

/// --- OFF ---

/// Read OFF (object file format)
/// @arg in input file stream
bool
readOff(ifstream& in);

/// Read OFF (overload)
/// @arg f off file name
/// @return true if it succeed
extern "C" bool
readOff(const char* f);

/// Normalize vertex coordinates
extern "C" void
normalizeVertices(void);

/// --- Incid ---

/// Read Incid (incidents in vertex)
/// @arg in input file stream
bool
readIncid(ifstream& in);

/// Read Incid (overload)
/// @arg f incid file name
/// @return true if it succeed
extern "C" bool
readIncid(const char* f);

/// Build incidTet array
/// @return true if it succeed
extern "C" bool
buildIncid(void);

/// Write Incid (incidents in vertex)
/// @arg out output file stream
bool
writeIncid(ofstream& out);

/// Write Incid (overload)
/// @arg f incid file name
/// @return true if it succeed
extern "C" bool
writeIncid(const char* f);

/// Delete incidTet array
/// @return true if it succeed
extern "C" bool
deleteIncid(void);

/// --- Con ---

/// Read Con (tetrahedra connectivity)
/// @arg in input file stream
bool
readCon(ifstream& in);

/// Read Con (overload)
/// @arg f conTet file name
/// @return true if it succeed
extern "C" bool
readCon(const char* f);

/// Build tetrahedra connectivity
/// @return true if it succeed
extern "C" bool
buildCon(void);

/// Write Con (tetrahedra connectivity)
/// @arg out output file stream
bool
writeCon(ofstream& out);

/// Write Con (overload)
/// @arg f conTet file name
/// @return true if it succeed
extern "C" bool
writeCon(const char* f);

/// --- TF ---

/// Read TF (transfer function)
/// @arg in input file stream
bool
readTF(ifstream& in);

/// Read TF (overload)
/// @arg f transfer function file name
/// @return true if it succeed
extern "C" bool
readTF(const char* f);

/// Build TF (generic TF array)
/// @return true if it succeed
extern "C" bool
buildTF(void);

/// Write TF (transfer function)
/// @arg out output file stream
bool
writeTF(ofstream& out);

/// Write TF (overload)
/// @arg f transfer function file name
/// @return true if it succeed
extern "C" bool
writeTF(const char* f);

/// --- ExtF ---

/// Read ExtF (external faces)
/// @arg in input file stream
bool
readExtF(ifstream& in);

/// Read ExtF (overload)
/// @arg f external faces file name
/// @return true if it succeed
extern "C" bool
readExtF(const char* f);

/// Build External Faces
/// @return true if it succeed
extern "C" bool
buildExtF(void);

/// Write ExtF (external faces)
/// @arg out output file stream
bool
writeExtF(ofstream& out);

/// Write ExtF (overload)
/// @arg f external faces file name
/// @return true if it succeed
extern "C" bool
writeExtF(const char* f);

/// --- Lmt ---

/// Read Lmt (limits)
/// @arg in input file stream
bool
readLmt(ifstream& in);

/// Read Lmt (overload)
/// @arg f limit file name
/// @return true if it succeed
extern "C" bool
readLmt(const char* f);

/// Build Limits
/// @return true if it succeed
extern "C" bool
buildLmt(void);

/// Write Lmt (limits)
/// @arg out output file stream
bool
writeLmt(ofstream& out);

/// Write Lmt (overload)
/// @arg f limits file name
/// @return true if it succeed
extern "C" bool
writeLmt(const char* f);

/// --- Other calls ---

/// Rotate entire volume
/// @arg modelView matrix
extern "C" void
rotateVolume(real *modelView);

/// Write a ppm image
/// @arg fn image file name
/// @arg fb image pixels (frame buffer)
/// @arg dimX x-dimension (width)
/// @arg dimY y-dimension (height)
/// @return true if it succeed
extern "C" bool
ppmWrite(const char *fn, const uchar *fb, uint dimX, uint dimY);

#endif
