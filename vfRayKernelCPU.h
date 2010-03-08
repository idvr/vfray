/**
 *   Visible Faces Ray Casting (VF-Ray)
 *
 * In CPU -- January, 2008
 *
 */

/**
 *   VF-Ray Kernel
 *
 * C code.
 *
 */

/// ----------------------------------   Definitions   ------------------------------------

#ifndef _VFRAYKERNELCPU_H_
#define _VFRAYKERNELCPU_H_

#include <cassert>
#include <cmath>
#include <cstdio>

#define USEPARTIALPREINT 1

#define ABS(n) ((n) < 0 ? -(n) : (n)) ///< absolute value

#define RDELTA 1e-12  ///< |R delta

typedef unsigned char uchar;
typedef unsigned int uint;

typedef float real;

/// Basic types

typedef struct _uchar2 { uchar x, y; } uchar2;
typedef struct _uchar3 { uchar x, y, z; } uchar3;
typedef struct _uchar4 { uchar x, y, z, w; } uchar4;

typedef struct _uint2 { uint x, y; } uint2;
typedef struct _uint3 {	uint x, y, z; } uint3;
typedef struct _uint4 { uint x, y, z, w; } uint4;

typedef struct _real2 { real x, y; } real2;
typedef struct _real3 { real x, y, z; } real3;
typedef struct _real4 { real x, y, z, w; } real4;

/// Face types

typedef struct _faceData {
	real3 zParams; ///< x,y,z: z parameters
	real3 sParams; ///< x,y,z: s parameters
} faceData;

typedef struct _projectedFace {
	uint4 bBox; ///< Bounding Box ; x,y: min ; z,w: max
} projectedFace;

/// Kernel 1
void
findVisibleFaces( faceData* visFaces, const real* vertList,
		  const uint* tetList, const uint* extFaces,
		  uint extFaceId );

/// Kernel 2
void
projectVisibleFaces( projectedFace* projVisFaces, const real* vertList,
		     const uint* tetList, const uint* extFaces,
		     const faceData* visFaces, uint extFaceId,
		     uint width, uint height );

/// Kernel 3
void
vfRay( uchar* finalImage, const real* vertList,
       const uint* tetList, const uint* extFaces,
       const faceData* visFaces, const projectedFace* projVisFaces,
       const uint* conTet, const real* transferFunction,
       uint extFaceId, uint width, uint height,
       real maxEdgeLength, real maxZ, real minZ );

#endif
