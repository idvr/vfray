/**
 *   Visible Faces Ray Casting (VF-Ray)
 *
 * Using C++ -- January, 2008
 *
 */

/**
 *   VF-Ray Kernel
 *
 * CPU code.
 *
 */

#ifndef _VFRAYKERNEL_H_
#define _VFRAYKERNEL_H_

#include "vfRayKernelCPU.h"

#if USEPARTIALPREINT

#include "psiGammaTable512.h"

#endif

/// ----------------------------------   Functions   -------------------------------------

///
/// Face Matrix   *  Face Params   =  Face Target
///   |x_0 y_0 1|   | zsParams.x |   | zs_0 |
///   |x_1 y_1 1|   | zsParams.y |   | zs_1 |
///   |x_2 y_2 1|   | zsParams.z |   | zs_2 |
///
///  -- zs means z or s coordinate of the vertex
///

/// Support functions to map texFetch CUDA functions in C load functions
uint2 uint2Fetch(const uint* v, const uint p) {
	uint2 r;
	r.x = v[ p*2 + 0 ]; r.y = v[ p*2 + 1 ];
	return r;
}
real4 real4Fetch(const real* v, const uint p) {
	real4 r;
	r.x = v[ p*4 + 0 ]; r.y = v[ p*4 + 1 ];
	r.z = v[ p*4 + 2 ]; r.w = v[ p*4 + 3 ];
	return r;
}

/// Kernel 1
void
findVisibleFaces( faceData* visFaces, const real* vertList,
		  const uint* tetList, const uint* extFaces,
		  uint extFaceId ) {

	/// Tetrahedron and face ids from external face
	uint2 tetFaceId = uint2Fetch(extFaces, extFaceId);

	/// Load face vertices
 	real4 fv0 = real4Fetch( vertList, tetList[ tetFaceId.x*4 + ((tetFaceId.y+0)&3) ] );
 	real4 fv1 = real4Fetch( vertList, tetList[ tetFaceId.x*4 + ((tetFaceId.y+1)&3) ] );
 	real4 fv2 = real4Fetch( vertList, tetList[ tetFaceId.x*4 + ((tetFaceId.y+2)&3) ] );

	/// Load opposite face vertex
	real4 opVert = real4Fetch( vertList, tetList[ tetFaceId.x*4 + ((tetFaceId.y+3)&3) ] );

	/// Compute Z interpolation parameters of the external face
	faceData f;
	/// Z and S parameters
	f.zParams.x = 0.0; f.zParams.y = 0.0; f.zParams.z = 0.0;
	f.sParams.x = 0.0; f.sParams.y = 0.0; f.sParams.z = 0.0;

	/// Load face matrix
	real3 fmCol0, fmCol1;
	fmCol0.x = fv0.x; fmCol0.y = fv1.x; fmCol0.z = fv2.x;
	fmCol1.x = fv0.y; fmCol1.y = fv1.y; fmCol1.z = fv2.y;

	/// Compute determinant
	real determinant;
	determinant = fmCol0.x * ( fmCol1.y - fmCol1.z )
		- fmCol1.x * ( fmCol0.y - fmCol0.z )
		+ ( (fmCol0.y * fmCol1.z) - (fmCol1.y * fmCol0.z) );

	//if ( ABS(determinant) < RDELTA ) return;
	determinant = ( ABS(determinant) < RDELTA ) ? RDELTA : determinant;

	/// Load face target and solve linear system
	real3 faceTarget, tmp;

	faceTarget.x = fv0.z; faceTarget.y = fv1.z; faceTarget.z = fv2.z;

	tmp.x = faceTarget.y - faceTarget.z;
	tmp.y = (faceTarget.y * fmCol1.z) - (fmCol1.y * faceTarget.z);
	tmp.z = (fmCol0.y * faceTarget.z) - (faceTarget.y * fmCol0.z);

	/// Compute z parameters
	f.zParams.x = ( faceTarget.x * ( fmCol1.y - (fmCol1.z) )
			- (fmCol1.x * tmp.x) + tmp.y ) / determinant;

	f.zParams.y = ( (fmCol0.x * tmp.x) - faceTarget.x
			* ( fmCol0.y - (fmCol0.z) )
			+ tmp.z ) / determinant;

	f.zParams.z = ( fmCol0.x * (-tmp.y) - (fmCol1.x * tmp.z) + faceTarget.x
			* (fmCol0.y * fmCol1.z - fmCol1.y * fmCol0.z) ) / determinant;

	faceTarget.x = fv0.w; faceTarget.y = fv1.w; faceTarget.z = fv2.w;

	tmp.x = faceTarget.y - faceTarget.z;
	tmp.y = (faceTarget.y * fmCol1.z) - (fmCol1.y * faceTarget.z);
	tmp.z = (fmCol0.y * faceTarget.z) - (faceTarget.y * fmCol0.z);

	/// Compute s parameters
	f.sParams.x = ( faceTarget.x * ( fmCol1.y - (fmCol1.z) )
			- (fmCol1.x * tmp.x) + tmp.y ) / determinant;

	f.sParams.y = ( (fmCol0.x * tmp.x) - faceTarget.x
			* ( fmCol0.y - (fmCol0.z) )
			+ tmp.z ) / determinant;

	f.sParams.z = ( fmCol0.x * (-tmp.y) - (fmCol1.x * tmp.z) + faceTarget.x
			* (fmCol0.y * fmCol1.z - fmCol1.y * fmCol0.z) ) / determinant;

	/// Z coordinate of opposite vertex projeted on the external face
	real faceZ = opVert.x * f.zParams.x + opVert.y * f.zParams.y + f.zParams.z;

	/// Test if face is not visible
	if ( faceZ < opVert.z ) f.sParams.x = -1.0;

	visFaces[ extFaceId ] = f;

}

/// Kernel 2
void
projectVisibleFaces( projectedFace* projVisFaces, const real* vertList,
		     const uint* tetList, const uint* extFaces,
		     const faceData* visFaces, uint extFaceId,
		     uint width, uint height ) {

	/// Create a null bounding box
	projVisFaces[ extFaceId ].bBox.x = width-1;
	projVisFaces[ extFaceId ].bBox.y = height-1;
	projVisFaces[ extFaceId ].bBox.z = 0;
	projVisFaces[ extFaceId ].bBox.w = 0;

	/// Tetrahedron and face ids from external face
	uint2 tetFaceId = uint2Fetch(extFaces, extFaceId);

	/// Build bounding box looping in each face vertex
	uint2 img;
	real4 faceVert;

	for (uint i = 0; i < 3; ++i) { /// for each visible face vertex

		faceVert = real4Fetch( vertList, tetList[ tetFaceId.x*4 + ((tetFaceId.y+i)&3) ] );

		img.x = ((uint)(( width  - 1.0 ) * ( faceVert.x + 1.0 ) / 2.0) );
		img.y = ((uint)(( height - 1.0 ) * ( faceVert.y + 1.0 ) / 2.0) );

		/// Make bounding box
		if (img.x < projVisFaces[ extFaceId ].bBox.x) projVisFaces[ extFaceId ].bBox.x = img.x;
		if (img.y < projVisFaces[ extFaceId ].bBox.y) projVisFaces[ extFaceId ].bBox.y = img.y;
		if (img.x > projVisFaces[ extFaceId ].bBox.z) projVisFaces[ extFaceId ].bBox.z = img.x;
		if (img.y > projVisFaces[ extFaceId ].bBox.w) projVisFaces[ extFaceId ].bBox.w = img.y;

	} // i

}

/// Kernel 3
void
vfRay( uchar* finalImage, const real* vertList,
       const uint* tetList, const uint* extFaces,
       const faceData* visFaces, const projectedFace* projVisFaces,
       const uint* conTet, const real* transferFunction,
       uint extFaceId, uint width, uint height,
       real maxEdgeLength, real maxZ, real minZ ) {

	/// Retrieve current face in visible faces array
	faceData currFace = visFaces[ extFaceId ];
			      
	/// Tetrahedron and face ids from external face
	uint2 tfPrev = uint2Fetch(extFaces, extFaceId);

	/// Retrieve projected visible face bounding box
	uint4 boundingBox = projVisFaces[ extFaceId ].bBox;

	/// Retrieve projected visible face vertices
 	real4 fv0 = real4Fetch( vertList, tetList[ tfPrev.x*4 + ((tfPrev.y+0)&3) ] );
 	real4 fv1 = real4Fetch( vertList, tetList[ tfPrev.x*4 + ((tfPrev.y+1)&3) ] );
 	real4 fv2 = real4Fetch( vertList, tetList[ tfPrev.x*4 + ((tfPrev.y+2)&3) ] );

	/// Compute projected visible face vectors
	real2 vf_01, vf_02;
	vf_01.x = fv1.x - fv0.x; vf_01.y = fv1.y - fv0.y;
	vf_02.x = fv2.x - fv0.x; vf_02.y = fv2.y - fv0.y;

	/// Compute visible face crosses to be used in inside face test
	real3 vf_crosses;
	vf_crosses.x = ( vf_01.x * vf_02.y ) - ( vf_01.y * vf_02.x );

	/// Verify if visible face is co-planar
	//if (ABS(vf_crosses.x) < RDELTA) return;
	vf_crosses.x = ( ABS(vf_crosses.x) < RDELTA ) ? RDELTA : vf_crosses.x;

	real2 tmp;
	if (vf_crosses.x < 0.0) { /// Swap vectors

		tmp.x = vf_01.x; tmp.y = vf_01.y;
		vf_01.x = vf_02.x; vf_01.y = vf_02.y;
		vf_02.x = tmp.x; vf_02.y = tmp.y;
		vf_crosses.x = - vf_crosses.x;

	}

	real4 vf_0p;
	real2 obj;
	uint2 pix;

	real2 zsPrev;
	real2 zsNext;
	uint2 tfNext;

	faceData nextFace;

	uint faceId, i;

	real3 nf_crosses;

	real dz;

#define MAXBUFFER 512

	faceData vfRayBuffer[MAXBUFFER]; /// MR = 100 * 6 * 8B ~ 4800 B

	uint2 vfRayIds[MAXBUFFER]; /// MR = 100 * 2 * 4B ~ 800 B

	for (i=0; i<MAXBUFFER; ++i) vfRayIds[i].y = 4;

	uint bufId;

	// Local array by centroid Z
	real stepZ = (maxZ - minZ) / (real)MAXBUFFER;
	real centroidZ;

	for (pix.x = boundingBox.x; pix.x <= boundingBox.z; ++pix.x) {

		obj.x = ( ( pix.x * 2 ) / (real)(width - 1) - 1.0 );

		for (pix.y = boundingBox.y; pix.y <= boundingBox.w; ++pix.y) {

			obj.y = ( ( pix.y * 2 ) / (real)(height - 1) - 1.0 );

			tfPrev = uint2Fetch(extFaces, extFaceId);

 			vf_0p = real4Fetch( vertList, tetList[ tfPrev.x*4 + ((tfPrev.y+0)&3) ] );

			vf_0p.x = obj.x - vf_0p.x;
			vf_0p.y = obj.y - vf_0p.y;

			vf_crosses.y = ( vf_01.x * vf_0p.y ) - ( vf_01.y * vf_0p.x );

			vf_crosses.z = ( vf_0p.x * vf_02.y ) - ( vf_0p.y * vf_02.x );

			if ( (vf_crosses.y >= 0.0) && (vf_crosses.z >= 0.0) &&
			     ((vf_crosses.y + vf_crosses.z) <= vf_crosses.x) ) {

				/// Re-start initial values
				zsPrev.x = obj.x * currFace.zParams.x + obj.y * currFace.zParams.y + currFace.zParams.z;
				zsPrev.y = obj.x * currFace.sParams.x + obj.y * currFace.sParams.y + currFace.sParams.z;

				if ( (zsPrev.y < 0.0) || (zsPrev.y > 1.0) ) break;

#if USEPARTIALPREINT

				real4 colorFront;
				real4 colorBack = real4Fetch(transferFunction, ((uint)(255*zsPrev.y)) );
				real3 accumColor;
				accumColor.x = 0.0; accumColor.y = 0.0; accumColor.z = 0.0;

#else

				uchar3 color;
				color.x = 0; color.y = 0; color.z = 0;

				uint3 rgb;
				rgb.x = 0; rgb.y = 0; rgb.z = 0;

				real4 tfColor = real4Fetch(transferFunction, ((uint)(255*zsPrev.y)) );
				real4 coPrev, coNext;
				coPrev.x = 255 * tfColor.x;
				coPrev.y = 255 * tfColor.y;
				coPrev.z = 255 * tfColor.z;

				coPrev.w = tfColor.w;
				real opacity = 0.0;
#endif

				while (1) {

					/// Set no-next face initially
					tfNext.y = tfPrev.y;

					for (i = 1; i < 4; i++) { // for each other face

						faceId = (tfPrev.y+i)&3;

						fv0 = real4Fetch(vertList, tetList[ tfPrev.x*4 + ((faceId+0)&3) ] );
						fv1 = real4Fetch(vertList, tetList[ tfPrev.x*4 + ((faceId+1)&3) ] );
						fv2 = real4Fetch(vertList, tetList[ tfPrev.x*4 + ((faceId+2)&3) ] );

						fv1.x -= fv0.x; fv1.y -= fv0.y;
						fv2.x -= fv0.x; fv2.y -= fv0.y;

						nf_crosses.x = ( fv1.x * fv2.y ) - ( fv1.y * fv2.x );

						if (ABS(nf_crosses.x) < RDELTA)
							continue;

						if (nf_crosses.x < 0.0) { /// Swap vectors

							tmp.x = fv1.x; tmp.y = fv1.y;
							fv1.x = fv2.x; fv1.y = fv2.y;
							fv2.x = tmp.x; fv2.y = tmp.y;
							nf_crosses.x = -nf_crosses.x;

						}

						fv0.x = obj.x - fv0.x;
						fv0.y = obj.y - fv0.y;

						nf_crosses.y = ( fv1.x * fv0.y ) - ( fv1.y * fv0.x );
						nf_crosses.z = ( fv0.x * fv2.y ) - ( fv0.y * fv2.x );

						if ( (nf_crosses.y >= 0.0) && (nf_crosses.z >= 0.0)
						     && ((nf_crosses.y + nf_crosses.z) <= nf_crosses.x + RDELTA) ) {

							tfNext.y = faceId;

							break;

						}

					}

					/// Anomally finished ray casting
					if ( tfNext.y == tfPrev.y ) break;

					/// Load next tetrahedron
					tfNext.x = conTet[ tfPrev.x * 4 + tfNext.y ];

					/// If ray didn't exit the volume
					if ( tfNext.x != tfPrev.x ) {

						if ( conTet[ tfNext.x * 4 + 0 ] == tfPrev.x ) tfPrev.y = 0;
						else if ( conTet[ tfNext.x * 4 + 1 ] == tfPrev.x ) tfPrev.y = 1;
						else if ( conTet[ tfNext.x * 4 + 2 ] == tfPrev.x ) tfPrev.y = 2;
						else if ( conTet[ tfNext.x * 4 + 3 ] == tfPrev.x ) tfPrev.y = 3;
						else return;

					}

					// Buffer indexing by centroidZ -- O(1) search
					fv0 = real4Fetch(vertList, tetList[ tfPrev.x*4 + ((tfNext.y+0)&3) ] );
					fv1 = real4Fetch(vertList, tetList[ tfPrev.x*4 + ((tfNext.y+1)&3) ] );
					fv2 = real4Fetch(vertList, tetList[ tfPrev.x*4 + ((tfNext.y+2)&3) ] );

					centroidZ = (fv0.z + fv1.z + fv2.z) / 3.0;

					bufId = (uint) ( (centroidZ - minZ) / stepZ );

					if ( (vfRayIds[ bufId ].y != 4) && (vfRayIds[ bufId ].x == tfPrev.x) && (vfRayIds[ bufId ].y == tfNext.y) ) {

						nextFace = vfRayBuffer[ bufId ];

					} else {

						/// Load face matrix
						real3 fmCol0, fmCol1;
						fmCol0.x = fv0.x; fmCol0.y = fv1.x; fmCol0.z = fv2.x;
						fmCol1.x = fv0.y; fmCol1.y = fv1.y; fmCol1.z = fv2.y;

						/// Compute determinant
						real determinant;
						determinant = fmCol0.x * ( fmCol1.y - fmCol1.z )
							- fmCol1.x * ( fmCol0.y - fmCol0.z )
							+ ( (fmCol0.y * fmCol1.z) - (fmCol1.y * fmCol0.z) );

						//if ( ABS(determinant) < RDELTA ) return;
						determinant = ( ABS(determinant) < RDELTA ) ? RDELTA : determinant;

						/// Load face target and solve linear system
						real3 faceTarget, tmp;

						faceTarget.x = fv0.z; faceTarget.y = fv1.z; faceTarget.z = fv2.z;

						tmp.x = faceTarget.y - faceTarget.z;
						tmp.y = (faceTarget.y * fmCol1.z) - (fmCol1.y * faceTarget.z);
						tmp.z = (fmCol0.y * faceTarget.z) - (faceTarget.y * fmCol0.z);

						/// Compute z parameters
						nextFace.zParams.x = ( faceTarget.x * ( fmCol1.y - (fmCol1.z) )
								       - (fmCol1.x * tmp.x) + tmp.y ) / determinant;

						nextFace.zParams.y = ( (fmCol0.x * tmp.x) - faceTarget.x
								       * ( fmCol0.y - (fmCol0.z) )
								       + tmp.z ) / determinant;

						nextFace.zParams.z = ( fmCol0.x * (-tmp.y) - (fmCol1.x * tmp.z) + faceTarget.x
								       * (fmCol0.y * fmCol1.z - fmCol1.y * fmCol0.z) ) / determinant;

						faceTarget.x = fv0.w; faceTarget.y = fv1.w; faceTarget.z = fv2.w;

						tmp.x = faceTarget.y - faceTarget.z;
						tmp.y = (faceTarget.y * fmCol1.z) - (fmCol1.y * faceTarget.z);
						tmp.z = (fmCol0.y * faceTarget.z) - (faceTarget.y * fmCol0.z);

						/// Compute s parameters
						nextFace.sParams.x = ( faceTarget.x * ( fmCol1.y - (fmCol1.z) )
								       - (fmCol1.x * tmp.x) + tmp.y ) / determinant;

						nextFace.sParams.y = ( (fmCol0.x * tmp.x) - faceTarget.x
								       * ( fmCol0.y - (fmCol0.z) )
								       + tmp.z ) / determinant;

						nextFace.sParams.z = ( fmCol0.x * (-tmp.y) - (fmCol1.x * tmp.z) + faceTarget.x
								       * (fmCol0.y * fmCol1.z - fmCol1.y * fmCol0.z) ) / determinant;

						vfRayBuffer[ bufId ] = nextFace;
						vfRayIds[ bufId ].x = tfPrev.x;
						vfRayIds[ bufId ].y = tfNext.y;

					}

					/// Compute zNext and sNext
					zsNext.x = obj.x * nextFace.zParams.x + obj.y * nextFace.zParams.y + nextFace.zParams.z;
					zsNext.y = obj.x * nextFace.sParams.x + obj.y * nextFace.sParams.y + nextFace.sParams.z;

					/// Compute ray integration
					dz = ABS(zsPrev.x - zsNext.x);

					dz = (dz < RDELTA) ? 0.0 : dz;

					if ( (zsNext.y < 0.0) || (zsNext.y > 1.0) ) break;

#if USEPARTIALPREINT

					dz /= maxEdgeLength;

					colorFront = real4Fetch(transferFunction, ((uint)(255*zsNext.y)) );

					real2 tau;
					tau.x = colorFront.w * dz;
					tau.y = colorBack.w * dz;

					real zeta = exp( -(tau.x * 0.5 + tau.y * 0.5) );

					real2 gamma;
					gamma.x = tau.x / (1.0 + tau.x);
					gamma.y = tau.y / (1.0 + tau.y);

					real2 texCoord;
					texCoord.x = gamma.x + (0.5 / 512);
					texCoord.y = gamma.y + (0.5 / 512);

					if ( (texCoord.x < 0.0) || (texCoord.x > 1.0) || (texCoord.y < 0.0) || (texCoord.y > 1.0) ) break;

					real psi = psiGammaTable[ (uint)(texCoord.x*512) ][ (uint)(texCoord.y*512) ];

					real4 finalColor;
					finalColor.x = colorFront.x*(1.0 - psi) + colorBack.x*(psi - zeta);
					finalColor.y = colorFront.y*(1.0 - psi) + colorBack.y*(psi - zeta);
					finalColor.z = colorFront.z*(1.0 - psi) + colorBack.z*(psi - zeta);
					finalColor.w = 1.0 - zeta;

					accumColor.x = finalColor.x + (1.0 - finalColor.w) * accumColor.x;
					accumColor.y = finalColor.y + (1.0 - finalColor.w) * accumColor.y;
					accumColor.z = finalColor.z + (1.0 - finalColor.w) * accumColor.z;

					colorBack = colorFront;

#else
					
					real transparency;

					/// Load transfer function color
					tfColor = real4Fetch(transferFunction, ((uint)(255*zsNext.y)) );

					coNext.x = 255 * tfColor.x;
					coNext.y = 255 * tfColor.y;
					coNext.z = 255 * tfColor.z;

					coNext.w = tfColor.w;

					if ( (coNext.w < 0.0) || (coNext.w > 1.0) ) return;

 					transparency = 1.0 - opacity;

					rgb.x = (uint)( color.x + ( 0.5 * (coPrev.x + coNext.x) * transparency * dz
								    - 0.04166666 * (3.0 * coPrev.x * coPrev.w + 5.0 * coNext.x * coPrev.w
										    + coPrev.x * coNext.w + 3.0 * coNext.x * coNext.w)
								    * dz * dz ) );
					
					rgb.y = (uint)( color.y + ( 0.5 * (coPrev.y + coNext.y) * transparency * dz
								    - 0.04166666 * (3.0 * coPrev.y * coPrev.w + 5.0 * coNext.y * coPrev.w
										    + coPrev.y * coNext.w + 3.0 * coNext.y * coNext.w)
								    * dz * dz ) );

					rgb.z = (uint)( color.z + ( 0.5 * (coPrev.z + coNext.z) * transparency * dz
								    - 0.04166666 * (3.0 * coPrev.z * coPrev.w + 5.0 * coNext.z * coPrev.w
										    + coPrev.z * coNext.w + 3.0 * coNext.z * coNext.w)
								    * dz * dz ) );

					if ( coPrev.w < 0.0 ) return;

					if ( coNext.w < 0.0 ) return;

					opacity += (coPrev.w + coNext.w) * dz / 2.0;

					if ( opacity < 0.0 ) return;

					/// Early ray termination
					if (opacity > 1.0) {

						opacity = 1.0;
						break;

					}

#endif

					/// If ray exits the volume
					if ( tfNext.x == tfPrev.x ) break;

					/// Pass previous information to the next ray casting step
					tfPrev.x = tfNext.x;
					zsPrev.x = zsNext.x;
					zsPrev.y = zsNext.y;

				} // while

#if USEPARTIALPREINT

				if (accumColor.x > 1.0) accumColor.x = 1.0;
				if (accumColor.y > 1.0) accumColor.y = 1.0;
				if (accumColor.z > 1.0) accumColor.z = 1.0;

				finalImage[ (pix.y * width + pix.x)*3 + 0 ] = (uchar)(255 * accumColor.x);
				finalImage[ (pix.y * width + pix.x)*3 + 1 ] = (uchar)(255 * accumColor.y);
				finalImage[ (pix.y * width + pix.x)*3 + 2 ] = (uchar)(255 * accumColor.z);
#else

 				finalImage[ (pix.y * width + pix.x)*3 + 0 ] = (uchar)(color.x * opacity);
 				finalImage[ (pix.y * width + pix.x)*3 + 1 ] = (uchar)(color.y * opacity);
				finalImage[ (pix.y * width + pix.x)*3 + 2 ] = (uchar)(color.z * opacity);

#endif
			} // if

		} // yP

	} // xP

}

#endif
