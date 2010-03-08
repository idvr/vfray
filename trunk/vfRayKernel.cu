/**
 *   Visible Faces Ray Casting (VF-Ray)
 *
 * Using CUDA -- January, 2008
 *
 */

/**
 *   VF-Ray Kernel
 *
 * Device code.
 *
 */

#ifndef _VFRAYKERNEL_H_
#define _VFRAYKERNEL_H_

#define ABS(n) ((n) < 0 ? -(n) : (n)) ///< absolute value

#define RDELTA 1e-12  ///< |R delta

typedef unsigned char uchar;
typedef unsigned int uint;

typedef float real;

typedef float2 real2;

typedef float3 real3;

typedef float4 real4;

/// Face types

typedef struct _faceData {
	real3 zParams; ///< x,y,z: z parameters
	real3 sParams; ///< x,y,z: s parameters
} faceData;

typedef struct _projectedFace {
	uint4 bBox; ///< Bounding Box ; x,y: min ; z,w: max
} projectedFace;

/// Declare 1D textures reference
texture<real4, 1, cudaReadModeElementType> vertListTex;
//texture<uint4, 1, cudaReadModeElementType> tetListTex;
///-- tetListTex needs branches (worst approach) to be accessed
texture<uint2, 1, cudaReadModeElementType> extFacesTex;
#if USECONTETTEX
texture<uint4, 1, cudaReadModeElementType> conTetTex;
#endif
texture<real4, 1, cudaReadModeElementType> transferFunctionTex;

#if USEPARTIALPREINT
texture<real, 2> psiGammaTableTex;
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

/// Kernel 1
__global__ void
findVisibleFaces( faceData* visFaces,
		  const uint* tetList,
		  const uint numExtFaces,
		  const uint gridSize ) {

	/// Convert 2D-block and 2D-thread array id in 1D-extFace array id
// 	uint blockId = __mul24( __mul24(blockIdx.y, gridDim.x)
// 				+ blockIdx.x, blockDim.x ) + threadIdx.y;
// 	uint extFaceId = __mul24(blockId, blockDim.y) + threadIdx.x;
	uint extFaceId = blockIdx.y * BLOCK_SIZE * BLOCK_SIZE * gridSize
		+ blockIdx.x * BLOCK_SIZE * BLOCK_SIZE
		+ threadIdx.y * BLOCK_SIZE
		+ threadIdx.x;

	/// The thread is useless if its id is out of bound
	if (extFaceId >= numExtFaces) {

		extFaceId = numExtFaces - 1;

	}

	/// Tetrahedron and face ids from external face
	uint2 tetFaceId = tex1Dfetch(extFacesTex, extFaceId);

	/// Load face vertices
 	real4 fv0 = tex1Dfetch(vertListTex, tetList[ tetFaceId.x*4 + ((tetFaceId.y+0)&3) ] );
 	real4 fv1 = tex1Dfetch(vertListTex, tetList[ tetFaceId.x*4 + ((tetFaceId.y+1)&3) ] );
 	real4 fv2 = tex1Dfetch(vertListTex, tetList[ tetFaceId.x*4 + ((tetFaceId.y+2)&3) ] );

	/// Load opposite face vertex
	real4 opVert = tex1Dfetch(vertListTex, tetList[ tetFaceId.x*4 + ((tetFaceId.y+3)&3) ]);

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
__global__ void
projectVisibleFaces( projectedFace* projVisFaces,
		     const uint* tetList,
		     const faceData* visFaces,
		     const uint numExtFaces,
		     const uint gridSize,
		     const uint width,
		     const uint height ) {

	/// Convert 2D-block and 2D-thread array id in 1D-visFace array id
	uint extFaceId = blockIdx.y * BLOCK_SIZE * BLOCK_SIZE * gridSize
		+ blockIdx.x * BLOCK_SIZE * BLOCK_SIZE
		+ threadIdx.y * BLOCK_SIZE
		+ threadIdx.x;

	/// The thread is useless if its id is out of bound
	if (extFaceId >= numExtFaces) {

		extFaceId = numExtFaces - 1;

	}

	/// Create a null bounding box
	projVisFaces[ extFaceId ].bBox.x = width-1;
	projVisFaces[ extFaceId ].bBox.y = height-1;
	projVisFaces[ extFaceId ].bBox.z = 0;
	projVisFaces[ extFaceId ].bBox.w = 0;

#if REDUCTION
	/// Avoid branch
#else
	/// External face not visible
	if (visFaces[ extFaceId ].sParams.x == -1.0) return;
#endif

	/// Tetrahedron and face ids from external face
	uint2 tetFaceId = tex1Dfetch(extFacesTex, extFaceId);

	/// Build bounding box looping in each face vertex
	uint2 img;
	real4 faceVert;

	for (uint i = 0; i < 3; ++i) { /// for each visible face vertex

		faceVert = tex1Dfetch(vertListTex, tetList[ tetFaceId.x * 4 + ((tetFaceId.y + i)&3) ] );

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
__global__ void
vfRay( uchar* finalImage,
       const uint* tetList,
#if USECONTETTEX
#else
       const uint* conTet,
#endif
       const faceData* visFaces,
       const projectedFace* projVisFaces,
       const uint numExtFaces,
       const uint gridSize,
       const uint width,
       const uint height
#if USEVFRAYBUFFER
       ,const real maxZ
       ,const real minZ
#endif
#if USEPARTIALPREINT
       ,const real maxEdgeLength
#endif
	) {

#if USEBBOXTILES

	/// Each block running one visible face, with the projected
	/// bounding box divided for each thread
	uint extFaceId = blockIdx.y * gridSize + blockIdx.x;

#else

	/// Convert 2D-block and 2D-thread array id in 1D-visFace array id
	uint extFaceId = blockIdx.y * BLOCK_SIZE * BLOCK_SIZE * gridSize
		+ blockIdx.x * BLOCK_SIZE * BLOCK_SIZE
		+ threadIdx.y * BLOCK_SIZE
		+ threadIdx.x;

#endif

	/// The thread is useless if its id is out of bound
	if (extFaceId >= numExtFaces) {

		extFaceId = numExtFaces - 1;

	}

	/// Retrieve current face in visible faces array
	faceData currFace = visFaces[ extFaceId ];

#if REDUCTION
	/// Avoid branch
#else
	/// External face not visible
	if (currFace.sParams.x == -1.0) return;
#endif

	/// Tetrahedron and face ids from external face
	uint2 tfPrev = tex1Dfetch(extFacesTex, extFaceId );

	/// Retrieve projected visible face bounding box
	uint4 boundingBox = projVisFaces[ extFaceId ].bBox;

	/// Retrieve projected visible face vertices
	real4 fv0 = tex1Dfetch(vertListTex, tetList[ tfPrev.x*4 + ((tfPrev.y+0)&3) ] );
	real4 fv1 = tex1Dfetch(vertListTex, tetList[ tfPrev.x*4 + ((tfPrev.y+1)&3) ] );
	real4 fv2 = tex1Dfetch(vertListTex, tetList[ tfPrev.x*4 + ((tfPrev.y+2)&3) ] );

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

#if USEOPACITY
	real opacity;
#endif

	uint faceId, i;

	real3 nf_crosses;

#if USEBBOXTILES

	/// Compute Tiled Bounding Box for each thread in a block

	uint2 bbInit, bbEnd;

	uint2 bbSize;

	bbSize.x = (boundingBox.z+1 - boundingBox.x);
	bbSize.y = (boundingBox.w+1 - boundingBox.y);

	uint2 bbStep;

	bbStep.x = bbSize.x / BLOCK_SIZE;
	bbStep.y = bbSize.y / BLOCK_SIZE;

	uint2 modBB;

	modBB.x = bbSize.x % BLOCK_SIZE;
	modBB.y = bbSize.y % BLOCK_SIZE;

	bbInit.x = boundingBox.x;
	uint tid = 0;
	while ( tid < threadIdx.x) {
		++tid;
		bbInit.x += bbStep.x;
		if ( tid < modBB.x ) bbInit.x += 1;
	}

	bbInit.y = boundingBox.y;
	tid = 0;
	while ( tid < threadIdx.y) {
		++tid;
		bbInit.y += bbStep.y;
		if ( tid < modBB.y ) bbInit.y += 1;
	}

	// +1 to get ceil of tile bb (bounding box)
	if ( threadIdx.x < modBB.x ) bbStep.x += 1;
	if ( threadIdx.y < modBB.y ) bbStep.y += 1;

	if ( bbStep.x == 0 ) return;
	if ( bbStep.y == 0 ) return;

	bbEnd.x = bbInit.x + bbStep.x;
	bbEnd.y = bbInit.y + bbStep.y;

	// +1 to include both bb (bounding box) boundaries
	if ( threadIdx.x == BLOCK_SIZE-1 ) bbEnd.x += 1;
	if ( threadIdx.y == BLOCK_SIZE-1 ) bbEnd.y += 1;

#endif

#if USEVFRAYBUFFER

#define MAXBUFFER 100

//	__syncthreads();

	/// VF-Ray Buffer
//	__shared__
		faceData vfRayBuffer[MAXBUFFER]; /// MR = 100 * 6 * 8B ~ 4800 B

//	__shared__
		uint2 vfRayIds[MAXBUFFER]; /// MR = 100 * 2 * 4B ~ 800 B

	for (i=0; i<MAXBUFFER; ++i) vfRayIds[i].y = 4;

	uint bufId;

	// Local array by centroid Z
	real stepZ = (maxZ - minZ) / (real)MAXBUFFER;
	real centroidZ;

#endif

	/// Iterates in all pixels inside the bounding box

#if USEBBOXTILES
	for (pix.x = bbInit.x; pix.x < bbEnd.x; ++pix.x) {
#else
	for (pix.x = boundingBox.x; pix.x <= boundingBox.z; ++pix.x) {
#endif
		obj.x = ( ( pix.x * 2 ) / (real)(width - 1) - 1.0 );

#if USEBBOXTILES
		for (pix.y = bbInit.y; pix.y < bbEnd.y; ++pix.y) {
#else
		for (pix.y = boundingBox.y; pix.y <= boundingBox.w; ++pix.y) {
#endif

			obj.y = ( ( pix.y * 2 ) / (real)(height - 1) - 1.0 );

			tfPrev = tex1Dfetch(extFacesTex, extFaceId );

 			vf_0p = tex1Dfetch(vertListTex, tetList[ tfPrev.x*4 + ((tfPrev.y+0)&3) ] );

			vf_0p.x = obj.x - vf_0p.x;
			vf_0p.y = obj.y - vf_0p.y;

			vf_crosses.y = ( vf_01.x * vf_0p.y ) - ( vf_01.y * vf_0p.x );

			vf_crosses.z = ( vf_0p.x * vf_02.y ) - ( vf_0p.y * vf_02.x );

			if ( (vf_crosses.y >= 0.0) && (vf_crosses.z >= 0.0) &&
			     ((vf_crosses.y + vf_crosses.z) <= vf_crosses.x) ) {

				/// Re-start initial values
				zsPrev.x = obj.x * currFace.zParams.x + obj.y * currFace.zParams.y + currFace.zParams.z;
				zsPrev.y = obj.x * currFace.sParams.x + obj.y * currFace.sParams.y + currFace.sParams.z;

#if USEPARTIALPREINT

				real4 colorFront;
				real4 colorBack = tex1Dfetch(transferFunctionTex, ((uint)(255*zsPrev.y)) );
				real3 accumColor;
				accumColor.x = 0.0; accumColor.y = 0.0; accumColor.z = 0.0;

#else

				uchar3 color = make_uchar3(0, 0, 0);
				uint3 rgb = make_uint3(0, 0, 0);

				real4 tfColor = tex1Dfetch(transferFunctionTex, ((uint)(255*zsPrev.y)) );
				real4 coPrev, coNext;
				coPrev.x = 255 * tfColor.x;
				coPrev.y = 255 * tfColor.y;
				coPrev.z = 255 * tfColor.z;
#if USEOPACITY
				coPrev.w = tfColor.w;
				opacity = 0.0;
#endif
#endif

				while (1) {

					/// Set no-next face initially
					tfNext.y = tfPrev.y;

					for (i = 1; i < 4; i++) { // for each other face

						faceId = (tfPrev.y+i)&3;

						fv0 = tex1Dfetch(vertListTex, tetList[ tfPrev.x*4 + ((faceId+0)&3) ] );
						fv1 = tex1Dfetch(vertListTex, tetList[ tfPrev.x*4 + ((faceId+1)&3) ] );
						fv2 = tex1Dfetch(vertListTex, tetList[ tfPrev.x*4 + ((faceId+2)&3) ] );

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
#if USECONTETTEX
					uint4 conTetId;
					conTetId = tex1Dfetch(conTetTex, tfPrev.x);
					if (tfNext.y == 0) tfNext.x = conTetId.x;
					else if (tfNext.y == 1) tfNext.x = conTetId.y;
					else if (tfNext.y == 2) tfNext.x = conTetId.z;
					else if (tfNext.y == 3) tfNext.x = conTetId.w;
#else
					tfNext.x = conTet[ tfPrev.x * 4 + tfNext.y ];
#endif

					/// If ray didn't exit the volume
					if ( tfNext.x != tfPrev.x ) {

						/// Find which face connects both tets
#if USECONTETTEX
						conTetId = tex1Dfetch(conTetTex, tfNext.x );
						if( conTetId.x == tfPrev.x ) tfPrev.y = 0;
						else if( conTetId.y == tfPrev.x ) tfPrev.y = 1;
						else if( conTetId.z == tfPrev.x ) tfPrev.y = 2;
						else if( conTetId.w == tfPrev.x ) tfPrev.y = 3;
						else return;
#else
						if ( conTet[ tfNext.x * 4 + 0 ] == tfPrev.x ) tfPrev.y = 0;
						else if ( conTet[ tfNext.x * 4 + 1 ] == tfPrev.x ) tfPrev.y = 1;
						else if ( conTet[ tfNext.x * 4 + 2 ] == tfPrev.x ) tfPrev.y = 2;
						else if ( conTet[ tfNext.x * 4 + 3 ] == tfPrev.x ) tfPrev.y = 3;
						else return;
#endif

					}

#if USEVFRAYBUFFER

					// Buffer indexing by centroidZ -- O(1) search
					fv0 = tex1Dfetch(vertListTex, tetList[ tfPrev.x*4 + ((tfNext.y+0)&3) ] );
					fv1 = tex1Dfetch(vertListTex, tetList[ tfPrev.x*4 + ((tfNext.y+1)&3) ] );
					fv2 = tex1Dfetch(vertListTex, tetList[ tfPrev.x*4 + ((tfNext.y+2)&3) ] );

					centroidZ = (fv0.z + fv1.z + fv2.z) / 3.0;

					bufId = (uint) ( (centroidZ - minZ) / stepZ );

					if ( (vfRayIds[ bufId ].y != 4) &&
					     ( ((vfRayIds[ bufId ].x == tfPrev.x) &&
						(vfRayIds[ bufId ].y == tfNext.y)) ) ) {

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

#else

					/// Load face vertices
					fv0 = tex1Dfetch(vertListTex, tetList[ tfPrev.x*4 + ((tfNext.y+0)&3) ] );
					fv1 = tex1Dfetch(vertListTex, tetList[ tfPrev.x*4 + ((tfNext.y+1)&3) ] );
					fv2 = tex1Dfetch(vertListTex, tetList[ tfPrev.x*4 + ((tfNext.y+2)&3) ] );

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

#endif
					/// Compute zNext and sNext
					zsNext.x = obj.x * nextFace.zParams.x + obj.y * nextFace.zParams.y + nextFace.zParams.z;
					zsNext.y = obj.x * nextFace.sParams.x + obj.y * nextFace.sParams.y + nextFace.sParams.z;

					/// Compute ray integration
					real dz;
					dz = ABS(zsPrev.x - zsNext.x);

					dz = (dz < RDELTA) ? 0.0 : dz;

#if USEPARTIALPREINT

					dz /= maxEdgeLength;

					colorFront = tex1Dfetch(transferFunctionTex, ((uint)(255*zsNext.y)) );

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

					real psi = tex2D(psiGammaTableTex, texCoord.x, texCoord.y);

					real4 finalColor;
					finalColor.x = colorFront.x*(1.0 - psi) + colorBack.x*(psi - zeta);
					finalColor.y = colorFront.y*(1.0 - psi) + colorBack.y*(psi - zeta);
					finalColor.z = colorFront.z*(1.0 - psi) + colorBack.z*(psi - zeta);
					finalColor.w = 1.0 - zeta;

					accumColor.x = finalColor.x + (1.0 - finalColor.w) * accumColor.x;
					accumColor.y = finalColor.y + (1.0 - finalColor.w) * accumColor.y;
					accumColor.z = finalColor.z + (1.0 - finalColor.w) * accumColor.z;

					colorBack = colorFront;

#if USEOPACITY
					opacity += finalColor.w;
#endif

#else
					
					real transparency;

					/// Load transfer function color
					tfColor = tex1Dfetch(transferFunctionTex, ((uint)(255*zsNext.y)) );

					coNext.x = 255 * tfColor.x;
					coNext.y = 255 * tfColor.y;
					coNext.z = 255 * tfColor.z;
#if USEOPACITY
					coNext.w = tfColor.w;

					if ( (coNext.w < 0.0) || (coNext.w > 1.0) ) return;

 					transparency = 1.0 - opacity;
#else
					transparency = 1.0;
					coPrev.w = 0.0;
					coNext.w = 0.0;
#endif

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

#if USEOPACITY
					if ( coPrev.w < 0.0 ) return;

					if ( coNext.w < 0.0 ) return;

					opacity += (coPrev.w + coNext.w) * dz / 2.0;

					if ( opacity < 0.0 ) return;
#endif

					coPrev = coNext;

					if (rgb.x > 255) color.x = 255;
					else color.x = rgb.x;
					if (rgb.y > 255) color.y = 255;
					else color.y = rgb.y;
					if (rgb.z > 255) color.z = 255;
					else color.z = rgb.z;

#endif

#if USEOPACITY

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
					zsPrev = zsNext;

				} // while

#if USEPARTIALPREINT

				if (accumColor.x > 1.0) accumColor.x = 1.0;
				if (accumColor.y > 1.0) accumColor.y = 1.0;
				if (accumColor.z > 1.0) accumColor.z = 1.0;

#if USEOPACITY
				finalImage[ (pix.y * width + pix.x)*3 + 0 ] = 255 * accumColor.x * opacity;
				finalImage[ (pix.y * width + pix.x)*3 + 1 ] = 255 * accumColor.y * opacity;
				finalImage[ (pix.y * width + pix.x)*3 + 2 ] = 255 * accumColor.z * opacity;
#else
				finalImage[ (pix.y * width + pix.x)*3 + 0 ] = 255 * accumColor.x;
				finalImage[ (pix.y * width + pix.x)*3 + 1 ] = 255 * accumColor.y;
				finalImage[ (pix.y * width + pix.x)*3 + 2 ] = 255 * accumColor.z;
#endif

#else

#if USEOPACITY
 				finalImage[ (pix.y * width + pix.x)*3 + 0 ] = color.x * opacity;
 				finalImage[ (pix.y * width + pix.x)*3 + 1 ] = color.y * opacity;
 				finalImage[ (pix.y * width + pix.x)*3 + 2 ] = color.z * opacity;
#else
				finalImage[ (pix.y * width + pix.x)*3 + 0 ] = color.x;
				finalImage[ (pix.y * width + pix.x)*3 + 1 ] = color.y;
				finalImage[ (pix.y * width + pix.x)*3 + 2 ] = color.z;
#endif

#endif
			} // if

		} // yP

	} // xP

}

#endif
