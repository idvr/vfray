/**
 *   Visible Faces Ray Casting (VF-Ray)
 *
 * Using C++ -- January, 2008
 *
 */

/**
 *   VF-Ray Setup and Initialization
 *
 * C++ code.
 *
 */

/// ----------------------------------   Definitions   ------------------------------------

// CUDA requires
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "vfRayPreComp.h"

#define HEIGHT_RES  512
#define WIDTH_RES   512

#include "vfRayKernelCPU.h"

const char datasetDir[] = "../tet_offs/";

uint numVerts, numTets;

real *h_vertList;
uint *h_tetList;
uint *h_extFaces;

uint *h_conTet;
real *h_transferFunction;

uint numExtFaces, numVisFaces;

real h_maxEdgeLength, h_maxZ, h_minZ;

faceData *d_visFaces;

projectedFace *d_projVisFaces;

uchar *d_finalImage;

clock_t ctimer = 0;
real elapsedTime;

/// ----------------------------------   Functions   -------------------------------------

void
prepareDevice1(void) {

	// create and start timer
	ctimer = clock();

	uint msVisFaces = sizeof(faceData) * numExtFaces;
	d_visFaces = (faceData*) malloc(msVisFaces);

	// stop and destroy timer
	elapsedTime = ( clock() - ctimer ) / (real)CLOCKS_PER_SEC;
	printf("** Prepare 1 ** Processing time: %d (ms) \n", (uint)(elapsedTime * 1000.0));

}

void
executeKernel1(void) {

	// create and start timer
	ctimer = clock();

	// execute the kernel
	for (uint i = 0; i < numExtFaces; ++i) {
		findVisibleFaces( d_visFaces, h_vertList,
				  h_tetList, h_extFaces,
				  i );
	}

	// stop and destroy timer
	elapsedTime = ( clock() - ctimer ) / (real)CLOCKS_PER_SEC;
	printf("Processing time: %d (ms) \n", (uint)(elapsedTime * 1000.0));

	// create and start timer
	ctimer = clock();

	numVisFaces = 0;

	for (uint i = 0; i < numExtFaces; ++i) {

		if (d_visFaces[i].sParams.x != -1.0)
			++numVisFaces;

	}

	++numVisFaces;

	uint msVisFaces2 = sizeof(faceData) * numVisFaces;
	faceData* d_visFaces2 = (faceData*) malloc(msVisFaces2);

	uint msExtFaces2 = sizeof(uint) * 2 * numVisFaces;
	uint* h_extFaces2 = (uint*) malloc(msExtFaces2);

	for (uint i = 0, j = 0; i < numExtFaces; ++i) {

		if (d_visFaces[i].sParams.x != -1.0) {

			d_visFaces2[j] = d_visFaces[i];
			h_extFaces2[j*2 + 0] = h_extFaces[i*2 + 0];
			h_extFaces2[j*2 + 1] = h_extFaces[i*2 + 1];

			++j;

		}

	}

	d_visFaces2[numVisFaces-1] = d_visFaces2[numVisFaces-2];
	h_extFaces2[(numVisFaces-1)*2 + 0] = h_extFaces2[(numVisFaces-2)*2 + 0];
	h_extFaces2[(numVisFaces-1)*2 + 1] = h_extFaces2[(numVisFaces-2)*2 + 1];

	free(d_visFaces);
	free(h_extFaces);

	d_visFaces = (faceData*) malloc(msVisFaces2);
	h_extFaces = (uint*) malloc(msExtFaces2);

	memcpy( d_visFaces, d_visFaces2, msVisFaces2 );
	memcpy( h_extFaces, h_extFaces2, msExtFaces2 );

	free( d_visFaces2 );
	free( h_extFaces2 );

	// stop and destroy timer
	elapsedTime = ( clock() - ctimer ) / (real)CLOCKS_PER_SEC;
	printf("** Reduction ** Processing time: %d (ms) \n", (uint)(elapsedTime * 1000.0));

	printf(" # VisFaces: %d\n", numVisFaces);

}

void
prepareDevice2(void) {

	// create and start timer
	ctimer = clock();

	// allocate device memory for result
	uint msProjVisFaces = sizeof(projectedFace) * numVisFaces;
	d_projVisFaces = (projectedFace*) malloc(msProjVisFaces);

	// stop and destroy timer
	elapsedTime = ( clock() - ctimer ) / (real)CLOCKS_PER_SEC;
	printf("** Prepare 2 ** Processing time: %d (ms) \n", (uint)(elapsedTime * 1000.0));

}

void
executeKernel2(void) {

	// create and start timer
	ctimer = clock();

	// execute the kernel
	for (uint i = 0; i < numVisFaces; ++i) {	
		projectVisibleFaces( d_projVisFaces, h_vertList,
				     h_tetList, h_extFaces,
				     d_visFaces, i,
				     WIDTH_RES, HEIGHT_RES );
	}

	// stop and destroy timer
	elapsedTime = ( clock() - ctimer ) / (real)CLOCKS_PER_SEC;
	printf("Processing time: %d (ms) \n", (uint)(elapsedTime * 1000.0));

}

void
prepareDevice3(void) {

	// create and start timer
	ctimer = clock();

	// allocate host memory for the result
	uint msFinalImage = sizeof(uchar) * WIDTH_RES * HEIGHT_RES * 3;
	d_finalImage = (uchar*) malloc(msFinalImage);

	memset( d_finalImage, 0, msFinalImage );

	// stop and destroy timer
	elapsedTime = ( clock() - ctimer ) / (real)CLOCKS_PER_SEC;
	printf("** Prepare 3 ** Processing time: %d (ms) \n", (uint)(elapsedTime * 1000.0));

}

void
executeKernel3(void) {

	// create and start timer
	ctimer = clock();

	// execute the kernel
	for (uint i = 0; i < numVisFaces; ++i) {

		vfRay( d_finalImage, h_vertList,
		       h_tetList, h_extFaces,
		       d_visFaces, d_projVisFaces,
		       h_conTet, h_transferFunction,
		       i, WIDTH_RES, HEIGHT_RES,
		       h_maxEdgeLength, h_maxZ, h_minZ );

	}

	// stop and destroy timer
	elapsedTime = ( clock() - ctimer ) / (real)CLOCKS_PER_SEC;
	printf("Processing time: %d (ms) \n", (uint)(elapsedTime * 1000.0));

}

void
cleanUp(void) {

	// clean up CPU memory
	if (h_vertList) free( h_vertList );
	if (h_tetList) free( h_tetList );
	if (h_extFaces) free( h_extFaces );
	if (d_visFaces) free( d_visFaces );
	if (h_conTet) free( h_conTet );
	if (h_transferFunction) free( h_transferFunction );
	if (d_finalImage) free( d_finalImage );

}

/// -------------------------------------   Main   -----------------------------------------

int
main(int argc, char** argv) {

	if ( argc != 2 ) {
		printf("use: %s model\nWhere model can be: spx, blunt, ...\n", argv[0]);
		return 1;
	}
	argc--;

	char fnOff[64];
	strcpy(fnOff, datasetDir);
	strcat(fnOff, argv[1]);
	strcat(fnOff, ".off");
	printf("Reading %s file\n", fnOff);

	if ( !readOff( fnOff ) ) {

		printf("Cannot read %s aborting!\n", fnOff);
		return 1;

	}

	printf("Normalizing vertices\n");
	normalizeVertices();

	char fnCon[64];
	strcpy(fnCon, datasetDir);
	strcat(fnCon, argv[1]);
	strcat(fnCon, ".con");
	printf("Reading %s file\n", fnCon);

	if ( !readCon( fnCon ) ) {

		printf("%s file not found!\n Computing conTet\n", fnCon);

		char fnIncid[64];
		strcpy(fnIncid, datasetDir);
		strcat(fnIncid, argv[1]);
		strcat(fnIncid, ".incid");
		printf("Reading %s file\n", fnIncid);

		if ( !readIncid( fnIncid ) ) {
			printf("%s file not found!\n Building incid\n", fnIncid);
			buildIncid();
			printf("Writing %s file\n", fnIncid);
			writeIncid( fnIncid );
		}

		buildCon();
		printf("Writing %s file\n", fnCon);
		writeCon( fnCon );

		printf("Deleting incidents in vertices\n");
		deleteIncid();

	}

	char fnTF[64];
	strcpy(fnTF, datasetDir);
	strcat(fnTF, argv[1]);
	strcat(fnTF, ".tf");
	printf("Reading %s file\n", fnTF);

	if ( !readTF( fnTF ) ) {

		printf("%s file not found!\n Computing TF\n", fnTF);
		buildTF();
		printf("Writing %s file\n", fnTF);
		writeTF( fnTF );

	}

	char fnExtF[64];
	strcpy(fnExtF, datasetDir);
	strcat(fnExtF, argv[1]);
	strcat(fnExtF, ".extf");
	printf("Reading %s file\n", fnExtF);

	if ( !readExtF( fnExtF ) ) {

		printf("%s file not found!\n Computing extFaces\n", fnExtF);
		buildExtF();
		printf("Writing %s file\n", fnExtF);
		writeExtF( fnExtF );

	}

	char fnLmt[64];
	strcpy(fnLmt, datasetDir);
	strcat(fnLmt, argv[1]);
	strcat(fnLmt, ".lmt");
	printf("Reading %s file\n", fnLmt);

	if ( !readLmt( fnLmt ) ) {

		printf("%s file not found!\n Computing limits\n", fnLmt);
		buildLmt();
		printf("Writing %s file\n", fnLmt);
		writeLmt( fnLmt );

	}

	printf("Volume dataset information: %s\n", fnOff);
	printf("  # Tets: %d ; # Verts: %d\n", numTets, numVerts);
	printf("  # ExtFaces: %d\n", numExtFaces);
	printf("  Max Edge Length: %f\n", h_maxEdgeLength);
	printf("  Max Z: %f ; Min Z: %f\n", h_maxZ, h_minZ);

	printf("Preparing device for kernel 1\n");
	prepareDevice1();

	printf("Execute kernel 1 on device\n");
	executeKernel1();

	printf("Preparing device for kernel 2\n");
	prepareDevice2();

	printf("Execute kernel 2 on device\n");
	executeKernel2();

	printf("Preparing device for kernel 3\n");
	prepareDevice3();

	printf("Execute kernel 3 on device\n");
	executeKernel3();

	char fnImage[64];
	strcpy(fnImage, argv[1]);
	strcat(fnImage, "_cpu_img.ppm");

	printf("Saving ppm image: %s\n", fnImage);
	ppmWrite(fnImage, d_finalImage, WIDTH_RES, HEIGHT_RES);

	printf("Clean up device\n");
	cleanUp();

	return 0;

}
