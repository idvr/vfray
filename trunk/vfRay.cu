/**
 *   Visible Faces Ray Casting (VF-Ray)
 *
 * Using CUDA -- January, 2008
 *
 */

/**
 *   VF-Ray Setup and Initialization
 *
 * Host code.
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

#include <cutil.h>

#include </usr/include/GL/glut.h> ///< Use system GLUT

#define HEIGHT_RES  512
#define WIDTH_RES   512

#define USEBBOXTILES 1

#define USEVFRAYBUFFER 0

#define USECONTETTEX 0

#define USEPARTIALPREINT 1

#define REDUCTION 1

#define USEOPACITY 0

#define OPENGL 0

#define BLOCK_SIZE 8

#include <vfRayKernel.cu>

/// From vfRayPreComp
extern "C" bool
readOff(const char*);

extern "C" void
normalizeVertices(void);

extern "C" bool
readIncid(const char* f);

extern "C" bool
buildIncid(void);

extern "C" bool
writeIncid(const char* f);

extern "C" bool
deleteIncid(void);

extern "C" bool
readCon(const char* f);

extern "C" bool
buildCon(void);

extern "C" bool
writeCon(const char* f);

extern "C" bool
readTF(const char* f);

extern "C" bool
buildTF(void);

extern "C" bool
writeTF(const char* f);

extern "C" bool
readExtF(const char* f);

extern "C" bool
buildExtF(void);

extern "C" bool
writeExtF(const char* f);

extern "C" bool
readLmt(const char* f);

extern "C" bool
buildLmt(void);

extern "C" bool
writeLmt(const char* f);

extern "C" void
rotateVolume(real *modelView);

const char datasetDir[] = "../tet_offs/";

uint numVerts, numTets;

real *h_vertList;
uint *h_tetList;
uint *h_extFaces;

uint *h_conTet;
real *h_transferFunction;

uint numExtFaces, numVisFaces;

real h_maxEdgeLength, h_maxZ, h_minZ;

real *d_vertList;
uint *d_tetList;
uint *d_extFaces;

faceData *d_visFaces;

projectedFace *d_projVisFaces;

uint *d_conTet;
real *d_transferFunction;

uchar *d_finalImage, *h_finalImage;

#if USEPARTIALPREINT

cudaArray* d_psiGammaTable;

#include "../psiGammaTable/psiGammaTable512.h"

#endif

uint msVertList, msTetList,
	msExtFaces, msVisFaces,
	msConTet, msTransferFunction,
	msFinalImage;

uint timer = 0;
clock_t ctimer = 0;
real elapsedTime;

dim3 gridPerThread, gridPerBlock;

dim3 block(BLOCK_SIZE, BLOCK_SIZE), grid;

/// GLUTisess
static int winWidth = WIDTH_RES, winHeight = HEIGHT_RES;
static int buttonPressed = -1; ///< button state
static GLfloat oldx = 0.0, oldy = 0.0, xangle = 0.0, yangle = 0.0;
static int xmouse, ymouse;

/// ----------------------------------   Functions   -------------------------------------

//Round a / b to nearest higher integer value
__host__ inline uint
iDivUp(uint a, uint b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Align a to nearest higher multiple of b
__host__ inline uint
iAlignUp(uint a, uint b) {
	return (a % b != 0) ? (a - a % b + b) : a;
}

/// Setup Grid with Blocks with Threads
__host__ void
setupGrids(uint targetNum) {

	// setup kernels parameters
	uint desiredGridWidth = iDivUp(targetNum, BLOCK_SIZE*BLOCK_SIZE);

	if (desiredGridWidth < 512) {

 		gridPerThread.x = iAlignUp(desiredGridWidth, 16) / 16;
 		gridPerThread.y = 16;

	} else {

		gridPerThread.x = iAlignUp(desiredGridWidth, 64) / 64;
		gridPerThread.y = 64;

	}

	desiredGridWidth = targetNum;

	if (desiredGridWidth < 512) {

		gridPerBlock.x = iAlignUp(desiredGridWidth, 16) / 16;
		gridPerBlock.y = 16;

	} else {

 		gridPerBlock.x = iAlignUp(desiredGridWidth, 64) / 64;
 		gridPerBlock.y = 64;

	}

	printf("# External/Visible Faces: %d\n", targetNum);
	printf("Block dimension: ( %d, %d )\n", gridPerThread.x, gridPerThread.y);
	printf("Grid dimension : ( %d, %d )\n", gridPerBlock.x, gridPerBlock.y);

}

__host__ void
completeExtFaces(void) {

	// Get last extFaceId;
	uint last = numExtFaces - 1;

	// Allocate one more space for invalid thread computations
	numExtFaces = numExtFaces + 1;

	uint i;
	uint *tmp_extFaces;

	uint msExtFaces = sizeof(uint) * 2 * numExtFaces;
	tmp_extFaces = (uint*) malloc(msExtFaces);

	// Copy old extFaces array to a temporary array
	for (i = 0; i < numExtFaces; ++i) {

		tmp_extFaces[i*2 + 0] = ( i <= last ) ? h_extFaces[i*2 + 0] : h_extFaces[last*2 + 0];
		tmp_extFaces[i*2 + 1] = ( i <= last ) ? h_extFaces[i*2 + 1] : h_extFaces[last*2 + 1];

	}

	// Delete and recreate extFaces array
	free( h_extFaces );
	h_extFaces = (uint*) malloc(msExtFaces);

	// Copy the temporary array to extFaces
	for (i = 0; i < numExtFaces; ++i) {

		h_extFaces[i*2 + 0] = tmp_extFaces[i*2 + 0];
		h_extFaces[i*2 + 1] = tmp_extFaces[i*2 + 1];

	}

	// Delete temporary array
	free( tmp_extFaces );

}

__host__ void
initCUDA(void) {

	msVertList = sizeof(real) * 4 * numVerts;
	msExtFaces = sizeof(uint) * 2 * numExtFaces;
	msTetList = sizeof(uint) * 4 * numTets;
	msConTet = sizeof(uint) * 4 * numTets;
	msTransferFunction = sizeof(real) * 4 * 256;

	// allocate device memory
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_vertList, msVertList) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_extFaces, msExtFaces) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_tetList, msTetList) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_conTet, msConTet) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_transferFunction, msTransferFunction) );

 	CUDA_SAFE_CALL( cudaMemcpy(d_tetList, h_tetList, msTetList,
 				   cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_conTet, h_conTet, msConTet,
				   cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_transferFunction, h_transferFunction, msTransferFunction,
				   cudaMemcpyHostToDevice) );

#if USEPARTIALPREINT
	uint size = 512 * 512 * sizeof(float);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	CUDA_SAFE_CALL( cudaMallocArray( &d_psiGammaTable, &channelDesc, 512, 512 ));
	CUDA_SAFE_CALL( cudaMemcpyToArray( d_psiGammaTable, 0, 0, psiGammaTable, size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaBindTextureToArray(psiGammaTableTex, d_psiGammaTable, channelDesc) );
#endif


}

__host__ void
prepareDevice1(void) {

	// create and start timer
	ctimer = clock();

	msExtFaces = sizeof(uint) * 2 * numExtFaces;
	msVisFaces = sizeof(faceData) * numExtFaces;

	// reallocate extFaces array
	CUDA_SAFE_CALL( cudaFree(d_extFaces) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_extFaces, msExtFaces) );

	// copy host memory to device
	CUDA_SAFE_CALL( cudaMemcpy(d_vertList, h_vertList, msVertList,
				   cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_extFaces, h_extFaces, msExtFaces,
				   cudaMemcpyHostToDevice) );

	// reallocate visFaces array
	CUDA_SAFE_CALL( cudaFree(d_visFaces) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_visFaces, msVisFaces) );

	// stop and destroy timer
	elapsedTime = ( clock() - ctimer ) / (real)CLOCKS_PER_SEC;
	printf("** Prepare 1 ** Processing time: %d (ms) \n", (uint)(elapsedTime * 1000.0));

}

__host__ void
executeKernel1(void) {

	grid = gridPerThread;

	// create and start timer
	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));

	// bind device memory to texture
	CUDA_SAFE_CALL( cudaBindTexture(0, vertListTex, d_vertList, msVertList) );
	CUDA_SAFE_CALL( cudaBindTexture(0, extFacesTex, d_extFaces, msExtFaces) );

	// execute the kernel
	findVisibleFaces<<< grid, block >>>( d_visFaces,
					     d_tetList,
					     numExtFaces,
					     grid.x );

	// check if kernel execution generated and error
	CUT_CHECK_ERROR("Kernel 1 execution failed");

	// unbind texture
	CUDA_SAFE_CALL( cudaUnbindTexture(vertListTex) );
	CUDA_SAFE_CALL( cudaUnbindTexture(extFacesTex) );

#if REDUCTION

	// create and start timer
	ctimer = clock();

	msVisFaces = sizeof(faceData) * numExtFaces;
	faceData* h_visFaces = (faceData*) malloc(msVisFaces);
	CUDA_SAFE_CALL(cudaMemcpy(h_visFaces, d_visFaces, msVisFaces,
	                          cudaMemcpyDeviceToHost) );

	numVisFaces = 0;

	for (uint i = 0; i < numExtFaces; ++i) {

		if (h_visFaces[i].sParams.x != -1.0)
			++numVisFaces;

	}

	++numVisFaces;

	uint msVisFaces2 = sizeof(faceData) * numVisFaces;
	faceData* h_visFaces2 = (faceData*) malloc(msVisFaces2);

	uint msExtFaces2 = sizeof(uint) * 2 * numVisFaces;
	uint* h_extFaces2 = (uint*) malloc(msExtFaces2);

	for (uint i = 0, j = 0; i < numExtFaces; ++i) {

		if (h_visFaces[i].sParams.x != -1.0) {

			h_visFaces2[j] = h_visFaces[i];
			h_extFaces2[j*2 + 0] = h_extFaces[i*2 + 0];
			h_extFaces2[j*2 + 1] = h_extFaces[i*2 + 1];

			++j;

		}

	}

	h_visFaces2[numVisFaces-1] = h_visFaces2[numVisFaces-2];
	h_extFaces2[(numVisFaces-1)*2 + 0] = h_extFaces2[(numVisFaces-2)*2 + 0];
	h_extFaces2[(numVisFaces-1)*2 + 1] = h_extFaces2[(numVisFaces-2)*2 + 1];

	CUDA_SAFE_CALL( cudaFree(d_visFaces) );
	CUDA_SAFE_CALL( cudaFree(d_extFaces) );

	CUDA_SAFE_CALL( cudaMalloc((void**) &d_visFaces, msVisFaces2) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_extFaces, msExtFaces2) );

	CUDA_SAFE_CALL( cudaMemcpy(d_visFaces, h_visFaces2, msVisFaces2,
				   cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_extFaces, h_extFaces2, msExtFaces2,
				   cudaMemcpyHostToDevice) );

	free( h_visFaces2 );
	free( h_extFaces2 );

	setupGrids(numVisFaces);

	// stop and destroy timer
	elapsedTime = ( clock() - ctimer ) / (real)CLOCKS_PER_SEC;
	printf("** Reduction ** Processing time: %d (ms) \n", (uint)(elapsedTime * 1000.0));

#else

	numVisFaces = numExtFaces;

#endif

	// stop and destroy timer
	CUT_SAFE_CALL(cutStopTimer(timer));
	printf("** Kernel 1  ** Processing time: %f (ms) \n", cutGetTimerValue(timer));
	CUT_SAFE_CALL(cutDeleteTimer(timer));

}

__host__ void
prepareDevice2(void) {

	// create and start timer
	ctimer = clock();

	CUDA_SAFE_CALL( cudaFree(d_projVisFaces) );

	// allocate device memory for result
	uint msProjVisFaces = sizeof(projectedFace) * numVisFaces;
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_projVisFaces, msProjVisFaces) );

	// stop and destroy timer
	elapsedTime = ( clock() - ctimer ) / (real)CLOCKS_PER_SEC;
	printf("** Prepare 2 ** Processing time: %d (ms) \n", (uint)(elapsedTime * 1000.0));

}

__host__ void
executeKernel2(void) {

	grid = gridPerThread;

	// create and start timer
	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));

	// bind device memory to texture
	CUDA_SAFE_CALL( cudaBindTexture(0, vertListTex, d_vertList, msVertList) );
	CUDA_SAFE_CALL( cudaBindTexture(0, extFacesTex, d_extFaces, msExtFaces) );

	// execute the kernel
	projectVisibleFaces<<< grid, block >>>( d_projVisFaces,
						d_tetList,
						d_visFaces,
						numVisFaces,
						grid.x,
						winWidth,
						winHeight );

	// check if kernel execution generated and error
	CUT_CHECK_ERROR("Kernel 2 execution failed");

	// unbind texture
	CUDA_SAFE_CALL( cudaUnbindTexture(vertListTex) );
	CUDA_SAFE_CALL( cudaUnbindTexture(extFacesTex) );

	// stop and destroy timer
	CUT_SAFE_CALL(cutStopTimer(timer));
	printf("** Kernel 2  ** Processing time: %f (ms) \n", cutGetTimerValue(timer));
	CUT_SAFE_CALL(cutDeleteTimer(timer));

}

__host__ void
prepareDevice3(void) {

	// create and start timer
	ctimer = clock();

	free( h_finalImage );
	CUDA_SAFE_CALL( cudaFree(d_finalImage) );

	// allocate host memory for the result
	msFinalImage = sizeof(uchar) * winWidth * winHeight * 3;
	h_finalImage = (uchar*) malloc(msFinalImage);

	// allocate device memory for result and copy host memory to device
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_finalImage, msFinalImage) );
	CUDA_SAFE_CALL( cudaMemset( d_finalImage, 0, msFinalImage ) );

	// stop and destroy timer
	elapsedTime = ( clock() - ctimer ) / (real)CLOCKS_PER_SEC;
	printf("** Prepare 3 ** Processing time: %d (ms) \n", (uint)(elapsedTime * 1000.0));

}

__host__ void
executeKernel3(void) {

#if USEBBOXTILES
	dim3 grid = gridPerBlock;
#else
	dim3 grid = gridPerThread;
#endif

	// create and start timer
	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));

	// bind device memory to texture
	CUDA_SAFE_CALL( cudaBindTexture(0, vertListTex, d_vertList, msVertList) );
	CUDA_SAFE_CALL( cudaBindTexture(0, extFacesTex, d_extFaces, msExtFaces) );
#if USECONTETTEX
	CUDA_SAFE_CALL( cudaBindTexture(0, conTetTex, d_conTet, msConTet) );
#endif
	CUDA_SAFE_CALL( cudaBindTexture(0, transferFunctionTex, d_transferFunction, msTransferFunction) );

#if USEPARTIALPREINT
	CUDA_SAFE_CALL( cudaBindTextureToArray(psiGammaTableTex, d_psiGammaTable) );
#endif

	// execute the kernel
	vfRay<<< grid, block >>>( d_finalImage,
				  d_tetList,
#if USECONTETTEX
#else
				  d_conTet,
#endif
				  d_visFaces,
				  d_projVisFaces,
				  numVisFaces,
				  grid.x,
				  winWidth,
				  winHeight
#if USEVFRAYBUFFER
				  ,h_maxZ
				  ,h_minZ
#endif
#if USEPARTIALPREINT
				  ,h_maxEdgeLength
#endif
		);

	// check if kernel execution generated and error
	CUT_CHECK_ERROR("Kernel 3 execution failed");

	// unbind texture
	CUDA_SAFE_CALL( cudaUnbindTexture(vertListTex) );
	CUDA_SAFE_CALL( cudaUnbindTexture(extFacesTex) );
#if USECONTETTEX
	CUDA_SAFE_CALL( cudaUnbindTexture(conTetTex) );
#endif
	CUDA_SAFE_CALL( cudaUnbindTexture(transferFunctionTex) );

	// copy result from device to host
	CUDA_SAFE_CALL(cudaMemcpy(h_finalImage, d_finalImage, msFinalImage,
	                          cudaMemcpyDeviceToHost) );

	// stop and destroy timer
	CUT_SAFE_CALL(cutStopTimer(timer));
	printf("** Kernel 3  ** Processing time: %f (ms) \n", cutGetTimerValue(timer));
	CUT_SAFE_CALL(cutDeleteTimer(timer));


}

__host__ void
cleanUp(void) {

	// clean up CPU memory
	if (h_vertList) free( h_vertList );
	if (h_tetList) free( h_tetList );
	if (h_extFaces) free( h_extFaces );
	if (h_conTet) free( h_conTet );
	if (h_transferFunction) free( h_transferFunction );
	if (h_finalImage) free( h_finalImage );

	// clean up GPU memory
	CUDA_SAFE_CALL(cudaFree(d_vertList));
	CUDA_SAFE_CALL(cudaFree(d_tetList));
	CUDA_SAFE_CALL(cudaFree(d_extFaces));
	CUDA_SAFE_CALL(cudaFree(d_visFaces));
	CUDA_SAFE_CALL(cudaFree(d_projVisFaces));
	CUDA_SAFE_CALL(cudaFree(d_conTet));
	CUDA_SAFE_CALL(cudaFree(d_transferFunction));
	CUDA_SAFE_CALL(cudaFree(d_finalImage));
#if USEPARTIALPREINT
	CUDA_SAFE_CALL(cudaFreeArray(d_psiGammaTable));
#endif

}

/// -----------------------------------   OpenGL   -----------------------------------------

void keyboard( unsigned char key, int x, int y ) {

	static real mv[16];
	static real rotx = 0.0, roty = 0.0;
// 	static real phix, phiy, phiz, cos_phix, cos_phiy,
// 		sin_phix, sin_phiy;

	switch(key) {
	case 'o':
		rotx = 0.0; roty = 0.0;
		printf("* Rot X = %f ; Rot Y = %f\n", rotx, roty);
		break;
	case 'x':
		rotx += 10.0;
		printf("* Rot X = %f\n", rotx);
		break;
	case 'y':
		roty += 10.0;
		printf("* Rot Y = %f\n", roty);
		break;
	case ' ':
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glRotatef(rotx, 1.0, 0.0, 0.0);
		glRotatef(roty, 0.0, 1.0, 0.0);
		glGetFloatv(GL_MODELVIEW_MATRIX, mv);
		glLoadIdentity();
		rotateVolume(mv);
		printf("Normalizing vertices\n");
		normalizeVertices();
		printf("Setup all kernel grids\n");
		setupGrids(numExtFaces);
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
		break;
	case 'q': case 'Q': case 27:
		cleanUp();
		glutDestroyWindow( glutGetWindow() );
		return;
        default:
		break;
	}

}

void display(void) {

	glClear(GL_COLOR_BUFFER_BIT);

	glDrawPixels(winWidth, winHeight, GL_RGB, GL_UNSIGNED_BYTE, h_finalImage); 

	glutSwapBuffers();

}

void mouse(int button, int state, int x, int y) {

	buttonPressed = button;
	xmouse = x;
	ymouse = y;

	if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_UP)) {

		oldx += xangle;
		oldy += yangle;
		if (oldx > 360.0) oldx -= 360.0;
		if (oldx < 0.0) oldx += 360.0;
		if (oldy > 360.0) oldy -= 360.0;
		if (oldy < 0.0) oldy += 360.0;
		xangle = 0.0;
		yangle = 0.0;

		glutPostRedisplay();

	}

}

void motion (int x, int y) {

	if (buttonPressed == GLUT_LEFT_BUTTON) {

		yangle = (x - xmouse) * 360 / (GLfloat) winWidth;
		xangle = (y - ymouse) * 180 / (GLfloat) winHeight;

		glutPostRedisplay();

	}

}

void reshape(int w, int h) {

	glViewport(0, 0, winWidth=w, winHeight=h);

	printf("Preparing device for kernel 2\n");
	prepareDevice2();
	printf("Execute kernel 2 on device\n");
	executeKernel2();
	printf("Preparing device for kernel 3\n");
	prepareDevice3();
	printf("Execute kernel 3 on device\n");
	executeKernel3();

}

void idle(void) {

	glutPostRedisplay();

}

void initGL(int& argc, char** argv) {

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(winWidth, winHeight);
	glutCreateWindow("VF-Ray in CUDA");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutReshapeFunc(reshape);
	glutIdleFunc(idle);

}

/// -------------------------------------   Main   -----------------------------------------

__host__ int
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

/*
	real mv[16] = { 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 };

	real phix = 22.0;
	real phiy = 35.0;
	real phiz = 0.0;

	phix *= M_PI/180.0;
	phiy *= M_PI/180.0;
	phiz *= M_PI/180.0;

	real cos_phix = cos(phix);
	real cos_phiy = cos(phiy);
	real sin_phix = sin(phix);
	real sin_phiy = sin(phiy);

	mv[0*4+0] = cos_phiy;
	mv[0*4+1] = 0;
	mv[0*4+2] = -sin_phiy;
	mv[1*4+0] = sin_phix*sin_phiy;
	mv[1*4+1] = cos_phix;
	mv[1*4+2] = cos_phiy*sin_phix;
	mv[2*4+0] = cos_phix*sin_phiy;
	mv[2*4+1] = -sin_phix;
	mv[2*4+2] = cos_phix*cos_phiy;

	printf("Rotate volume\n");
	rotateVolume(mv);
*/

	printf("Complete external faces array for invalid threads\n");
	completeExtFaces();

	printf("Setup all kernel grids\n");
	setupGrids(numExtFaces);

	//-- CUDA-calls

	printf("***********************************\n");
	printf("Initializing CUDA\n");

	CUT_DEVICE_INIT(argc, argv);

	initCUDA();

	printf("***********************************\n");

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


#if OPENGL

	initGL(argc, argv);

	glutMainLoop();

#else

	char fnImage[64];
	strcpy(fnImage, argv[1]);
	strcat(fnImage, "_gpu_img.ppm");

	printf("Saving ppm image: %s\n", fnImage);
	cutSavePPMub(fnImage, h_finalImage, winWidth, winHeight);

	printf("Clean up device\n");
	cleanUp();

	CUT_EXIT(argc, argv);

#endif

	return 0;

}
