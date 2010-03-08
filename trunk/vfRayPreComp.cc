/**
 *   Visible Faces Ray Casting (VF-Ray)
 *
 * C++ only code -- January, 2008
 *
 */

/**
 *   VF-Ray Pre-Computation
 *
 * C++ code.
 *
 */

/// ----------------------------------   Definitions   ------------------------------------

#include "vfRayPreComp.h"

static incident *incidVert; ///< tetrahedra and vertices incident in each vertex

/// ----------------------------------   Functions   -------------------------------------

/// --- OFF ---

/// Read OFF (object file format)
/// @arg in input file stream
/// @return true if it succeed

bool readOff(ifstream& in) {

	if (in.fail()) return false;

	in >> numVerts >> numTets;

	/// Allocating memory for vertices and tetrahedra data

	if (h_vertList) delete [] h_vertList;
	h_vertList = new real[ 4 * numVerts ];
	if (!h_vertList) return false; ///< MR = nV * 4 * (real)

	if (h_tetList) delete [] h_tetList;
	h_tetList = new uint[ 4 * numTets ];
	if (!h_tetList) return false; ///< MR = nT * 4 * (real)

	/// Reading vertices and tetrahedra information

	for(uint i = 0; i < numVerts; i++) {

		in >> h_vertList[i*4 + 0] >> h_vertList[i*4 + 1]
		   >> h_vertList[i*4 + 2] >> h_vertList[i*4 + 3];

		if (in.fail()) return false;

	}

	for(uint i = 0; i < numTets; i++) {

		in >> h_tetList[i*4+0] >> h_tetList[i*4+1]
		   >> h_tetList[i*4+2] >> h_tetList[i*4+3];

		if (in.fail()) return false;

	}

	in.close();

	return true;

}

/// Read OFF (overload)
/// @arg f off file name
/// @return true if it succeed
bool readOff(const char* f) {

	ifstream in(f);

	return readOff(in);

}

/// Normalize vertex coordinates

void normalizeVertices() {

	real scaleCoord, scaleScalar, maxCoord, value;
	real center[3];
	real min[4], max[4];

	for(uint i = 0; i < 4; i++) {
		min[i] = h_vertList[0*4+i];
		max[i] = h_vertList[0*4+i];
	}

	/// Find out the min/max points

	for(uint i = 1; i < numVerts; i++) {

		for(uint j = 0; j < 4; j++) { // x, y, z, s

			value = h_vertList[i*4+j];

			if (value < min[j])
				min[j] = value;

			if (value > max[j])
				max[j] = value;

		}

	}

	/// Compute the center point

	for(uint i = 0; i < 3; ++i) { // x, y, z

		center[i] = (min[i] + max[i]) / 2.0;

		/// Center volume in origin
		max[i] -= center[i];
		min[i] -= center[i];

	}

	maxCoord = (max[1] > max[2]) ? max[1] : max[2];
	maxCoord = (max[0] > maxCoord) ? max[0] : maxCoord;

	scaleCoord = 1.0 / maxCoord;
	scaleScalar = 1.0 / (max[3] - min[3]);

	/// Update vertex list
  
	for(uint i = 0; i < numVerts; ++i) {

		for (int j = 0; j < 3; ++j) { // x, y, z

			h_vertList[i*4+j] = (h_vertList[i*4+j] - center[j]) * scaleCoord;

		}

		h_vertList[i*4+3] = (h_vertList[i*4+3] - min[3]) * scaleScalar;

	}

}

/// --- Incid ---

/// Read incidVert (incidents in vertex)
/// @arg in input file stream
/// @return true if it succeed

bool readIncid(ifstream& in) {

	if (in.fail()) return false;

	uint i, j, numIncidVerts, tetIdSize, vertIdSize, tetId, vertId;

	in >> numIncidVerts;

	if (numIncidVerts != numVerts) return false;

	/// Allocating memory for incidents in vertex data

	if (incidVert) delete [] incidVert;
	incidVert = new incident[ numVerts ];
	if (!incidVert) return false; ///< MR = ?

	/// Reading indents in vertex information

	for(i = 0; i < numVerts; i++) {

		in >> tetIdSize;

		for (j = 0; j < tetIdSize; j++) {

			in >> tetId;

			incidVert[i].tetId.push_back( tetId );

			if (in.fail()) return false;

		}

		in >> vertIdSize;

		for (j = 0; j < vertIdSize; j++) {

			in >> vertId;

			incidVert[i].vertId.insert( vertId );

			if (in.fail()) return false;

		}

	}

	in.close();

	return true;

}

/// Read Incid (overload)
/// @arg f incid file name
/// @return true if it succeed
bool readIncid(const char* f) {

	ifstream in(f);

	return readIncid(in);

}

/// Build Incid
/// @return true if it is succeed

bool buildIncid(void) {

        uint i, j, k;

	if (numVerts <= 0) return false;

	if (incidVert) delete [] incidVert;
	incidVert = new incident[ numVerts ];
	if (!incidVert) return false; ///< MR = ?
    
	if (!h_tetList) return false;

        for (i = 0; i < numVerts; ++i) { /// for each vertex
            
		incidVert[i].tetId.clear();

		incidVert[i].vertId.clear();

                for (j = 0; j < numTets; ++j) {
               
			for (k = 0; k < 4; ++k) {

				/// if tetrahedron j contains the vertex i
                                if (i == h_tetList[j*4+k]) {

					/// tetrahedron j is incident on vertex i
					incidVert[i].tetId.push_back( j );

					/// other three vertices are incident on vertex i
					incidVert[i].vertId.insert( h_tetList[j*4+MOD4(1, k)] );
					incidVert[i].vertId.insert( h_tetList[j*4+MOD4(2, k)] );
					incidVert[i].vertId.insert( h_tetList[j*4+MOD4(3, k)] );

                                        break;

                                } // if

			} // k

                } // j

                if ( incidVert[i].tetId.empty() ) {

                        return false; ///< some anomaly happens

                } // if

        } // i

	return true;

}

/// Write Incid (incidents in vertex)
/// @arg out output file stream
/// @return true if it succeed

bool writeIncid(ofstream& out) {

	if (out.fail()) return false;

	if (!incidVert) return false;

	uint i, j;

	out << numVerts << endl;

	/// Writing indents in vertex information
	///  Line 1: [ # incident tet ] [ list of tet ids ... ]
	///  Line 2: [ # incident vert ] [ list of vert ids ... ]

	for(i = 0; i < numVerts; i++) {

		out << incidVert[i].tetId.size() << " ";

		for (j = 0; j < incidVert[i].tetId.size(); j++) {

			out << incidVert[i].tetId[j];

			if ( j < (incidVert[i].tetId.size()-1) )
				out << " ";
			else
				out << endl;

			if (out.fail()) return false;

		}

		out << incidVert[i].vertId.size() << " ";

		j = 0;

		for (set< uint, less<uint> >::iterator it = incidVert[i].vertId.begin(); it != incidVert[i].vertId.end(); it++) {

			out << (*it);

			if ( j < (incidVert[i].vertId.size()-1) )
				out << " ";
			else
				out << endl;

			if (out.fail()) return false;

			j++;

		}

	}

	out.close();

	return true;

}

/// Write Incid (overload)
/// @arg f incid file name
/// @return true if it succeed
bool writeIncid(const char* f) {

	ofstream out(f);

	return writeIncid(out);

}


/// Delete incidVert array
/// @return true if it succeed

bool deleteIncid(void) {

        if (!incidVert)
		return false;

        for (uint i = 0; i < numVerts; ++i) {

                incidVert[i].tetId.clear();

		incidVert[i].vertId.clear();

        }

        if (incidVert)
                delete [] incidVert;

	return true;

}

/// --- Con ---

/// Read Con (tetrahedra connectivity)
/// @arg in input file stream
/// @return true if it succeed

bool readCon(ifstream& in) {

	if (in.fail()) return false;

	uint numCons;

	in >> numExtFaces;

	in >> numCons;

	if (numCons != numTets) return false;

	/// Allocating memory for tetrahedra connectivity data

	if (h_conTet) delete [] h_conTet;
	h_conTet = new uint[ 4 * numTets ];
	if (!h_conTet) return false; ///< MR = nT * 4 * (uint)

	/// Reading tetrahedra connectivity information

	for(uint i = 0; i < numTets; i++) {

		in >> h_conTet[i*4 + 0] >> h_conTet[i*4 + 1]
		   >> h_conTet[i*4 + 2] >> h_conTet[i*4 + 3];

		if (in.fail()) return false;

	}

	in.close();

	return true;

}

/// Read Con (overload)
/// @arg f h_conTet file name
/// @return true if it succeed
bool readCon(const char* f) {

	ifstream in(f);

	return readCon(in);

}

/// Build tetrahedra connectivity
/// @return true if it succeed

bool buildCon(void) {

        uint i, j, k, l, f, currTetId;

	numExtFaces = 0;

	uint currVId[3];

        if (!incidVert) return false;

	if (h_conTet) delete [] h_conTet;
	h_conTet = new uint[ 4 * numTets ];
	if (!h_conTet) return false; ///< MR = nT * 4 * (uint)

	for (i = 0; i < numTets; i++) { /// for each tet

                for (f = 0; f < 4; f++) { /// for each face

			for (k = 0; k < 3; k++) { // for each vertex of current face

				currVId[k] = h_tetList[ i*4 + MOD4(k, f) ];

			}

			h_conTet[i*4+f] = i; /// sets all face as external face

                        for (j = 0; j < incidVert[ currVId[0] ].tetId.size(); j++) {

                                currTetId = incidVert[ currVId[0] ].tetId[j];

				if (i != currTetId) {

                                        for (k = 0; k < incidVert[ currVId[1] ].tetId.size(); k++) {

                                                if ( incidVert[ currVId[1] ].tetId[k] > currTetId )
							break;

                                                if ( currTetId == incidVert[ currVId[1] ].tetId[k] ) {

                                                        for (l = 0; l < incidVert[ currVId[2] ].tetId.size(); l++) {

                                                                if ( incidVert[ currVId[2] ].tetId[l] > currTetId )
									break;

                                                                if ( currTetId == incidVert[ currVId[2] ].tetId[l] ) {

									h_conTet[i*4+f] = currTetId;

                                                                        break;

								}

							} // l

							if (h_conTet[i*4+f] == currTetId) /// it has already been found
								break;

						} // if

					} // k

					if (h_conTet[i*4+f] == currTetId) /// it has already been found
						break;

				} // if

			} // j

			if (h_conTet[i*4+f] == i) /// if this is an external face
				numExtFaces++;

                } // f

        } // i

	return true;

}

/// Write Con (tetrahedra connectivity)
/// @arg out output file stream
/// @return true if it succeed

bool writeCon(ofstream& out) {

	if (out.fail()) return false;

	if (!h_conTet) return false;

	out << numExtFaces << endl;

	out << numTets << endl;

	/// Writing tetrahedra connectivity information

	for(uint i = 0; i < numTets; i++) {

		out << h_conTet[i*4 + 0] << " " << h_conTet[i*4 + 1] << " "
		    << h_conTet[i*4 + 2] << " " << h_conTet[i*4 + 3] << endl;

		if (out.fail()) return false;

	}

	out.close();

	return true;

}

/// Write Con (overload)
/// @arg f h_conTet file name
/// @return true if it succeed
bool writeCon(const char* f) {

	ofstream out(f);

	return writeCon(out);

}

/// --- TF ---

/// Read TF (transfer function)
/// @arg in input file stream
/// @return true if it succeed

bool readTF(ifstream& in) {

	if (in.fail()) return false;

	uint numColors;

	in >> numColors;
	if (numColors != 256) return false;

	if (h_transferFunction) delete [] h_transferFunction;
	h_transferFunction = new real[ 4 * 256 ];
	if (!h_transferFunction) return false; ///< MR = 256 * 4 * (real)

	for(uint i = 0; i < 256; i++) {

		in >> h_transferFunction[ i*4 + 0 ]
		   >> h_transferFunction[ i*4 + 1 ]
		   >> h_transferFunction[ i*4 + 2 ]
		   >> h_transferFunction[ i*4 + 3 ];

		if (in.fail()) return false;

	}

	in.close();

	return true;

}

/// Read TF (overload)
/// @arg f transfer function file name
/// @return true if it succeed
bool readTF(const char* f) {

	ifstream in(f);

	return readTF(in);

}

/// Build a generic TF array
/// @return true if it succeed

bool buildTF(void) {

	real c[4];

	if (h_transferFunction)	delete [] h_transferFunction;
	h_transferFunction = new real[ 4 * 256 ];
	if (!h_transferFunction) return false; ///< MR = 256 * 4 * (real)

	for(int i = 0; i < 256; i++) {

		if (i < 1*(256/4)) {
			c[0] = 1.0; c[1] = 0.0; c[2] = 0.0; c[3] = 0.2;
		} else if (i < 2*(256/4)) {
			c[0] = 1.0; c[1] = 1.0; c[2] = 0.0; c[3] = 0.2;
		} else if (i < 3*(256/4)) {
			c[0] = 0.0; c[1] = 1.0; c[2] = 1.0; c[3] = 0.2;
		} else if (i < 4*(256/4)) {
			c[0] = 0.0; c[1] = 0.0; c[2] = 1.0; c[3] = 0.2;
		}

		for (uint j = 0; j < 4; j++)
			h_transferFunction[i*4 + j] = c[j];

	}

	return true;

}

/// Write TF (transfer function)
/// @arg out output file stream
/// @return true if it succeed

bool writeTF(ofstream& out) {

	if (out.fail()) return false;

	if (!h_transferFunction) return false;

	out << 256 << endl;

	for(uint i = 0; i < 256; i++) {

		out << h_transferFunction[i*4 + 0] << " " << h_transferFunction[i*4 + 1] << " "
		    << h_transferFunction[i*4 + 2] << " " << h_transferFunction[i*4 + 3] << endl;

		if (out.fail()) return false;

	}

	out.close();

	return true;

}

/// Write TF (overload)
/// @arg f transfer function file name
/// @return true if it succeed
bool writeTF(const char* f) {

	ofstream out(f);

	return writeTF(out);

}

/// --- External Faces ---

/// Read ExtF (external faces)
/// @arg in input file stream
/// @return true if it succeed

bool readExtF(ifstream& in) {

	if (in.fail()) return false;

	uint nExtFaces;

	in >> nExtFaces;
	if (nExtFaces != numExtFaces) return false;

	if (h_extFaces) delete [] h_extFaces;
	h_extFaces = new uint[ 2 * numExtFaces ];
	if (!h_extFaces) return false; ///< MR = numExtFaces * 2 * (uint)

	for(uint i = 0; i < numExtFaces; ++i) {

		in >> h_extFaces[ i*2 + 0 ]
		   >> h_extFaces[ i*2 + 1 ];

		if (in.fail()) return false;

	}

	in.close();

	return true;

}

/// Read ExtF (overload)
/// @arg f external faces file name
/// @return true if it succeed
bool readExtF(const char* f) {

	ifstream in(f);

	return readExtF(in);

}

/// Build External Faces vector
/// @return true if it succeed

bool buildExtF(void) {

	if (h_extFaces) delete [] h_extFaces;
	h_extFaces = new uint[ 2 * numExtFaces ];
	if (!h_extFaces) return false; ///< MR = numExtFaces * 2 * (uint)

	uint extFacesId = 0;

	for (uint i = 0; i < numTets; ++i) { // tets

		for (uint f = 0; f < 4; ++f) { // faces

			/// discard non-external faces
			if (h_conTet[i*4+f] != i) continue;

			/// fill external faces vector
			h_extFaces[ extFacesId*2 + 0 ] = i;
			h_extFaces[ extFacesId*2 + 1 ] = f;

			extFacesId++; // external faces id

		}

	}

	return true;

}

/// Write ExtF (external faces)
/// @arg out output file stream
/// @return true if it succeed

bool writeExtF(ofstream& out) {

	if (out.fail()) return false;

	if (!h_extFaces) return false;

	out << numExtFaces << endl;

	for(uint i = 0; i < numExtFaces; i++) {

		out << h_extFaces[i*2 + 0] << " " << h_extFaces[i*2 + 1] << endl;

		if (out.fail()) return false;

	}

	out.close();

	return true;

}

/// Write ExtF (overload)
/// @arg f external faces file name
/// @return true if it succeed
bool writeExtF(const char* f) {

	ofstream out(f);

	return writeExtF(out);

}

/// --- Limits ---

/// Read Lmt (limits)
/// @arg in input file stream
/// @return true if it succeed

bool readLmt(ifstream& in) {

	if (in.fail()) return false;

	in >> h_maxEdgeLength >> h_maxZ >> h_minZ;

	in.close();

	return true;

}

/// Read Lmt (overload)
/// @arg f limits file name
/// @return true if it succeed
bool readLmt(const char* f) {

	ifstream in(f);

	return readLmt(in);

}

/// Build Limits
/// @return true if it succeed

bool buildLmt(void) {

	h_maxEdgeLength = 0.0;

	for (uint i = 0; i < numTets; ++i) { // for each tet

		real edge[6][3];

		for (uint k = 0; k < 3; ++k) { // for each coordinate

			edge[0][k] = h_vertList[h_tetList[i*4+0]*4 + k] - h_vertList[h_tetList[i*4+1]*4 + k];
			edge[1][k] = h_vertList[h_tetList[i*4+0]*4 + k] - h_vertList[h_tetList[i*4+2]*4 + k];
			edge[2][k] = h_vertList[h_tetList[i*4+0]*4 + k] - h_vertList[h_tetList[i*4+3]*4 + k];
			edge[3][k] = h_vertList[h_tetList[i*4+1]*4 + k] - h_vertList[h_tetList[i*4+2]*4 + k];
			edge[4][k] = h_vertList[h_tetList[i*4+1]*4 + k] - h_vertList[h_tetList[i*4+3]*4 + k];
			edge[5][k] = h_vertList[h_tetList[i*4+2]*4 + k] - h_vertList[h_tetList[i*4+3]*4 + k];

		}

		for (uint j = 0; j < 6; ++j) { // for each edge

			real len = sqrt( edge[j][0] * edge[j][0] + edge[j][1] * edge[j][1] + edge[j][2] * edge[j][2] );

			if (len > h_maxEdgeLength)
				h_maxEdgeLength = len;

		}

	}

	h_maxZ = -1.0;
	h_minZ = 1.0;

	for (uint i = 0; i < numVerts; ++i) { // vertices

		if ( h_vertList[ i*4 + 2 ] > h_maxZ )
			h_maxZ = h_vertList[ i*4 + 2 ];

		if ( h_vertList[ i*4 + 2 ] < h_minZ )
			h_minZ = h_vertList[ i*4 + 2 ];

	}

	return true;

}

/// Write Lmt (external faces)
/// @arg out output file stream
/// @return true if it succeed

bool writeLmt(ofstream& out) {

	if (out.fail()) return false;

	out << h_maxEdgeLength << " " << h_maxZ << " " << h_minZ << endl;

	out.close();

	return true;

}

/// Write Lmt (overload)
/// @arg f limits file name
/// @return true if it succeed
bool writeLmt(const char* f) {

	ofstream out(f);

	return writeLmt(out);

}

/// --- Other calls ---

/// Rotate entire volume
/// @arg modelView matrix
extern "C" void
rotateVolume(real *modelView) {

	/// Rotate volume vertices
	for (uint i = 0; i < numVerts; i++) {

		real vOrig[4] = { h_vertList[ i*4 + 0 ], h_vertList[ i*4 + 1 ],
				  h_vertList[ i*4 + 2 ], 1.0 };

		real vRot[4] = { 0.0, 0.0, 0.0, 0.0 };

		for (uint r = 0; r < 4; r++)
			for (uint c = 0; c < 4; c++)
				vRot[r] += modelView[ r*4 + c ] * vOrig[c];

		for (uint j = 0; j < 3; j++)
			h_vertList[ i*4 + j ] = vRot[j];

	}

}

/// Write a ppm image
/// @arg fn image file name
/// @arg fb image pixels (frame buffer)
/// @arg dimX x-dimension (width)
/// @arg dimY y-dimension (height)
/// @return true if it succeed

bool
ppmWrite(const char *fn, const uchar *fb, uint dimX, uint dimY) {

	FILE *f = fopen( fn, "w" );

	if( f == NULL ) {
		cerr << "Could not save ppm file." << endl;
		return false;
	}

	fprintf( f, "P6\n" );
	fprintf( f, "%u %u\n", dimX, dimY );
	fprintf( f, "255\n" );
	fwrite( fb, 1, dimX*dimY*3, f );

	fclose(f);

	return true;

}
