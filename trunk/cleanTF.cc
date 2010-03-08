/**
 *   Clean Transfer Function File
 *
 * Using C++ -- February, 2008
 *
 */

#include <iostream>
#include <fstream>

using namespace std;

int
main(int argc, char** argv) {

	if ( argc != 3 ) {

		cerr << "use: " << argv[0]
		     << " old.off.tf new.tf" << endl
		     << "Where *.tf is the transfer function file." << endl;

		return 1;

	}

	cout << "Reading " << argv[1] << flush;

	ifstream in(argv[1]);

	if (in.fail()) return false;

	char buffer[255];

	cout << " . "  << flush;

	for (int l = 0; l < 7; ++l)
		in.getline(buffer, 200);

	int index;
	double tf[256][4];

	cout << " . "  << flush;

	for(int i = 0; i < 256; i++) {

		in >> index >> tf[i][0] >> tf[i][1]
		   >> tf[i][2] >> tf[i][3];

		if (in.fail()) return 0;

	}

	cout << " . "  << flush;

	in.close();

	cout << " done!"  << endl;

	cout << "Writing " << argv[2] << flush;

	ofstream out(argv[2]);

	if (out.fail()) return false;

	cout << " . "  << flush;

	out << 256 << endl;

	cout << " . "  << flush;

	for(int i = 0; i < 256; i++) {

		out << tf[i][0] << " " << tf[i][1] << " "
		    << tf[i][2] << " " << tf[i][3] << endl;

		if (out.fail()) return 0;

	}

	cout << " . "  << flush;

	out.close();

	cout << " done!"  << endl;

	return 0;

}
