
#include "KHCore.h"
using namespace std;

void save(vector<int> &core, string path, string name) {
	path = path+name;
	FILE *fout = fopen(path.c_str(), "wb");
	int n = (int) core.size();
	printf("path=%s, n=%d\n", path.c_str(), n);
	fwrite(&n, sizeof(int), 1, fout);
	for(int i = 0; i < n; ++i) {
		int val = core[i];
		fwrite(&val, sizeof(int), 1, fout);
	}
	fclose(fout);
}

bool load(string path, string name, vector<int> &core) {
	int n;
	path = path+name;
	FILE *fin = fopen(path.c_str(), "rb");
	if( fin == NULL ) return false;
	fread(&n, sizeof(int), 1, fin);
	core.clear();
	for(int i = 0; i < n; ++i) {
		int val;
		fread(&val, sizeof(int), 1, fin);
		core.push_back(val);
	}
	fclose(fin);
	return true;
}

void compare(string path, string name1, string name2) {
	vector<int> core1, core2;

	load(path, name1, core1);
	load(path, name2, core2);

	double avg_error = 0;
	int cnt_error = 0;
	int max_core1 = -1, max_core2 = -1;

	int n = (int)core1.size();
	if( n != (int) core2.size() ) printf( "n1!=n2\n" );
	for( int i = 0; i < n; ++i ) {
		if( core1[i] != core2[i] ) {
			++cnt_error;
			avg_error += fabs((core2[i] - core1[i])*1.0)/core1[i];
		}
		max_core1 = max_core1 < core1[i] ? core1[i] : max_core1;
		max_core2 = max_core2 < core2[i] ? core2[i] : max_core2;
	}
	avg_error /= n;
	printf( "cnt_error=%d/%d,avg_error=%0.4lf%%\n", cnt_error, n, avg_error*100.0 );
	printf( "max core:accurate/sampling=%d/%d\n", max_core1, max_core2 );
}

int main(int argc, char *argv[]) {
	// printf( "argc=%d\n", argc );
	// for( int i = 0; i < argc; ++i )
	// 	printf( "argv[%d]=%s\n", i, argv[i] );

	setvbuf(stdout, NULL, _IONBF, 0);
	setvbuf(stderr, NULL, _IONBF, 0);
	double t = omp_get_wtime();

	if( argc > 1 ) {
		if(strcmp(argv[1], "txt-to-bin") == 0)
			KHCore::create_bin( /*dataset*/argv[2], /*output_path*/argv[3] );
		else if(strcmp(argv[1], "decompose") == 0) {
			vector<int> core;
			KHCore khcore(/*dataset*/ argv[2]);
			khcore.decompose(/*h*/ atoi(argv[3]),
					/*n_threads*/argc>4?atoi(argv[4]):1,
					/*sample_rate*/argc>5?atof(argv[5]):1.0,
					/*error_rate*/argc>6?atof(argv[6]):0, core);
			if(argc>7) save(core, argv[7], argv[8]); 
		} else if(strcmp(argv[1], "compare") == 0) {
			compare(argv[2], argv[3], argv[4]);
		} else if (strcmp(argv[1], "scal") == 0){
			if (argc < 6) { printf("error parameters, usage: ./run scal rate vary path h threads <samp-rate> <err-rate>\n"); abort();
			}
			double scal = atof(argv[2]); int vary = atoi(argv[3]);
			vector<int> core;
			KHCore khcore(scal, vary, /*dataset*/ argv[4]);
			khcore.decompose(/*h*/ atoi(argv[5]),
					/*n_threads*/argc>6?atoi(argv[6]):1,
					/*sample_rate*/argc>7?atof(argv[7]):1.0,
					/*error_rate*/argc>8?atof(argv[8]):0, core);
		}else {
			printf("please select: txt-to-bin, decompose, compare, scal, or casestudy\n");
			abort();
		}
	}

	t = omp_get_wtime() - t;
	// printf( "Total time=%0.3lf msecs\n", t * 1000);

	return 0;
}
