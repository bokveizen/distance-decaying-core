
#ifndef KHCORE_H_
#define KHCORE_H_

#include <omp.h>
#include <string>
#include <vector>
#include <cstring>
#include <algorithm>
#include <utility>
#include <cstdio>
#include <cmath>
#include <xmmintrin.h>
#include <stdint.h>

using namespace std;

#define MAXST 1024
#define one ((unsigned long long)1)


class KHCore{
public:
	int n;
	long long m;
	int *deg, *dat, **adj;

	double graph_memory, maximum_memory;

	KHCore(string path);
	KHCore(double sacl, int vary, string path); //scalability
	~KHCore();

	static void create_bin(string input_path, string output_path);
	static bool get_edge(char *line, int &a, int &b, int num_cnt);
	static int get_num_cnt(string path);

	int init(vector<int> &core, vector<int> &cnt_select, int h, vector<bool> &selecte);
	void decompose(int h, int n_threads, double sample_rate, double error_rate, vector<int> &core);

	void casestudy(string core, string authors, string name);
};

//======implementation========

KHCore::KHCore(string path) {
	// printf( "path=%s\n", path.c_str() );
	FILE* fin = fopen( (path/*+"graph.bin"*/).c_str(), "rb" );
	fread( &n, sizeof(int), 1, fin );
	fread( &m, sizeof(long long), 1, fin );
	deg = new int[n]; dat = new int[m]; adj = new int*[n];

	// printf( "Loading graph...\n" );
	fread( deg, sizeof(int), n, fin );
	fread( dat, sizeof(int), m, fin );
	fclose(fin);

	int max_deg = 0;
	for( int i = 0; i < n; ++i ) max_deg = max(max_deg, deg[i]);
	// printf( "n=%d,m=%lld,max_deg=%d\n", n, m / 2, max_deg );

	long long p = 0;
	for( int i = 0; i < n; ++i ) {
		adj[i] = dat+p; p+=deg[i];
	}
	graph_memory = (double) sizeof(int) * (n * 2 + m) / 1024.0; maximum_memory = graph_memory;
}

KHCore::KHCore(double scal, int vary, string path) {
	printf( "path=%s\n", path.c_str() );
	FILE* fin = fopen( (path/*+"graph.bin"*/).c_str(), "rb" );
	fread( &n, sizeof(int), 1, fin );
	fread( &m, sizeof(long long), 1, fin );
	deg = new int[n]; dat = new int[m]; adj = new int*[n];

	printf( "Loading graph...\n" );
	fread( deg, sizeof(int), n, fin );
	fread( dat, sizeof(int), m, fin );
	fclose(fin);

	int max_deg = 0;
	for( int i = 0; i < n; ++i ) max_deg = max(max_deg, deg[i]);
	printf( "n=%d,m=%lld,max_deg=%d\n", n, m / 2, max_deg );

	long long p = 0;
	for( int i = 0; i < n; ++i ) {
		adj[i] = dat+p; p+=deg[i];
	}
	graph_memory = (double) sizeof(int) * (n * 2 + m) / 1024.0; maximum_memory = graph_memory;

	srand(100);
	int v_num = 0; vector <int> selected(n, -1), adj_s(n, 0);
	printf("scal rate=%.2f, ", scal);
	switch (vary){
		case 1:
			// vary v
			m = 0; v_num = 0; p = 0;
			for (int i = 0; i < n; ++i){
				if (rand() < int ( RAND_MAX * scal + 1e-8) )
					selected[i] = v_num++;
			}
			for (int i = 0; i < n; ++i){
				if (selected[i] < 0)
					continue;
				int u; long long k = p;
				for (int j = 0; j < deg[i]; ++j){
					u = adj[i][j];
					if (selected[u] >= 0)
						dat[p++] = selected[u];
				}
				deg[selected[i]] = p - k;
			}
			n = v_num; p = 0;
			for (int i = 0; i < n; ++i){
				adj[i] = dat + p;
				p += deg[i];
			}
			m = p;
			printf("vary v\n");
			break;
		case 2:
			// vary m
			m = 0; v_num = 0; 
			for (int i = 0; i < n; ++i){
				int k = 0;
				for (int j = 0; j < deg[i]; ++j){
					int u = adj[i][j];
					if ( i > u || rand() >= int ( RAND_MAX * scal + 1e-8) )
						continue;
					adj[i][adj_s[i]++] = u;
					adj[u][adj_s[u]++] = i;
				}
				deg[i] = adj_s[i];
				m += deg[i];
			}
			printf("vary m\n");
			break;
		default: printf("Invalid parameter: 'vary' is only selected from 1 or 2\n"); abort(); break;
	}

	printf("scal n=%d, m=%lld\n", n, m / 2);
	srand(1);
}

KHCore::~KHCore() {
	delete[] deg; delete[] dat; delete[] adj;
}


bool KHCore::get_edge(char *line, int &a, int &b, int num_cnt) {
	if( !isdigit(line[0]) ) return false;
	vector<char*> v_num;
	int len = (int) strlen(line);
	for( int i = 0; i < len; ++i )
		if( !isdigit(line[i]) && line[i] != '.') line[i] = '\0';
		else if(i == 0 || !line[i-1]) v_num.push_back(line+i);
	if( (int) v_num.size() != num_cnt ) return false;
	sscanf( v_num[0], "%d", &a );
	sscanf( v_num[1], "%d", &b );
	return true;
}

int KHCore::get_num_cnt(string path) {
	FILE *fin = fopen( (path/*+ "graph.txt"*/).c_str(), "r" );
	char line[MAXST];
	int cnt = 0, min_cnt = 100;

	while( fgets( line, MAXST, fin ) && cnt < 10 ) {
		if( !isdigit(line[0]) ) continue;
		vector<char*> v_num;
		int len = (int) strlen(line);
		for( int i = 0; i < len; ++i )
			if( !isdigit(line[i]) && line[i] != '.' ) line[i] = '\0';
			else if(i == 0 || !line[i-1]) v_num.push_back(line+i);
		if( (int) v_num.size() < 2 ) continue;
		min_cnt = min(min_cnt, (int) v_num.size());
		++cnt;
	}
	fclose( fin );
	return min_cnt;
}


void KHCore::create_bin(string input_path, string output_path) {
	FILE *fin = fopen( (input_path).c_str(), "r" );
	char line[MAXST];
	int n = 0, a, b, num_cnt = get_num_cnt(input_path);
	vector< pair<int,int> > el;
	long long cnt = 0, m = 0;

	printf( "Loading text, num_cnt=%d...\n", num_cnt );
	while( fgets( line, MAXST, fin ) ) {
		if( !get_edge(line, a, b, num_cnt) ) continue;
		if( a < 0 || b < 0 || a == b ) continue;
		el.push_back(make_pair(a, b));
		n = max(max(n, a+1), b+1);
		if( (++cnt) % (long long) 10000000 == 0 ) printf( "%lld lines finished\n", cnt );
	}
	fclose( fin );

	vector<int> *con = new vector<int>[n];
	printf( "Deduplicating...\n" );

	for(size_t i = 0; i < el.size(); ++i) {
		con[el[i].first].push_back(el[i].second);
		con[el[i].second].push_back(el[i].first);
	}

	for( int i = 0; i < n; ++i )
		if( con[i].size() > 0 ){
			sort( con[i].begin(), con[i].end() );
			int p = 1;
			for( int j = 1; j < (int) con[i].size(); ++j )
				if( con[i][j-1] != con[i][j] ) con[i][p++] = con[i][j];
			con[i].resize( p ); m += p;
		}

	int *dat = new int[m], *deg = new int[n], **adj = new int *[n];

	long long pos = 0;
	for( int i = 0; i < n; ++i ) {
		adj[i] = dat + pos;
		pos += (int) con[i].size();
	}
	memset( deg, 0, sizeof(int) * n );

	for( int i = 0; i < n; ++i )
		for( int p = 0; p < (int) con[i].size(); ++p )
			adj[i][deg[i]++] = con[i][p];

	printf( "Saving binary...\n" );

	FILE *fout = fopen( (output_path).c_str(), "wb" );
	fwrite( &n, sizeof(int), 1, fout );
	fwrite( &m, sizeof(long long), 1, fout );
	fwrite( deg, sizeof(int), n, fout );
	fwrite( dat, sizeof(int), m, fout );

	fclose( fout );
	printf( "Created binary file, n = %d, m = %lld\n", n, m );


	delete[] adj; delete[] deg; delete[] dat; delete[] con;
}

int KHCore::init(vector<int> &core, vector<int> &cnt_select, int h, vector<bool> &selected) {
	int max_val = 0;
	#pragma omp parallel
	{
		int pid = omp_get_thread_num(), np = omp_get_num_threads();
		// if( pid == 0 ) printf( "init: n_thread = %d\n", np );
		vector<int> q;
		vector<bool> used(n,false);

		#pragma omp for schedule(dynamic) reduction(max:max_val)
		for( int v = 0; v < n; ++v ) {
			q.clear(); q.push_back(v); used[v] = true;
			size_t pre_p = 0, now_p = 1;
			int cnt = 0;

			for( int hop = 0; hop < h; ++hop ) {
				for( size_t i = pre_p; i < now_p; ++i ) {
					int u = q[i];
					for( int j = 0; j < deg[u]; ++j ) {
						int w = adj[u][j];
						if( !used[w] ) {
							used[w] = true; q.push_back(w);
							if( selected[w] ) ++cnt;
						}
					}
				}
				pre_p = now_p; now_p = q.size();
			}

			core[v] = (int) q.size()-1;
			cnt_select[v] = cnt;
			max_val = max(core[v], max_val);
			for( size_t i = 0; i < q.size(); ++i )
				used[q[i]] = false;
		}
	}

	//for( int v = 0; v < n; ++v ) max_val = max(max_val, core[v]);
	// printf( "Init finished, maximum init_core = %d\n", max_val );
	//for (int i = 0; i < n; ++i)
	//	printf("%d\t%d\n",i ,core[i]);
	return max_val;
}

void KHCore::decompose(int h, int n_threads, double sample_rate, double error_rate, vector<int> &core) {
	omp_set_num_threads(n_threads);
	double t_start = omp_get_wtime();

	vector<bool> selected(n, false);
	vector<int> num_t_comp(n_threads, 0);

	for(int v = 0; v < n; ++v){
		if (deg[v] <= 0) continue;
		if( rand() < (int) (RAND_MAX * sample_rate + 1e-8) || sample_rate > 1.0-(1e-8) )
			selected[v] = true;
	}
	vector<int> cnt_select(n, 0);
	core.clear();
	for(int v = 0; v < n; ++v)
		core.push_back(0);
	vector<double> rate(n);

	int max_core = init(core, cnt_select, h, selected);
	// printf( "Init time=%0.3lf msecs\n", ( omp_get_wtime()-t_start ) * 1000 );
	for(int v = 0; v < n; ++v) rate[v] = cnt_select[v] * 1.0 / core[v];

	vector<int> remove_time(n,n);
	int now_time = 0;

	vector<int> cnt_remove(n,0);
	vector<unsigned long long> vll_empty;

	vector<int> v_left;
	vector<int> v_now;
	for(int v = 0; v < n; ++v)
		if(deg[v] > 0) v_left.push_back(v);

	int cnt_processed = 0;

	int **id_new_all = new int*[n_threads];
	for(int i = 0; i < n_threads; ++i) {
		id_new_all[i] = new int[n];
		memset(id_new_all[i], -1, sizeof(int) * n);
	}

	//maximum_memory = graph_memory;
	maximum_memory += ((double) (sizeof(int) *(n + n + n + n) + sizeof(bool) * n) / 1024.0);
	if (1.0 - sample_rate > 1e-8)
		maximum_memory += (double) sizeof(double) * n / 1024.0;

int core_min = n + 1;
int tm = 0, iteration = 0;
// printf("maximum mem=%.3f kb\n", maximum_memory);

#pragma omp parallel
{
	int pid = omp_get_thread_num();
	int *id_new = id_new_all[pid];
	vector<int> q;
	vector<int> local_cnt_remove;

	vector<int> hop_pos_select;
	vector<int> hop_pos;

	vector<vector<unsigned long long> > reach[2];

	vector<pair<int,int> > local_edge;

	for(;;) {

		#pragma omp single
		{
			v_now.clear();
			int p = 0;
			core_min = n + 1;
			for(size_t i = 0; i < v_left.size(); ++i) {
				int v = v_left[i];
				if(remove_time[v] == n) {
					core_min = min(core_min, core[v]);
					v_left[p++] = v;
				}
			}
			v_left.resize(p);

			if (v_left.size() > 0 && core_min < n + 1)
				max_core = core_min;

			for(size_t i = 0; i < v_left.size(); ++i) {
				int v = v_left[i];
				if(core[v] == core_min || (error_rate > 1e-8 && core[v] <= (int) core_min * (1+error_rate))){
					v_now.push_back(v);

					for (int j = 0; j < deg[v] && (1 - sample_rate) <= 1e-8; ++j)
					{
						int u = adj[v][j];
						if (cnt_select[u] == core_min + 1 && remove_time[u] == n)
						{
							core[u] == core_min;
							cnt_select[u] = core_min;
							v_now.push_back(u);
						}
					}
				}
			}

			//printf( "iteration=%d, core_min=%d, v_now.size=%d, v_left.size=%d\n", iteration++, core_min, (int) v_now.size(), (int) v_left.size() );
			//if(v_left.size() - v_now.size() <= core_min) break;
			for(size_t i = 0; i < v_now.size(); ++i) {
				int v = v_now[i];
				remove_time[v] = now_time++;
			}
		}

		if(v_left.size() - v_now.size() <= core_min){
			#pragma omp for
			for(size_t i = 0; i < v_left.size(); ++i)
				core[v_left[i]] = core_min;
			double mem = double (sizeof(int) * (q.capacity() + local_cnt_remove.capacity() + hop_pos_select.capacity() + hop_pos.capacity() 
				+ sizeof(unsigned long long ) * reach[0].capacity() * 2 + sizeof(pair<int,int>)* local_edge.capacity() )) / 1024.0;
			#pragma omp atomic
				maximum_memory += mem;
			break;
		}

		#pragma omp for schedule(dynamic)
		for( size_t i_now = 0; i_now < v_now.size(); ++i_now ) {
			int v0 = v_now[i_now];
			int time_v0 = remove_time[v0];
			int nown = 0, nows = 0;
			++num_t_comp[pid];

			q.clear();
			hop_pos.clear(); hop_pos.push_back(0);
			hop_pos_select.clear(); hop_pos_select.push_back(0);
			local_edge.clear(); local_cnt_remove.clear();
			int pre_p = -1, now_p = 0;
			for(int hop = 1; hop <= h; ++hop) {
				for(int i = pre_p; i < now_p; ++i) {
					int u = (i == -1) ? v0 : q[i];
					for(int j = 0; j < deg[u]; ++j) {
						int w = adj[u][j];
						if( remove_time[w] > time_v0 ) {
							if( id_new[w] == -1 ) {
								id_new[w] = nown++; q.push_back(w);
								if( hop < h && selected[w] ) ++nows;
							}
							if(i >= 0 && id_new[w] > i) local_edge.push_back(make_pair(i, id_new[w]));
						}
					}
				}
				pre_p = now_p; now_p = q.size();
				hop_pos_select.push_back(nows); hop_pos.push_back(now_p);
			}

			int nows_bit = (nows+63)/64;
			int cnt = 0;
			int pre = 0, now = 1;

			for(int i = 0; i < 2; ++i)
				while((int) reach[i].size() < nown) reach[i].push_back(vll_empty);

			for(int i = 0; i < nown; ++i) {
				for(int j = 0; j < 2; ++j) {
					reach[j][i].clear();
					for(int k = 0; k < nows_bit; ++k) reach[j][i].push_back((unsigned long long) 0);
				}
				if( i < hop_pos[h-1] && selected[q[i]] ) {
					reach[pre][i][cnt/64]|=one<<(cnt%64); reach[now][i][cnt/64]|=one<<(cnt%64); ++cnt;
				}
			}


			for(int hop = 1; hop <= h; ++hop) {
				now = pre; pre = 1-pre;
				for(size_t i = 0; i < local_edge.size(); ++i) {
					int u = local_edge[i].first, v = local_edge[i].second;
					for( int j = 0; j < nows_bit; ++j) {
						reach[now][v][j] |= reach[pre][u][j];
						reach[now][u][j] |= reach[pre][v][j];
					}
				}
			}

			for(int hop = 1; hop <= h; ++hop) {
				int maxs = hop_pos_select[h-hop];
				for( int p = hop_pos[hop-1]; p < hop_pos[hop]; ++p ) {
					cnt = selected[v0] ? 1 : 0;
					for(int pos = 0, i = 0; pos < maxs; ++i, pos += 64) {
						unsigned long long unreach = ~reach[now][p][i];
						if( unreach )
							for( int j = 0; j < 64 && pos + j < maxs; ++j )
								if( unreach & (one<<j) ){
									++cnt;
								}
					}
					local_cnt_remove.push_back(cnt);
				}
			}

			for(int i = 0; i < nown; ++i) id_new[q[i]] = -1;

			#pragma omp critical
			for(int i = 0; i < nown; ++i) cnt_remove[q[i]] += local_cnt_remove[i];

		}

		#pragma omp for reduction(+:cnt_processed)
		for(int i = 0; i < v_left.size(); ++i)
		{
			int v = v_left[i]; 
			if( cnt_remove[v] && core[v] > core_min ) {
				cnt_select[v] -= cnt_remove[v];
				cnt_processed += cnt_remove[v];
				cnt_remove[v] = 0;
				int now_core = rate[v] > 1.0-1e-8 ? cnt_select[v] : (int) (cnt_select[v] / rate[v] + 0.5);
				if( now_core < core_min ) now_core = core_min;
				core[v] = now_core;
			}
		}
	
		if( cnt_processed >= m * 4 ) {
			//printf( "Update graph: " );
			tm = 0;
			#pragma omp for reduction(+:tm)
			for(int u = 0; u < n; ++u) {
				int p = 0;
				for( int i = 0; i < deg[u]; ++i ) {
					int v = adj[u][i];
					if( remove_time[v] == n )
						adj[u][p++] = v;
				}
				deg[u] = p;
				tm += p;
			}
			cnt_processed = 0;
			m = tm;
			//printf( "m=%d, t=%0.3lf msecs\n", m, ( omp_get_wtime()-t_start ) * 1000 );
		}
		
	}
}

	for(int i = 0; i < n_threads; ++i) delete[] id_new_all[i];
	delete[] id_new_all;
	// printf( "Decomposition time=%0.3lf msecs\n", ( omp_get_wtime()-t_start ) * 1000 );
	// printf("graph_memory=%.3f kb, maximum memory=%.3f kb\n", graph_memory, maximum_memory);
	// if (1 - sample_rate < 1e-8)
	// 	printf("max_core=%d\n", max_core);
	// else{
	// 	int max_coren = 0; 
	// 	for (int i = 0; i < n; ++i) max_coren = max(max_coren, core[i]);
	// 		printf("max_core=%d\n", max_coren);
	// }
	//for (int i = 0; i < n_threads; ++i)
	//	printf("thread=%d, comp=%d\n",i,num_t_comp[i]);
	// printf("core distribution:\n");
	for (int i = 0; i < n; ++i)
		printf("%d %d\n",i, core[i]);
}

#endif /* KHCORE_H_ */
