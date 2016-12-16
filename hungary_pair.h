#ifndef HUNGARY
#define HUNGARY


#include <iostream>
#include <vector>
#include <cmath> 


using namespace std; 


vector<int> hungary_master(vector< vector<long> >& w);

void build_dist_mat(vector< vector<double> >& pts1, vector< vector<double> >& pts2, vector< vector<long> >& mat, double mult);

#endif
