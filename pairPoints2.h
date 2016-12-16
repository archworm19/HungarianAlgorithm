#ifndef PAIRPOINTS
#define PAIRPOINTS

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include "parameters2.h"
#include <algorithm>
#include "hungary_pair.h" 

using namespace std;
using namespace cv;

vector<double> calc_mean_struct(vector< vector<double> >& cont_struct);


void make_offsets(vector< vector<double> >& cont_struct, vector<double>& mean_pt, vector< vector<double> >& cont_offs);


// find the best pairings for each prediction (1 pred per worm) 
void master_func_pair_pts(vector< vector< vector<double> > >& pred_pts_offset, vector< vector<Point> >& contours, vector<int>& collisions, vector<int>& pred_assigns, vector< vector< vector<double> > >& writes);

#endif
