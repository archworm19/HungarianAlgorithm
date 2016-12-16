/*
 

Pairs kmeans of contours with predicted points of worms

n = number of worms paired with given contour 
(1) iterate through pairings --> k = 5 * n
    run kmeans 
(2) use the hungarian algorithm to pair kmeans with predictions (use -dist and max_cost_assignment)   

*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <stdio.h>
#include <iostream>
#include "parameters2.h"
#include <algorithm>
#include "hungary_pair.h" 


using namespace std;
using namespace cv;


// -=-=-=-=-=-=-=-=-=-==- KMeans Analysis -=-=-=-=-=-=-=-=-

void mat_to_vecvec(Mat& m, vector< vector<double> >& vv){
    int i;
    for(i=0;i<vv.size();i++){
        vv[i][0] = (double)m.at<float>(i,0);
        vv[i][1] = (double)m.at<float>(i,1);
    }
}


Mat vec_to_mat(vector<Point>& in_vec){
    Mat ret(in_vec.size(), 2, CV_32FC1);
    int i;
    for(i=0; i<in_vec.size(); i++){
        ret.at<float>(i,0) = (float)in_vec[i].x;
        ret.at<float>(i,1) = (float)in_vec[i].y;  
    }
    return ret; 
}


// returns: (min_pt, max_pt)
vector<Point> find_extremes(vector<Point>& contour){
    int i; int min_x = contour[0].x; int min_y = contour[0].y;
    int max_x = contour[0].x; int max_y = contour[0].y;
    
    for(i=1; i<contour.size(); i++){
        if(contour[i].x < min_x){ min_x = contour[i].x;}
        if(contour[i].y < min_y){ min_y = contour[i].y;}
        if(contour[i].x > max_x){ max_x = contour[i].x;}
        if(contour[i].y > max_y){ max_y = contour[i].y;}   
    }

    vector<Point> ret;
    ret.push_back(Point(min_x,min_y)); ret.push_back(Point(max_x,max_y));
    return ret; 
}


void translate_pts(vector<Point>& contour, Point min_pt){
    int i;
    for(i=0; i<contour.size(); i++){
        contour[i] = contour[i] - min_pt; 
    } 
}


// find ones and translate back using minimum
Mat find_ones(Mat drawn_cont, Point min_pt){
    vector< Point > hold_pts;
    int i; int j; uchar * p;
    for(i=0; i < drawn_cont.rows; i++){
        p = drawn_cont.ptr<uchar>(i);
        for(j=0; j < drawn_cont.cols; j++){
            if(p[j] == 255){
                hold_pts.push_back(Point(j,i) + min_pt);  
            } 
        }
    } 
    // convert to mat:
    return vec_to_mat(hold_pts);     
}   


// contour filtering: re-draw contour to get all points associated
Mat contour_filtering(vector< Point > contour){
    // find min_x, min_y --> subtract by this pt  

    vector<Point> extremes = find_extremes(contour);

    // Expand the extremes (bug handling):
    extremes[0].x = extremes[0].x - 1;
    extremes[0].y = extremes[0].y - 1;
    extremes[1].x = extremes[1].x + 1; 
    extremes[1].y = extremes[1].y + 1; 

    translate_pts(contour, extremes[0]); 

    // draw contour:
    Mat holder(extremes[1].y - extremes[0].y, extremes[1].x - extremes[0].x, CV_8UC1);

    Scalar color = Scalar(255);
    vector< vector<Point> > hold_conts; hold_conts.push_back(contour); 
    drawContours(holder, hold_conts, 0, color, -1);  

    // testing:
    //imshow("holder", holder);
    //waitKey();

    // search drawn contour for 1s --> output as Mat and translate  
    return find_ones(holder, extremes[0]);   
}   


// Keep repeating pts until we get >= K of them 
Mat fill_out(Mat& true_cont_pts, int K){
    int num_reps = (int) ceil((double)K / (double)true_cont_pts.rows);
    Mat padded_pts(true_cont_pts.rows * num_reps, 2, CV_32FC1);
    int i; int j;
    for(i=0; i<num_reps; i++){
        for(j=0; j<true_cont_pts.rows; j++){
            padded_pts.at<float>(i*true_cont_pts.rows + j, 0) = true_cont_pts.at<float>(j,0);
            padded_pts.at<float>(i*true_cont_pts.rows + j, 1) = true_cont_pts.at<float>(j,1);

        }
    } 
    return padded_pts; 
} 


// TODO: if all_cont_pts has no pts and contours still has pts (maybe, was filled initially or something) --> add the original contours
// pts back in...make sense?...or, can we fix the contour_filtering method?     


// run kmeans to fit the worm
void worm_fit(vector< Point >& contour, vector< vector<double> >& km_verts, int K){
    // contour filtering:
    Mat all_cont_pts = contour_filtering(contour);

    // TODO: requirement: N (number of pts) > K     
    // one possible method    
    if(all_cont_pts.rows < K){
        all_cont_pts = fill_out(all_cont_pts, K);   
    }

    // run kmeans:
    Mat bestLabels; Mat centers;
    kmeans(all_cont_pts, K, bestLabels, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    // convert mat to vector of vectors:
    mat_to_vecvec(centers, km_verts);
}


// run kmeans for each relevant contour
// cont_structs = contour structure = the kmeans
void get_structure(vector< vector<Point> >& contours, vector<int>& collisions, vector< vector< vector<double> > >& cont_structs){

    int i; int K;
    for(i=0; i<contours.size(); i++){
        if(collisions[i] > 0){
            K = NUM_PTS_PER_WORM*collisions[i];

            vector< vector<double> > km_verts(K, vector<double>(2,0));  
            worm_fit(contours[i], km_verts, K); 

            cont_structs.push_back(km_verts); 
        }
        else{
            // add an empty vector:
            cont_structs.push_back(vector< vector<double> >(0, vector<double>(0,0)));  // TODO: figure out whether this will work
        }
    }
}   

// -=-=-=-=-=-=-=-----=-=-=-=-=-=-=-==-=-=-=


// -=-=-=-=-=-=-=-=- Point Pairing -=-=-=-==-

int find_closest(double sumx, double sumy, vector< vector<double> >& contour_struct, vector<int>& added){
    int i;
    double min_dist = -1.0; double cur_dist; int min_ind; 
    for(i=0; i<contour_struct.size(); i++){
        cur_dist = sqrt(pow(contour_struct[i][0] - sumx, 2.0) + pow(contour_struct[i][1] - sumy, 2.0)); 
        if(added[i] == 1){ continue; }
        if(min_dist < 0.0 || cur_dist < min_dist){
            min_dist = cur_dist;
            min_ind = i; 
        } 
    }
    return min_ind; 
}


// TODO: subset selection
// Possibility: pred_pts is smaller than contour_struct
// find n closest pts to mid pt of pred_pts (n = size of pred_pts)
void subset_selection(vector< vector<double> >& pred_pts, vector< vector<double> >& contour_struct, vector< vector<double> >& contour_sub){
    // calc mid_pt of pred_pts:
    int i; double sumx = 0.0; double sumy = 0.0;
    for(i=0; i<pred_pts.size(); i++){
        sumx = sumx + pred_pts[i][0]; 
        sumy = sumy + pred_pts[i][1];
    }
    sumx = sumx / pred_pts.size(); sumy = sumy / pred_pts.size(); 

    vector<int> added(contour_struct.size(), 0); 

    // find the n closest pts:
    int j; int close_ind; 
    for(i=0; i<pred_pts.size(); i++){
        // find closest:
        close_ind = find_closest(sumx, sumy, contour_struct, added); 

        contour_sub.push_back(contour_struct[close_ind]); 
        added[close_ind] = 1; 
    }
}


// pair the points
// creates cost matrix --> runs max_cost_assignment:
vector<int> pair_pts(vector< vector<double> >& pred_pts, vector< vector<double> >& contour_struct_full){

    // subset selection:
    vector< vector<double> > contour_struct; 
    subset_selection(pred_pts, contour_struct_full, contour_struct);

    // create cost matrix:
    vector< vector<long> > cost; 
    build_dist_mat(pred_pts, contour_struct, cost, -1.0); 

    // max cost assignment:
    vector<int> rev_assigns = hungary_master(cost); 

    // reverse the assignments:
    int i; vector<int> assigns(rev_assigns.size()); 
    for(i=0; i<rev_assigns.size(); i++){
        assigns[rev_assigns[i]] = i;
    }
    return assigns; 
} 


// pair worms == pair the points for each worm:
void pair_worm_pts(vector< vector< vector<double> > >& pred_pts, vector< vector< vector<double> > >& contour_structs, vector<int>& pred_assign, vector< vector<int> >& ret_inds){ 
    int i;
    for(i=0; i<pred_assign.size(); i++){
    
        if(pred_assign[i] < 0){ continue; } 

        vector<int> assigns;
        assigns = pair_pts(pred_pts[i], contour_structs[pred_assign[i]]);
        ret_inds.push_back(assigns);
    }
}


// -=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-

vector<double> calc_mean_struct(vector< vector<double> >& cont_struct){
    vector<double> mean_pt(2,0);
    int i;
    for(i=0; i<cont_struct.size(); i++){
        mean_pt[0] = (i*mean_pt[0] + cont_struct[i][0])/(i+1);
        mean_pt[1] = (i*mean_pt[1] + cont_struct[i][1])/(i+1);
    }
    return mean_pt; 
}


vector<double> vect_sub(vector<double>& pt1, vector<double>& pt2){
    int i; vector<double> ret(pt1.size());
    for(i=0; i<pt1.size(); i++){
        ret[i] = pt1[i] - pt2[i]; 
    }
    return ret; 
}


void make_offsets(vector< vector<double> >& cont_struct, vector<double>& mean_pt, vector< vector<double> >& cont_offs){
    int i;
    for(i=0; i<cont_struct.size(); i++){
        cont_offs.push_back(vect_sub(cont_struct[i],mean_pt));  
    }
} 


// TODO: convert contour structures into offsets:
void convert_to_offsets(vector< vector< vector<double> > >& cont_structs, vector< vector< vector<double> > >& cont_offsets){
    int i; vector<double> cur_mean;
    for(i=0; i<cont_structs.size(); i++){
        
        // calculate the mean of the struct: 
        cur_mean = calc_mean_struct(cont_structs[i]);
        
        // create offsets
        vector< vector<double> > struct_offs;
        make_offsets(cont_structs[i], cur_mean, struct_offs); 
        cont_offsets.push_back(struct_offs); 
    }
}  


// write assigned_pts:
void write_pts(vector< vector<int> >& pt_assignments, vector<int>& pred_assigns, vector< vector< vector<double> > >& contour_structs, vector< vector< vector<double> > >& write_pts){
    int i; int j;
    for(i=0; i<pred_assigns.size(); i++){
        vector< vector<double> > inner_write;

        // iterate through the pts in pt_assignment: 
        for(j=0; j < pt_assignments[i].size(); j++){

            vector<double> new_pt(2);
            new_pt[0] = contour_structs[pred_assigns[i]][pt_assignments[i][j]][0];
            new_pt[1] = contour_structs[pred_assigns[i]][pt_assignments[i][j]][1];

            inner_write.push_back(new_pt);
        } 
        write_pts.push_back(inner_write); 
    } 

}  




// -=-=-=-=-=-=-=- MASTER FUNCTION -=-=-=-=-=

// TODO: now recieves pred_pts_offsets (offset from mean) 

// find the best pairings for each prediction (1 pred per worm) 
void master_func_pair_pts(vector< vector< vector<double> > >& pred_pts_offset, vector< vector<Point> >& contours, vector<int>& collisions, vector<int>& pred_assigns, vector< vector< vector<double> > >& writes){
 
    // get the kmeans for each relevant contour:
    vector< vector< vector<double> > > contour_structures; 
    get_structure(contours, collisions, contour_structures);

    // convert contour_structures into offsets from mean     
    vector< vector< vector<double> > > cs_offsets; 
    convert_to_offsets(contour_structures, cs_offsets);
    
    // find the index pairings for each of the predictions
    vector< vector<int> > pt_assignments;
    pair_worm_pts(pred_pts_offset, cs_offsets, pred_assigns, pt_assignments);

    // write the output pts:
    write_pts(pt_assignments, pred_assigns, contour_structures, writes);
}



// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


