/*

Hungarian Algorithm ~ Implemented From Subhash Suri's Notes


Data Structures: 
(1) labels: 2 vectors
(2) M: vector of inds (backwards indexing) and -1s for empty...ex: [1,2,...] means 0th node in Y matches to 1st node in X
(3) F: vector of free X's (unmatched), binary
(4) S: subset of X (non-binary, inds)
(5) T: subset of Y (non-binary, inds)
(6) T_id: subset of Y (binary) == used for quick-checking identity
(7) slack_S: vector of longs of size Y == proc: get new z (elem of X) --> compare (min) slack_S to slack_Z --> creates new slack_S


Cast everything to integer...avoid very small number issue 


*/

#include <iostream>
#include <vector>
#include <cmath> 


using namespace std; 


// -=-=-=-=-=-=-=-=-=-=- Printing -=-=-=-=-=-=-=-=-
template <typename T>
void print_vec(vector<T> v){
    int i;
    for(i=0; i<v.size(); i++){
        cout << v[i] << " ";
    }
    cout << "\n"; 
}


template <typename T>
void print_mat(vector< vector<T> > m){
    int i; int j;
    for(i=0; i<m.size(); i++){
        for(j=0; j<m[i].size(); j++){
            cout << m[i][j] << " ";
        }
        cout << "\n";
    }
}



// -=-=-=-=-=-=-=-=-=-=-=- Initialization -=-=-=-=-=-=-=-=-=-=-

// TODO: initial labeling (2 vectors)
void init_label(vector< vector<long> >& w, vector<long>& l_X, vector<long>& l_Y){
    l_Y = vector<long>(w[0].size(), 0.0);
    l_X = vector<long>(w.size());
    int i; int j; long max; 
    for(i=0; i<w.size(); i++){
        max = w[i][0];
        for(j=1; j<w[i].size(); j++){
            if(w[i][j] > max){ max = w[i][j]; }
        }
        l_X[i] = max; 
    }
}


// TODO: intial matching (empty) 
void init_match(vector< vector<long> >& w, vector<int>& M){
    M = vector<int>(w[0].size(), -1);
}


// initialize free...everybody if free
void init_F(vector< vector<long> >& w, vector<int>& M, vector<int>& F){
    F = vector<int>(w.size(), 1);
    int i;
    for(i=0; i<M.size(); i++){
        if(M[i]>=0){
            F[M[i]] = 0; 
        }
    }    
}


// -=-=-=-=-=-=-=-=-=-=-=- Phase Initialization -=-=-=-=-=-=-=-=-

// TODO: Initialize S, T, T_id
// iterate through F (free vertices) --> add to S
void init_ST(vector< vector<long> >& w, vector<int>& S, vector<int>& T, vector<int>& T_id, vector<int>& F){
    S = vector<int>(); 
    T = vector<int>();
    T_id = vector<int>(w[0].size(), 0); 
    int i;
    for(i=0; i<F.size(); i++){
        if(F[i] == 1){
            S.push_back(i);     
            return;  
        }
    }
}


// TODO: initialize slack_S for first vertex in S (u) == vector of l(u) + l(y) - w(u,y) 
void init_slack(vector< vector<long> >& w, vector<long>& l_X, vector<long>& l_Y, vector<long>& slack_S, int s){
    slack_S = vector<long>(l_Y.size());
    int i;
    for(i=0; i<slack_S.size(); i++){
        slack_S[i] = l_X[s] + l_Y[i] - w[s][i]; 
    }  
}



// -=-=-=-=-=-=-=-=-=-=-=- Update Labels -=-=-=-=-=-=-=-=-=-=-

// TODO: does Nl(S) == T? returns 1 if yes
// Idea: use slack_S == go through slack_S --> is dist == 0 + not in T...
int Nl_T_check(vector<long>& slack_S, vector<int>& T_id){
    int i;
    for(i=0; i<slack_S.size(); i++){
        if(slack_S[i] == 0 && T_id[i] == 0){ return 0; }
    }
    return 1; 
}



// TODO: calculate dl:
// find min of slack_S, ensuring the considered elems are NOT in T (use T_id)
long calc_dl(vector<long>& slack_S, vector<int>& T_id){
    int i; vector<long> min; 
    for(i=0; i<slack_S.size(); i++){
        if(T_id[i] == 0){
            if(min.size() == 0){ min.push_back(slack_S[i]); continue; }
            if(slack_S[i] < min[0]){ min[0] = slack_S[i]; } 
        }
    }
    return min[0];
}


// TODO: update the labels:
// iterate through update values according to rule and dl
void update_labels(vector<long>& l_X, vector<long>& l_Y, vector<int>& S, vector<int>& T, long dl){
    int i;
    
    // l_X:
    for(i=0; i<S.size(); i++){
        l_X[S[i]] = l_X[S[i]] - dl;
    }   

    // l_Y:
    for(i=0; i<T.size(); i++){
        l_Y[T[i]] = l_Y[T[i]] + dl; 
    }
}



// update slack given inclusion of new S node
// TODO: update slack_S: 
// Note: z = index of node just added to S
void update_slack(vector< vector<long> >& w, vector<long>& l_X, vector<long>& l_Y, vector<long>& slack_S, int z){

    // create slack_z (for new node)
    vector<long> slack_z; 
    init_slack(w, l_X, l_Y, slack_z, z);

    // pair-wise min comparison between slack_S and slack z:
    int i;
    for(i=0; i<slack_S.size(); i++){
        if(slack_S[i] > slack_z[i]){ slack_S[i] = slack_z[i]; }  
    }
}



// TODO: update slack_S given relabeling/dl:
// iterate through T_id --> if not in T_id --> decrement slack location by dl
void update_slack_relabel(vector<long>& slack_S, vector<int>& T_id, long dl){
    int i;
    for(i=0; i<T_id.size(); i++){
        if(T_id[i] == 0){
            slack_S[i] = slack_S[i] - dl; 
        }
    }
}




// -=-=-=-=-=-=-=-=-=-=-=- Augmentation -=-=-=-=-=-=-=-=-=-=-

// ISSUE/TODO: none of these work properly...lol


// TODO: find y in Nl(S) - T
// should be slack_S == 0 and not in T (use T_id)
// TODO prioritize unmatched...iterate through twice 
int find_y(vector<long>& slack_S, vector<int>& T_id, vector<int>& M){
    int i;
    // look for unmatched 1st == earliest possible termination
    for(i=0; i<slack_S.size(); i++){
        if(slack_S[i] == 0 && T_id[i] == 0 && M[i] < 0){ return i; }
    }
    // look through all 2nd:
    for(i=0; i<slack_S.size(); i++){
        if(slack_S[i] == 0 && T_id[i] == 0){ return i; }
    }
}



void make_Sid(vector<int>& S, vector<int>& S_id){
    int i;
    for(i=0; i<S.size(); i++){
        S_id[S[i]] = 1; 
    }
}


// forward def:
vector<int> rec_augment_back(vector< vector<long> >& w, vector<long>& l_X, vector<long>& l_Y, vector<int> S_id, vector<int> T_id, vector<int> path, int node, int end_node, vector<int>& M);



// recursive augmentation forwards (S->T)
// !M 
vector<int> rec_augment_for(vector< vector<long> >& w, vector<long>& l_X, vector<long>& l_Y, vector<int> S_id, vector<int> T_id, vector<int> path, int node, int end_node, vector<int>& M){

    int i; long eval;
    path.push_back(node); vector<int> hold_path;

    for(i=0; i<T_id.size(); i++){
        if(T_id[i] == 1){
            eval = l_X[node] + l_Y[i] - w[node][i];
            if(eval == 0 && M[i] != node){
                T_id[i] = 0;
                hold_path = rec_augment_back(w, l_X, l_Y, S_id, T_id, path, i, end_node, M); 
                T_id[i] = 1;  

                if(hold_path.size() > 0){ return hold_path; }
            }
        }
    }
    return hold_path; 
}



// recursive augmentation back:
// iterate through S --> anybody 
// can only iterate through M's 
vector<int> rec_augment_back(vector< vector<long> >& w, vector<long>& l_X, vector<long>& l_Y, vector<int> S_id, vector<int> T_id, vector<int> path, int node, int end_node, vector<int>& M){

    // iterate through S --> check all possible paths:
    int i; long eval;
    path.push_back(node); vector<int> hold_path;

    // TODO: check end node 
    if(node == end_node){ return path; }

    // NOTE: don't need a loop == just follow M:

    // if stuck and not at end:
    if(M[node] < 0){ return hold_path; }

    // else: call forward with next node from M:
    T_id[node] = 0; 
    hold_path = rec_augment_for(w, l_X, l_Y, S_id, T_id, path, M[node], end_node, M); // Q: return reference?...if not --> this is slow 

    return hold_path; 
}


// NOTE: it may be better to return const vector (more efficient) 


// TODO: full O(n^2) augmentation
// Augment from s[0] to end unpaired y
// can only traverse each node once
// edges in Gl
// edit M at end
// forward = !M
// backward = M
void augment( vector< vector<long> >& w, vector<long>& l_X, vector<long>& l_Y, vector<int>& S, vector<int> T_id, vector<int>& M, int end_node){
    // make S_id (probs worth it)
    vector<int> S_id(l_X.size(), 0);
    make_Sid(S, S_id);

    // Ensure T value:
    T_id[end_node] = 1; 

    // end node:
    int start_node = S[0]; 

    // initialize path:
    vector<int> path; 

    // call recursive augment forward
    vector<int> ret_path; 
    ret_path =  rec_augment_for(w, l_X, l_Y, S_id, T_id, path, start_node, end_node, M);

    // make M: M uses backward inds
    // path is going forward
    int i;
    for(i=0; i<ret_path.size(); i=i+2){
        M[ret_path[i+1]] = ret_path[i]; 
    } 
}



// TODO: Extend alternating tree if y is matched to z
// S = S U {z}, T = T U {y}
void extend_tree(vector<int>& S, vector<int>& T, vector<int>& T_id, int z, int y){
    S.push_back(z);
    T.push_back(y); 
    T_id[y] = 1; 
}



// -=-=-=-=-=-=-=-=-=-=-=- Master -=-=-=-=-=-=-=-=-
// returns 1 if M is perfect
int M_perf(vector<int>& M){
    int i;
    for(i=0; i<M.size(); i++){
        if(M[i] < 0){ return 0; }
    }
    return 1; 
}


vector<int> hungary_master(vector< vector<long> >& w){
    // initialize:
    vector<long> l_X; vector<long> l_Y; vector<int> M; vector<int> F; 
    init_label(w, l_X, l_Y);    
    init_match(w, M);

    // vectors that will be needed: S, T, T_id:
    vector<int> S; vector<int> T; vector<int> T_id; vector<long> slack_S; 

    int nlt; long dl; int y; int z; 

    // outer while loop == check if M is perfect 
    while(M_perf(M) < 1){
        // initialize F:     
        init_F(w, M, F); 

        // initialize phase:
        init_ST(w, S, T, T_id, F);    

        // initialize slack matrix:
        init_slack(w, l_X, l_Y, slack_S, S[0]);

        // inner while loop: go until we can augment:
        while(1==1){
            // compare neighborhood to T:
            nlt = Nl_T_check(slack_S, T_id);

            if(nlt == 1){ // update labels:
                                
                // get dl:
                dl = calc_dl(slack_S, T_id);

                // update labels: 
                update_labels(l_X, l_Y, S, T, dl);     

                // ISSUE/TODO: Do something about slack_S here?     
                // all members in S --> balanced... so for dude in T --> no change
                // if not in T --> subtract dl from corresponding slack location     
                update_slack_relabel(slack_S, T_id, dl);
           
            }
                
            // is y free?
            y = find_y(slack_S, T_id, M);        

            if(M[y] < 0){ 

                // augment: 
                augment(w, l_X, l_Y, S, T_id, M, y);
                break; 
                    
            }
            else{
        
                // extend tree:
                z = M[y]; 
                extend_tree(S, T, T_id, z, y);

                update_slack(w, l_X, l_Y, slack_S, z);
            }
        }
    } 
    return M; 
}


// TODO: add in function for integration with tracker:
// (0) build distance matrix (vector<vector>>)...argument for negation (negation --> find min)

// NOTE: we are currently using distance squared...

// calculate distance between 2 n-dim pts:
double calc_dist(vector<double>& pt1, vector<double>& pt2){
    int i; double dist = 0.0;
    for(i=0;i<pt1.size();i++){
        dist = dist + ((pt2[i]-pt1[i]) * (pt2[i] - pt1[i]));
    }
    return sqrt(dist); 
    //return dist; 
}


// TODO: multiply by 100 and round --> cast everything to int


// build distance matrix using the 2 sets of pts
// mult = multiplier... ex: -1 --> find minimum
void build_dist_mat(vector< vector<double> >& pts1, vector< vector<double> >& pts2, vector< vector<long> >& mat, double mult){
    vector< vector<double> > hmat(pts1.size(), vector<double>(pts2.size()));
    int i; int j;
    for(i=0; i<pts1.size(); i++){
        for(j=0; j<pts2.size(); j++){
            hmat[i][j] = calc_dist(pts1[i], pts2[j]) * mult;  
        }
    }

    // mult by 100 and cast to long:
    mat = vector< vector<long> >(pts1.size(), vector<long>(pts2.size())); 
    for(i=0; i<pts1.size(); i++){
        for(j=0; j<pts2.size(); j++){
            mat[i][j] = (long) (hmat[i][j] * 100);  
        }
    }

}

/*

// testing:
int main(){ 
    vector< vector<double> > w(4, vector<double>(4)); 

    // test 1: fails now --> returns an illegal M (augmentation issue?) 
    
    w[0][0] = -90; w[0][1] = -75; w[0][2] = -75; w[0][3] = -80;
    w[1][0] = -35; w[1][1] = -85; w[1][2] = -55; w[1][3] = -65;
    w[2][0] = -125; w[2][1] = -95; w[2][2] = -90; w[2][3] = -105;
    w[3][0] = -45; w[3][1] = -110; w[3][2] = -95; w[3][3] = -115;  

    print_mat(w); 

    vector<int> M;
    M = hungary_master(w);

    int i; double sum=0;
    for(i=0; i<M.size(); i++){
        sum = sum + w[M[i]][i];
    }
    cout << "SUM = " << sum << "\n";  
    // CORRECT: M = [3, 2, 1, 0], SUM = -275


    
    // test 2:
    w[0][0] = -20; w[0][1] = -22; w[0][2] = -14; w[0][3] = -24;
    w[1][0] = -20; w[1][1] = -19; w[1][2] = -12; w[1][3] = -20;
    w[2][0] = -13; w[2][1] = -10; w[2][2] = -18; w[2][3] = -16;
    w[3][0] = -22; w[3][1] = -23; w[3][2] = -9; w[3][3] = -28;  

    print_mat(w); 


    M = hungary_master(w);

    // print out the sum:
    sum=0;
    for(i=0; i<M.size(); i++){
        sum = sum + w[M[i]][i];
    }
    cout << "SUM = " << sum << "\n"; 
    // CORRECT: M = [0, 2, 3, 1], SUM = -59 


    // Integration test: min cost for two pt sets:
    vector< vector<double> > pts1(3, vector<double>(2));
    pts1[0][0] = 0.0; pts1[0][1] = 0.0;
    pts1[1][0] = 1.0; pts1[1][1] = 1.0;
    pts1[2][0] = -1.0; pts1[2][1] = 1.0;

    vector< vector<double> > pts2(3, vector<double>(2));
    pts2[0][0] = .5; pts2[0][1] = .5;
    pts2[1][0] = 0.0; pts2[1][1] = 1.0;
    pts2[2][0] = -.5; pts2[2][1] = .5; 

    // build distance matrix: 
    vector< vector<double> > dist_mat; 
    build_dist_mat(pts1, pts2, dist_mat, -1.0);    

    print_mat(dist_mat); 

    M = hungary_master(dist_mat); 
    print_vec(M); 


    // TESTING: hang if two possibilities?
    vector< vector<double> > w2(2, vector<double>(2));
    w2[0][0] = 1.0; w2[0][1] = 1.0;
    w2[1][0] = 2.0; w2[1][1] = 2.0;

    M = hungary_master(w2);
    print_vec(M); 
}

*/

// OK: seems to be working for tougher example now...should we translate M? 
