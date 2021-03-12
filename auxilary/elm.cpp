//
// Created by Jees Augustine on 4/28/20.
//
#include <cstdio>
#include <algorithm>
#include <map>
#include <list>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <boost/heap/pairing_heap.hpp>
#include <ctime>
#include <cstdlib>
#include <boost/math/constants/constants.hpp>
#include <chrono>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/connected_components.hpp>
#include <utility>
#include <random>
#include <cmath>
#include <fstream>

const int DEBUG = 0;

using namespace std;

struct sps {
    pair<unsigned int, float> data;
    sps(unsigned int node, float dist) {
        data = make_pair(node, dist);
    }

    bool operator<(sps const & sps2) const {
        return data.second > sps2.data.second;
    }
};

struct VertexProperties {
  unsigned int index;
};

typedef boost::adjacency_list<boost::vecS, boost::listS, boost::undirectedS, VertexProperties> Graph;
typedef boost::erdos_renyi_iterator<boost::minstd_rand, Graph> ERGen;

vector<float>* Dijkstra(vector<list<pair<unsigned int, float>>*>* adj_lst,
        vector<vector<float>*>* adj_mat, unsigned int source, float threshold = 1.) {
    unsigned int nodes = adj_lst->size();
    vector<bool> visited(nodes, false);
    visited.at(source) = true;
    boost::heap::pairing_heap<sps> H;
    vector<boost::heap::detail::node_handle<boost::heap::detail::heap_node<sps, false>*, boost::heap::detail::make_pairing_heap_base<sps, boost::parameter::aux::empty_arg_list>::type, sps&>> handles;
    vector<float>* sp = new vector<float>(adj_lst->size(), 1.);
    sp->at(source) = 0.;
    for(unsigned int j = 0; j < nodes; ++j) {
        handles.push_back((boost::heap::detail::node_handle<boost::heap::detail::heap_node<sps, false>*, boost::heap::detail::make_pairing_heap_base<sps, boost::parameter::aux::empty_arg_list>::type, sps&>)NULL);
    }
    for (list<pair<unsigned int, float>>::iterator it=adj_lst->at(source)->begin();
            it != adj_lst->at(source)->end(); ++it) {
        handles.at(it->first) = H.push(sps(it->first, it->second));
    }
    while(!H.empty() && H.top().data.second < 1.) {
        unsigned int dest = H.top().data.first;
        float dist = H.top().data.second;
        H.pop();
        sp->at(dest) = dist;
        visited.at(dest) = true;
        for (list<pair<unsigned int, float>>::iterator it=adj_lst->at(dest)->begin();
                    it != adj_lst->at(dest)->end(); ++it) {
            unsigned int neighbour = it->first;
            if(visited.at(neighbour)) {
                continue;
            }
            float total = it->second + dist;
            if(handles.at(neighbour) == (boost::heap::detail::node_handle<boost::heap::detail::heap_node<sps, false>*, boost::heap::detail::make_pairing_heap_base<sps, boost::parameter::aux::empty_arg_list>::type, sps&>)NULL) {
                handles.at(neighbour) = H.push(sps(neighbour, total));
            } else if(handles.at(neighbour).node_->value.data.second > total) {
                H.increase(handles.at(neighbour), sps(neighbour, total));
            }
        }
    }
    return sp;
}

class EdgeLandMark {
    private:
        map<pair<unsigned int, unsigned int>, float>* known;
        unsigned int order_val, k, sample_size;
        vector<pair<unsigned int, unsigned int>>* sampled;
        set<pair<unsigned int, unsigned int>>* landmarks;
        vector<float>* lb_lm;
        vector<list<pair<unsigned int, float>>*>* adj_list;
        vector<vector<float>*>*  adj_mat;
        set<unsigned int>* node_dij;
        map<unsigned int, vector<float>*>* dij_paths;
        map<pair<unsigned int, unsigned int>, float>* lm_dist;

    public:
        EdgeLandMark(vector<list<pair<unsigned int, float>>*>* adj_list, vector<vector<float>*>*  adj_mat, map<pair<unsigned int, unsigned int>, float>* known_edges,unsigned int n_nodes,unsigned int k,unsigned int sampling_size){
            this->known = known_edges;
            this->order_val = n_nodes;
            this->k = k;
            this->sample_size = sampling_size;
            this->lb_lm = new vector<float>();
            this->landmarks = new set<pair<unsigned int, unsigned int>>();
            this->adj_list = adj_list;
            this->adj_mat = adj_mat;
            this->node_dij = new set<unsigned int>();
            this->dij_paths = new map<unsigned int, vector<float>*>();
            this->lm_dist = new map<pair<unsigned int, unsigned int>, float>();
	        this->sampled = new vector<pair<unsigned int, unsigned int>>();
        }

        ~EdgeLandMark() {
            delete lb_lm;
            delete landmarks;
            for (auto it=this->dij_paths->begin(); it!=this->dij_paths->end(); ++it) {
                delete it->second;
            }
            delete this->dij_paths;
            delete this->lm_dist;
	    delete this->sampled;
        }

        void sample_unknown_edges(){
            /* initialize random seed: */
//            srand (time(NULL));
//            srand (0);
            unsigned int i = 0;
            while(i < this->sample_size) {
                unsigned int u = rand() % this->order_val;
                unsigned int v = rand() % this->order_val;
                if(u == v) {
                    continue;
                }
                pair<unsigned int, unsigned int> uv = make_pair(u, v);
                pair<unsigned int, unsigned int> vu = make_pair(v, u);
//                if ((find(sampled->begin(), sampled->end(), uv) != sampled->end())||
//                        (find(sampled->begin(), sampled->end(), vu) != sampled->end()))
                if ((this->known->find(uv) != this->known->end()) || (this->known->find(vu) != this->known->end()))
                    continue;
                else
                {
                    if(u > v) {
                        unsigned int temp = u;
                        u = v;
                        v = temp;
                    }
                    sampled->push_back(uv);
                    this->node_dij->insert(u);
                    this->node_dij->insert(v);
                    ++i;
                }
            }
            cout << sampled->size() << endl;
        }

        void all_unknown_edges() {
            for(unsigned int i=0; i < this->adj_mat->size(); ++i) {
                for(unsigned int j=i+1; j < this->adj_mat->size(); ++j) {
                    if(this->adj_mat->at(i)->at(j) < -0.1) {
                        this->sampled->push_back(make_pair(i, j));
                        this->node_dij->insert(i);
                        this->node_dij->insert(j);
                    }
                }
            }
            this->sample_size = this->sampled->size();
        }

        void find_paths() {
            //find the dijkstra paths from the sampled edges in the variable 'sampled'
            for (auto it = this->node_dij->begin(); it != this->node_dij->end(); ++it){
                unsigned int src = *it;
                this->dij_paths->insert(make_pair(*it, Dijkstra(this->adj_list, this->adj_mat, src)));
            }
            delete this->node_dij;
        }

        void greedy_sampling(){
            // function greedily selects the edges from the known edges for our greedy algorithm

            pair<unsigned int, unsigned int> edge_selected;// = NULL;
            vector<float>* lbs = new vector<float>();
            this->node_dij = new set<unsigned int>();
            for(unsigned int i = 0; i < sampled->size(); ++i) {
                lbs->push_back((float)0.);
            }
            unsigned int k_local = this->k - 1;
            float total_e1 = 0;
            cout << "Constructing big matrix" << endl;
            unsigned int ind_known = 0;
            for ( auto it = this->known->begin(); it != this->known->end(); ++it ) {
                pair<unsigned int, unsigned int> edge = it->first;
		        if(++ind_known %1000 == 0) {
                    cout << ind_known << "/" << this->known->size() << endl;
                }
                float dist_m= it->second;
                unsigned int u =  edge.first, v =  edge.second;
                float total = 0;
                for(unsigned int i = 0; i < sampled->size(); ++i) {
                    pair<unsigned int, unsigned int> p = sampled->at(i);
                    unsigned int m_end_point1 = p.first, m_end_point2 = p.second;
                    vector<float>* m1_paths = this->dij_paths->find(m_end_point1)->second;
                    vector<float>* m2_paths = this->dij_paths->find(m_end_point2)->second;
                    float m1_u = m1_paths->at(u), m1_v = m1_paths->at(v), m2_u = m2_paths->at(u), m2_v = m2_paths->at(v);
                    float max_val = max(dist_m - m1_u - m2_v, dist_m - m1_v - m2_u);
                    max_val = max((float)0., max_val);
                    this->lb_lm->push_back(max_val);
                    total += max_val;
                }
                if(total > total_e1) {
                    total_e1 = total;
                    edge_selected = edge;
                }
            }
            cout << "Constructed big matrix" << endl;
            this->landmarks->insert(edge_selected);
            this->lm_dist->insert(make_pair(edge_selected, this->known->find(edge_selected)->second));
            this->node_dij->insert(edge_selected.first);
            this->node_dij->insert(edge_selected.second);
            while(k_local > 0) {
                auto iter = this->known->find(edge_selected);
                unsigned int index = distance(this->known->begin(), iter);
                // Update distances
                float lb_sum_iter = 0.0;
                for(unsigned int i = 0; i < sampled->size(); ++i) {
                    lbs->at(i) = max(lbs->at(i), this->lb_lm->at(index*this->sample_size + i));
                    lb_sum_iter += lbs->at(i);
                }
                cout << "Adding edge : " << (this->k - k_local + 1) << " " << lb_sum_iter << " " << 9.74897-lb_sum_iter << endl;
                int index_known = 0;
                float total_e1 = 0;
                for ( auto it = this->known->begin(); it != this->known->end(); ++it, ++index_known ) {
                    // if(this->landmarks->find(it->first) != this->landmarks->end()){
                    //    continue;
                    //}
                    if(find(this->landmarks->begin(), this->landmarks->end(), it->first) != this->landmarks->end())                        
continue;
                    float total = 0;
                    for(unsigned int i = 0; i < sampled->size(); ++i) {
                        float lb_cur = this->lb_lm->at(index_known*sampled->size() + i);
                        if(lb_cur > lbs->at(i))
                            total += lb_cur - lbs->at(i);
                    }
//                    if((total - total_e1 >= -0.1) && (landmarks->find(it->first) == landmarks->end())){
                    if((total - total_e1 >= 0.) && (landmarks->find(it->first) == landmarks->end())){
                        total_e1 = total;
                        edge_selected = it->first;
                    }
                }
                this->landmarks->insert(edge_selected);
                this->lm_dist->insert(make_pair(edge_selected, this->known->find(edge_selected)->second));
                this->node_dij->insert(edge_selected.first);
                this->node_dij->insert(edge_selected.second);
                --k_local;
                cout << "size of selected edge set: " << landmarks->size() << endl;
                cout << "selected edge is: " << edge_selected.first << " "<< edge_selected.second  << endl;
            }
            cout << "Found k nodes" << endl;
            auto it = this->dij_paths->cbegin();
            while (it != this->dij_paths->cend())
            {
                if (this->node_dij->find(it->first) != this->node_dij->cend())
                {
                    it = this->dij_paths->erase(it);
                }
                else {
                    ++it;
                }
            }
            for(auto it=this->node_dij->begin(); it != this->node_dij->end(); ++it) {
                if (this->dij_paths->find(*it) == this->dij_paths->cend()) {
                    this->dij_paths->insert(make_pair(*it, Dijkstra(this->adj_list, this->adj_mat, *it)));
                }
            }
            delete lbs;
            delete this->node_dij;
        }

        float lookup(unsigned int u, unsigned int v) {
            float lb = 0.;
            for(auto it=this->landmarks->begin(); it != this->landmarks->end(); ++it) {
                pair<unsigned int, unsigned int> p = *it;
                vector<float>* ptr1 = this->dij_paths->find(p.first)->second;
                vector<float>* ptr2 = this->dij_paths->find(p.second)->second;
                float u1 = ptr1->at(u), v1 = ptr1->at(v), u2 = ptr2->at(u), v2 = ptr2->at(v);
                float lm_edge = this->lm_dist->find(*it)->second;
                lb = max(max(lb, lm_edge - u1 - v2), lm_edge - u2 - v1);
            }
            return lb;
        }
};

void SashaWang(vector<vector<float>*>* lb, vector<vector<float>*>* ub) {
    size_t nodes = lb->size();
    for(size_t k=0; k < nodes; ++k) {
        for(size_t j=0; j < nodes; ++j) {
            for(size_t i=0; i < nodes; ++i) {
                float max_lb = max(lb->at(i)->at(k) - ub->at(k)->at(j),
                                    lb->at(k)->at(j) - ub->at(i)->at(k));
                lb->at(i)->at(j) = max(lb->at(i)->at(j), max_lb);
                ub->at(i)->at(j) = min(ub->at(i)->at(j), ub->at(i)->at(k) + ub->at(k)->at(j));
            }
        }
    }
}

int check_connected(Graph g, boost::property_map<Graph, unsigned int VertexProperties::*>::type id) {
    int count = 0;
    unsigned int nodes = boost::num_vertices(g);
    if(nodes == 0) {
        return count;
    }
    list<unsigned int> queue;
    unsigned int total_visited = 0;
    vector<bool> visited(nodes, false);
    while(total_visited != nodes) {
        ++count;
        queue.clear();
        for(unsigned int i=0; i < nodes;++i) {
            if(!visited.at(i)) {
                ++total_visited;
                visited.at(i) = true;
                queue.push_back(i);
                break;
            }
        }
        while(!queue.empty()) {
            unsigned int node = queue.back();
            queue.pop_back();
            boost::graph_traits<Graph>::out_edge_iterator ei, edge_end;
            boost::graph_traits<Graph>::vertex_iterator i, end;
            for (tie(i,end) = vertices(g); i != end; ++i) {
                if(id[*i] == node) {
                    break;
                }
            }
            for (tie(ei,edge_end) = out_edges(*i, g); ei != edge_end; ++ei) {
                unsigned int neighbour = id[target(*ei, g)];
                if(!visited.at(neighbour)) {
                    visited.at(neighbour) = true;
                    queue.push_back(neighbour);
                    ++total_visited;
                }
            }
        }
    }
    return count;
}

vector<list<pair<unsigned int, float>>*>* get_adjacency_list(Graph g,
        boost::property_map<Graph, unsigned int VertexProperties::*>::type id,
        vector<vector<float>*>* dist) {

    vector<list<pair<unsigned int, float>>*>* adj_list = new vector<list<pair<unsigned int, float>>*>();
    boost::graph_traits<Graph>::vertex_iterator i, end;
    boost::graph_traits<Graph>::out_edge_iterator ei, edge_end;

    for (tie(i,end) = vertices(g); i != end; ++i) {
        unsigned int end_point1 = id[*i];
        adj_list->push_back(new list<pair<unsigned int, float>>());
        for (tie(ei,edge_end) = out_edges(*i, g); ei != edge_end; ++ei) {
            unsigned int end_point2 = id[target(*ei, g)];
            float distance = dist->at(end_point1)->at(end_point2);
            std::pair<unsigned int, float> pr = make_pair(end_point2, distance);
            adj_list->at(end_point1)->push_back(pr);
        }
    }
    return adj_list;
}

vector<vector<float>*>* get_adjacency_matrix(Graph g,
    boost::property_map<Graph, unsigned int VertexProperties::*>::type id,
        vector<vector<float>*>* dist, float default_missing = -1.) {

    vector<vector<float>*>* adj_matrix = new vector<vector<float>*>();
    boost::graph_traits<Graph>::vertex_iterator i, end;
    boost::graph_traits<Graph>::out_edge_iterator ei, edge_end;
    unsigned int nodes = num_vertices(g);
    for(unsigned int i =0; i < nodes; ++i) {
        adj_matrix->push_back(new vector<float>());
        for(unsigned int j =0; j < nodes; ++j) {
            adj_matrix->at(i)->push_back(default_missing);
        }
        adj_matrix->at(i)->at(i) = 0.;
    }
    for (tie(i,end) = vertices(g); i != end; ++i) {
        unsigned int end_point1 = id[*i];
        for (tie(ei,edge_end) = out_edges(*i, g); ei != edge_end; ++ei) {
            unsigned int end_point2 = id[target(*ei, g)];
            float distance = dist->at(end_point1)->at(end_point2);
            adj_matrix->at(end_point1)->at(end_point2) = distance;
        }
    }
    return adj_matrix;
}

vector<vector<float>*>* distance_matrix(unsigned int nodes, unsigned int dims, unsigned p=2) {
    uniform_real_distribution<float> runif(-1.,1.);
    uniform_real_distribution<float> aunif(0.,2.*boost::math::constants::pi<float>());
    default_random_engine re;
    vector<vector<float>*>* dist = new vector<vector<float>*>(); // nodes X nodes
    vector<vector<float>*>* points = new vector<vector<float>*>(); // nodes X d
    float max_val = 0;
    for(unsigned int i=0;i < nodes; ++i){
        float r = runif(re);
        float entity = r;
        float angle = aunif(re);
        points->push_back(new vector<float>());
        for(unsigned int j = 1; j < dims; ++j) {
            points->at(i)->push_back(entity * cos(angle));
            entity *= sin(angle);
            angle = aunif(re);
        }
        points->at(i)->push_back(entity);
    }

    cout << "Allocated points" << endl;

    for(unsigned int i=0;i < nodes; ++i){
        dist->push_back(new vector<float>());
        for(unsigned int j=0;j < nodes; ++j){
            if(i==j){
                dist->at(i)->push_back((float)0.);
            } else {
                float total = 0;
                for(unsigned int k =0; k < dims; ++k) {
                    float val = points->at(i)->at(k) - points->at(j)->at(k);
                    if(val < 0)
                        val *= -1.;
                    total += pow(val, (float)p);
                }
                total = pow(total, (float)1./p)/2;
                max_val = max(max_val, total);
                dist->at(i)->push_back(total);
            }
        }
    }

    for(unsigned int i=0;i < nodes; ++i){
        if(DEBUG) {
            for(unsigned int j=0;j < dims; ++j){
                cout << " " << points->at(i)->at(j);
            }
            cout << endl;
        }
        delete points->at(i);
    }
    delete points;
    cout << "Maximum value encountered is " << max_val << endl;
    return dist;
}

void clean_up_adj_matrix(vector<vector<float>*>* adj_matrix) {
    unsigned int adj_matrix_size = adj_matrix->size();
    while(adj_matrix_size-- > 0) {
        delete adj_matrix->at(adj_matrix_size);
    }
    delete adj_matrix;
}

void clean_up_adj_list(vector<list<pair<unsigned int, float>>*>* adj_lst) {
    unsigned int adj_lst_size = adj_lst->size();
    while(adj_lst_size-- > 0) {
        delete adj_lst->at(adj_lst_size);
    }
    delete adj_lst;
}

map<pair<unsigned int, unsigned int>, float>* convert_adjList_to_knownEdges(vector<list<pair<unsigned int, float>>*>* adj_lst) {
    map<pair<unsigned int, unsigned int>, float>* known_edges = new map<pair<unsigned int, unsigned int>, float>();
    ofstream myfile;
    std::string file_name = "graph_" + to_string(adj_lst->size()) + ".txt";
    myfile.open (file_name);
    unsigned int ctr = 0;
    for(unsigned int u=0; u < adj_lst->size(); ++u) {
        list<pair<unsigned int, float>>* lst = adj_lst->at(u);
        for (auto it=lst->begin(); it != lst->end(); ++it) {
            pair<unsigned int, float> p = *it;
            unsigned int v = p.first, u1 = u;
            if(v < u1) {
                v = u1;
                u1 = p.first;
            }
            std::stringstream u_;
            std::stringstream v_;
            std::stringstream p_;
            u_ << u1;
            v_ << v;
            p_ << p.second;
            std::string out_u = u_.str();
            std::string out_v = v_.str();
            std::string out_p = p_.str();
            std::string outline = out_u + " " + out_v + " " + out_p + "\n";
            myfile << outline;
            known_edges->insert(make_pair(make_pair(u1, v), p.second));
//            cout << "U_1 " << u1 << " " << "V " << v << endl;
//            cout << "known edge size: " << known_edges->size() << endl;
            ++ctr;
        }
    }
//    cout << "counter: " << ctr << endl;
    myfile.close();
    cout << "Input (Graph)File is written; You can start Python code" << endl;

    return known_edges;
}

vector<list<pair<unsigned int, float>>*>* get_adj_list_file(char *filename) {
    vector<list<pair<unsigned int, float>>*>* adj_list = new vector<list<pair<unsigned int, float>>*>();
    ifstream ifs;
    ifs.open (filename, ifstream::in);
    unsigned int nodes;
    ifs >> nodes;
    for(int i=0; i < nodes; i++) {
        adj_list->push_back(new list<pair<unsigned int, float>>());
    }
    unsigned int u, v, edge_numbers = 0;
    float dist;
    while(ifs >> u >> v >> dist) {
        adj_list->at(u)->push_back(make_pair(v, dist));
        adj_list->at(v)->push_back(make_pair(u, dist));
        ++edge_numbers;
        unsigned int u1 = min(u, v);
        unsigned int v1 = max(u, v);
        cout << u1 << " " <<v1 << endl;
    }
    cout << "From read file" << edge_numbers << endl;
    return adj_list;
}

vector<vector<float>*>* get_adj_matrix_file(char *filename, float default_val=-1.) {
    vector<vector<float>*>* adj_mat = new vector<vector<float>*>();
    ifstream ifs;
    ifs.open (filename, ifstream::in);
    unsigned int nodes;
    ifs >> nodes;
    for(int i=0; i < nodes; i++) {
        adj_mat->push_back(new vector<float>());
        for(int j=0; j < nodes; j++) {
            adj_mat->at(i)->push_back(default_val);
        }
        adj_mat->at(i)->at(i) = 0.;
    }
    unsigned int u, v;
    float dist;
    while(ifs >> u >> v >> dist) {
//        cout << u << " " << v << " " << dist << endl;
        adj_mat->at(u)->at(v) = adj_mat->at(v)->at(u) = dist;
    }
    return adj_mat;
}

int main(int argc, char **argv) {
    srand(0);
    if(argc < 1) {
        return 0;
    }
    vector<list<pair<unsigned int, float>>*>* adj_lst =get_adj_list_file(argv[1]);

    vector<vector<float>*>* adj_mat = get_adj_matrix_file(argv[1], (float)-1.);
    vector<vector<float>*>* lb = get_adj_matrix_file(argv[1], (float)0.);
    vector<vector<float>*>* ub = get_adj_matrix_file(argv[1], (float)1.);
    vector<vector<float>*>* lb_elm = get_adj_matrix_file(argv[1], (float)0.);


    unsigned int nodes = adj_lst->size(), k=100;
    unsigned int sampling_size = 54000;

    if(argc > 2) {
        k = stoi(argv[2]);
    }
    if(argc > 3) {
        sampling_size = stoi(argv[3]);
    }
    auto start_lb_elm = chrono::high_resolution_clock::now();
    map<pair<unsigned int, unsigned int>, float>* known_edges = convert_adjList_to_knownEdges(adj_lst);
    cout << "size of known: " << known_edges->size() <<endl;
    EdgeLandMark* elm = new EdgeLandMark(adj_lst, adj_mat, known_edges, nodes, k, sampling_size);
    cout << "Sampling unknown edges" << endl;
    if(sampling_size > 0) {
        cout << "Sampling is chosen ..... " << endl;
        elm->sample_unknown_edges();
    } else{
        cout << "No Sampling; Full Unkown Edges are chosen ..... " << endl;
        elm->all_unknown_edges();
    }
    cout << "Finding paths" << endl;
    elm->find_paths();
//    cout << "Greedy sampling" << endl;
    elm->greedy_sampling();

    float total_lb_elm = 0.;
    for(unsigned int i=0; i < nodes; ++i) {
        for(unsigned int j=i+1; j < nodes; ++j) {
            if(adj_mat->at(i)->at(j) < -0.1) { // Change back to this later.
                lb_elm->at(i)->at(j) = elm->lookup(i, j);
                total_lb_elm += lb_elm->at(i)->at(j);
            }
        }
    }

    auto stop_lb_elm = chrono::high_resolution_clock::now();
    auto duration_lb_elm = chrono::duration_cast<chrono::microseconds>(stop_lb_elm - start_lb_elm);
    std::cout << "Duration ELM LB:" <<  duration_lb_elm.count()/1000000.0 << endl;
    cout << "Total ELM LB : " << total_lb_elm << endl;

    float total_lb_sw = 0.;
    float graph_weight_orig = 0.;
    auto start_lb_sw = chrono::high_resolution_clock::now();
    SashaWang(lb, ub);
    auto stop_lb_sw = chrono::high_resolution_clock::now();
    auto duration_lb_sw = chrono::duration_cast<chrono::microseconds>(stop_lb_sw - start_lb_sw);

    float relative = 0.;
    unsigned int relative_count = 0;
    float rmse = 0.;
    for(unsigned int i=0; i < nodes; ++i) {
        for(unsigned int j=i+1; j < nodes; ++j) {
            if(adj_mat->at(i)->at(j) < -0.1) {
                if(lb->at(i)->at(j) > 0) {
                    relative += 1 - (lb_elm->at(i)->at(j))/(lb->at(i)->at(j));
                    total_lb_sw += lb->at(i)->at(j);
                    rmse += ((lb_elm->at(i)->at(j)) - (lb->at(i)->at(j))) * ((lb_elm->at(i)->at(j)) - (lb->at(i)->at(j)));
                    relative_count++;
                }
            }
            else{
                graph_weight_orig += lb->at(i)->at(j);
            }
        }
    }


    std::cout << "Duration SW LB:" <<  duration_lb_sw.count()/1000000.0 << endl;
    cout << "Total Original Graph weight : " << graph_weight_orig << endl;
    cout << "Total SW LB : " << total_lb_sw << endl;
    cout << "Sum Relative Error on edge : " << relative/relative_count << endl;
    cout << "root mean square error : " << sqrt(rmse/relative_count) << endl;

    delete elm;
    delete known_edges;
    clean_up_adj_list(adj_lst);
    clean_up_adj_matrix(adj_mat);
    clean_up_adj_matrix(lb);
    clean_up_adj_matrix(ub);
    clean_up_adj_matrix(lb_elm);

    return 0;
}

int old_main(int argc, char **argv) {
    unsigned int init = time(NULL);
    unsigned int nodes = 1000, k=100, sampling_size = 54000;
    float prob = 0.05;
    if(argc > 1) {
        init = stoi(argv[1]);
    }
    if(argc > 2) {
        nodes = stoi(argv[2]);
    }
    if(argc > 3) {
        prob = stof(argv[3]);
    }
    if(argc > 4) {
        k = stoi(argv[4]);
    }
    if(argc > 5) {
        sampling_size = stoi(argv[5]);
    }

    // Create graph with 100 nodes and edges with probability 0.05
    srand(init);
    boost::minstd_rand gen;
    Graph g(ERGen(gen, nodes, prob), ERGen(), nodes);

    boost::property_map<Graph, unsigned int VertexProperties::*>::type id = get(&VertexProperties::index, g);
    boost::graph_traits<Graph>::vertex_iterator vi, viend;

    int vert_num= 0;
    for (tie(vi,viend) = vertices(g); vi != viend; ++vi) {
        id[*vi] = vert_num++;
    }

    boost::graph_traits<Graph>::vertex_iterator i, end;
    boost::graph_traits<Graph>::out_edge_iterator ei, edge_end;

    int total = 0;
    for (tie(i,end) = vertices(g); i != end; ++i) {
        if(DEBUG) {
            cout << id[*i] << " ";
        }
        int count = 0;
        for (tie(ei,edge_end) = out_edges(*i, g); ei != edge_end; ++ei) {
            if(DEBUG) {
                cout << id[target(*ei, g)] << "  ";
            }
            ++count;
        }
        total += count;
        if(DEBUG) {
            cout << count << endl;
        }
    }
    cout << "Total nodes : " << nodes << endl;
    cout << "Total edges : " << total/2. << endl;
    cout << "Components " << check_connected(g, id) << endl;
    cout << "Adjacency matrix" << endl;
    vector<vector<float>*>* distance = distance_matrix(nodes, 6);
    vector<list<pair<unsigned int, float>>*>* adj_lst = get_adjacency_list(g, id, distance);
    vector<vector<float>*>* adj_mat = get_adjacency_matrix(g, id, distance, (float)-1.);
    vector<vector<float>*>* lb = get_adjacency_matrix(g, id, distance, (float)0.);
    vector<vector<float>*>* ub = get_adjacency_matrix(g, id, distance, (float)1.);
    vector<vector<float>*>* lb_elm = get_adjacency_matrix(g, id, distance, (float)0.);


    auto start_lb_elm = chrono::high_resolution_clock::now();
    map<pair<unsigned int, unsigned int>, float>* known_edges = convert_adjList_to_knownEdges(adj_lst);
    EdgeLandMark* elm = new EdgeLandMark(adj_lst, adj_mat, known_edges, nodes, k, sampling_size);
    cout << "Sampling unknown edges" << endl;
    elm->sample_unknown_edges();
    cout << "Finding paths" << endl;
    elm->find_paths();
    cout << "Greedy sampling" << endl;
    elm->greedy_sampling();

    float total_lb_elm = 0.;
    for(unsigned int i=0; i < nodes; ++i) {
        for(unsigned int j=i+1; j < nodes; ++j) {
            if(adj_mat->at(i)->at(j) < -0.1) {
                lb_elm->at(i)->at(j) = elm->lookup(i, j);
                total_lb_elm += lb_elm->at(i)->at(j);
            }
        }
    }
    auto stop_lb_elm = chrono::high_resolution_clock::now();
    auto duration_lb_elm = chrono::duration_cast<chrono::microseconds>(stop_lb_elm - start_lb_elm);
    std::cout << "Duration ELM LB:" <<  duration_lb_elm.count()/1000000.0 << endl;
    cout << "Total ELM LB : " << total_lb_elm << endl;

    float total_lb_sw = 0.;
    float graph_weight_orig = 0.;
    auto start_lb_sw = chrono::high_resolution_clock::now();
    SashaWang(lb, ub);
    auto stop_lb_sw = chrono::high_resolution_clock::now();
    auto duration_lb_sw = chrono::duration_cast<chrono::microseconds>(stop_lb_sw - start_lb_sw);

    float relative = 0.;
    unsigned int relative_count = 0;
    float rmse = 0;
    for(unsigned int i=0; i < nodes; ++i) {
        for(unsigned int j=i+1; j < nodes; ++j) {
            if(adj_mat->at(i)->at(j) < -0.1) {
                if(lb->at(i)->at(j) > 0) {
                    relative += 1 - (lb_elm->at(i)->at(j))/(lb->at(i)->at(j));
                    relative_count++;
                    total_lb_sw += lb->at(i)->at(j);
                    rmse += ((lb_elm->at(i)->at(j)) - (lb->at(i)->at(j))) * ((lb_elm->at(i)->at(j)) - (lb->at(i)->at(j)));
                }
            }
            else{
                graph_weight_orig += lb->at(i)->at(j);
            }
        }
    }

    std::cout << "Duration SW LB:" <<  duration_lb_sw.count()/1000000.0 << endl;
    cout << "Total Original Graph weight : " << graph_weight_orig << endl;
    cout << "Total SW LB : " << total_lb_sw << endl;
    cout << "Sum Relative Error on edge : " << relative/relative_count << endl;
    cout << "root mean square error : " << sqrt(rmse/relative_count) << endl;

    delete elm;
    delete known_edges;
    clean_up_adj_list(adj_lst);
    clean_up_adj_matrix(adj_mat);
    clean_up_adj_matrix(lb);
    clean_up_adj_matrix(ub);
    clean_up_adj_matrix(lb_elm);
    //cout << "Distances " << endl;
    for(unsigned int i=0;i < nodes; ++i){
        if(DEBUG) {
            for(unsigned int j=0;j < nodes; ++j){
                cout << " " << distance->at(i)->at(j);
            }
            cout << endl;
        }
        delete distance->at(i);
    }
    delete distance;
    return 0;
}
