//
// Created by Jules on 01/12/2017.
//

#ifndef IMAGECOMPRESSION_NODE_H
#define IMAGECOMPRESSION_NODE_H

#include <iostream>


using namespace std;
class PNode {
    public:
        int value;
        int frequency;
        PNode *left;
        PNode *right;
        string prefix;
        float huffmanProbability;
        PNode(int v,int f,int total);
        PNode();
        PNode(int v, int f, PNode *l, PNode *r, string p, float h);
};


#endif //IMAGECOMPRESSION_NODE_H
