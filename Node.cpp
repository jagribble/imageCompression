//
// Created by Jules on 01/12/2017.
//

#include "Node.h"

PNode::PNode(int v,int f,int total) {
    value = v;
    frequency = f;
    huffmanProbability = frequency/float(total);
}

PNode::PNode() {
    value = -1;
}

PNode::PNode(int v, int f, PNode *l, PNode *r, string p, float h) {
    value = v;
    frequency = f;
    left = l;
    right = r;
    prefix = p;
    huffmanProbability =h;

}
