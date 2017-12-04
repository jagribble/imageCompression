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

}
