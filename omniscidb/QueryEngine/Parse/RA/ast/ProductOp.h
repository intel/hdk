/**
 * @file	ProductOp.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_PRODUCTOP_NODE_H
#define RA_PRODUCTOP_NODE_H

#include <cassert>
#include "BinaryOp.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class ProductOp : public BinaryOp {
    
public:
	RelExpr *n1 = NULL;
	RelExpr *n2 = NULL;

	/// Constructor
	ProductOp(RelExpr *n1, RelExpr *n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}

};

} // RA_Namespace

#endif // RA_PRODUCTOP_NODE_H
