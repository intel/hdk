/**
 * @file	MathExpr.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
/**
 * @file	MathExpr.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_MATHEXPR_NODE_H
#define RA_MATHEXPR_NODE_H

#include <cassert>
#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

enum Op {
    OP_PLUS,OP_MINUS,OP_TIMES,OP_DIVIDE
};

class MathExpr : public RelAlgNode {
    
public:
	MathExpr *n1 = NULL;
	MathExpr *n2 = NULL;
	Attribute *n3 = NULL;
	AggrExpr *n4 = NULL;
	//std::string op = "";
    Op op;
    bool isScalar;
	int intVal;
	float floatVal;
	bool intFloatFlag;	// true if int; otherwise float
    bool isBinaryOp;

	MathExpr(std::string op, MathExpr *n1, MathExpr *n2) {
		assert(op == "PLUS" || op == "MINUS" || op == "MULTIPLY" || op == "DIVIDE");
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
		if (op == "PLUS")
			this->op = PLUS;
		else if (op == "MINUS")
			this->op = MINUS; 
		else if (op == "MULTIPLY")
			this->op = MULTIPLY; 
		else if (op == "DIVIDE")
			this->op = DIVIDE; 

        binaryOp = true;
	}

	explicit MathExpr(MathExpr *n1) {
		assert(n1);
		this->n1 = n1;
	}

	explicit MathExpr(Attribute *n3) {
		assert(n3);
		this->n3 = n3;
	}

	explicit MathExpr(AggrExpr *n4) {
		assert(n4);
		this->n4 = n4;
	}

	explicit MathExpr(int intVal) {
		this->intVal = intVal;
        isScalar = true; // will this default to false
		intFloatFlag = true;
	}

	explicit MathExpr(float floatVal) {
		this->floatVal = floatVal;
        isScalar = true;
		intFloatFlag = false;
	}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

} // RA_Namespace

#endif // RA_MATHEXPR_NODE_H
