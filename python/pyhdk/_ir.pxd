#
# Copyright 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr

from pyhdk._common cimport CType
from pyhdk._storage cimport CTableInfoPtr

cdef extern from "omniscidb/IR/Expr.h":
  cdef cppclass CExpr "hdk::ir::Expr":
    string toString() const

    bool isExpr "is"[T]() const
    const T* asExpr "as"[T]() const

    const CType* type() const

  cdef cppclass CColumnRefExpr "hdk::ir::ColumnRef" (CExpr):
    int index() const


cdef extern from "omniscidb/IR/Node.h":
  cdef cppclass CNode "hdk::ir::Node":
    string toString() const

    bool isNode "is"[T]() const
    const T* asNode "as"[T]() const

  cdef cppclass CScan "hdk::ir::Scan" (CNode):
    CTableInfoPtr getTableInfo() const


ctypedef shared_ptr[CExpr] CExprPtr
ctypedef shared_ptr[CNode] CNodePtr
