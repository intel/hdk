#
# Copyright 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libc.stdint cimport int64_t
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, unique_ptr

from pyhdk._common cimport CContext, CConfigPtr, CType, TypeInfo
from pyhdk._ir cimport CExpr, CExprPtr, CNode, CNodePtr
from pyhdk._storage cimport CSchemaProviderPtr, CColumnInfoPtr
from pyhdk._sql cimport CExecutionResult, CQueryDag


cdef extern from "omniscidb/QueryBuilder/QueryBuilder.h":
  cdef cppclass CQueryBuilder "hdk::ir::QueryBuilder"
  cdef cppclass CBuilderExpr "hdk::ir::BuilderExpr"

  cdef cppclass CBuilderOrderByKey "hdk::ir::BuilderOrderByKey":
    CBuilderOrderByKey(const CBuilderExpr&, const string&, const string&) except +

  cdef cppclass CBuilderExpr "hdk::ir::BuilderExpr":
    CBuilderExpr()

    CExprPtr expr() const
    const CQueryBuilder &builder() const

    CBuilderExpr rename(const string&) const
    const string& name() const

    CBuilderExpr avg() except +
    CBuilderExpr min() except +
    CBuilderExpr max() except +
    CBuilderExpr sum() except +
    CBuilderExpr count(bool) except +
    CBuilderExpr approxCountDist() except +
    CBuilderExpr approxQuantile(double) except +
    CBuilderExpr sample() except +
    CBuilderExpr singleValue() except +
    CBuilderExpr stdDev() except +
    CBuilderExpr corr(const CBuilderExpr&) except +

    CBuilderExpr lag(int) except +
    CBuilderExpr lead(int) except +
    CBuilderExpr firstValue() except +
    CBuilderExpr lastValue() except +

    CBuilderExpr extract(const string&) except +

    CBuilderExpr cast(const CType*) except +
    CBuilderExpr castByStr "cast"(const string&) except +

    CBuilderExpr logicalNot() except +
    CBuilderExpr uminus() except +
    CBuilderExpr isNull() except +
    CBuilderExpr unnest() except +

    CBuilderExpr ceil() except +
    CBuilderExpr floor() except +

    CBuilderExpr pow(const CBuilderExpr&) except +

    CBuilderExpr add(const CBuilderExpr&) except +
    CBuilderExpr sub(const CBuilderExpr&) except +
    CBuilderExpr mul(const CBuilderExpr&) except +
    CBuilderExpr div(const CBuilderExpr&) except +
    CBuilderExpr mod(const CBuilderExpr&) except +

    CBuilderExpr addDate "add"(const CBuilderExpr&, const string&) except +
    CBuilderExpr subDate "sub"(const CBuilderExpr&, const string&) except +

    CBuilderExpr logicalAnd(const CBuilderExpr&) except +
    CBuilderExpr logicalOr(const CBuilderExpr&) except +

    CBuilderExpr eq(const CBuilderExpr&) except +
    CBuilderExpr ne(const CBuilderExpr&) except +
    CBuilderExpr lt(const CBuilderExpr&) except +
    CBuilderExpr le(const CBuilderExpr&) except +
    CBuilderExpr gt(const CBuilderExpr&) except +
    CBuilderExpr ge(const CBuilderExpr&) except +

    CBuilderExpr at(const CBuilderExpr&) except +

    CBuilderExpr over(const vector[CBuilderExpr]&) except +
    CBuilderExpr orderBy(const vector[CBuilderOrderByKey]&) except +

  cdef cppclass CBuilderSortField "hdk::ir::BuilderSortField":
    CBuilderSortField(CBuilderExpr, const string&, const string&) except +

  cdef cppclass CBuilderNode "hdk::ir::BuilderNode":
    CBuilderNode()

    CNodePtr node() const
    int size() const
    CColumnInfoPtr columnInfoByIndex "columnInfo"(int) except +
    CColumnInfoPtr columnInfoByName "columnInfo"(const string&) except +
    size_t rowCount() except +

    CBuilderExpr refByIndex "ref"(int) except +
    CBuilderExpr refByName "ref"(const string&) except +

    CBuilderNode proj(const vector[CBuilderExpr]&) except +

    CBuilderExpr parseAggString(const string&) except +
    CBuilderNode agg(const vector[CBuilderExpr]&, const vector[CBuilderExpr]&) except +
    CBuilderNode filter(CBuilderExpr condition) except +
    CBuilderNode joinBySameCols "join"(const CBuilderNode&, const string&) except +
    CBuilderNode joinByCols "join"(const CBuilderNode&, const vector[string]&, const string&) except +
    CBuilderNode joinByColPairs "join"(const CBuilderNode&, const vector[string]&, const vector[string]&, const string&) except +
    CBuilderNode joinByCond "join"(const CBuilderNode&, const CBuilderExpr&, const string&) except +

    CBuilderNode sort(const vector[CBuilderSortField]&, size_t, size_t) except +

    unique_ptr[CQueryDag] finalize() except +

  cdef cppclass CQueryBuilder "hdk::ir::QueryBuilder":
    CQueryBuilder(CContext&, CSchemaProviderPtr, CConfigPtr)

    CBuilderNode scan(const string&) except +
    CBuilderNode proj(const vector[CBuilderExpr]&) except +

    CBuilderExpr count() except +
    CBuilderExpr rowNumber() except +
    CBuilderExpr rank() except +
    CBuilderExpr denseRank() except +
    CBuilderExpr percentRank() except +
    CBuilderExpr nTile(int) except +

    CBuilderExpr cstFromIntNoType "cst"(int64_t) except +
    CBuilderExpr cstFromIntNoScale "cstNoScale"(int64_t, const CType*) except +
    CBuilderExpr cstFromInt "cst"(int64_t, const CType*) except +
    CBuilderExpr cstFromFpNoType "cst"(double) except +
    CBuilderExpr cstFromFp "cst"(double, const CType*) except +
    CBuilderExpr cstFromStrNoType "cst"(const string&) except +
    CBuilderExpr cstFromStr "cst"(const string&, const CType*) except +
    CBuilderExpr cstArray "cst"(const vector[CBuilderExpr]&, const CType*) except +
    CBuilderExpr trueCst() except +
    CBuilderExpr falseCst() except +
    CBuilderExpr nullCstNoType "nullCst"() except +
    CBuilderExpr nullCst(const CType*) except +
    CBuilderExpr date(const string&) except +
    CBuilderExpr time(const string&) except +
    CBuilderExpr timestamp(const string&) except +
