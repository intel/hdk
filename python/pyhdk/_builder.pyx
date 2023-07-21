#
# Copyright 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr, make_unique
from libcpp.utility cimport move

from pyhdk._storage cimport SchemaProvider, CColumnInfoPtr, ColumnInfo
from pyhdk._common cimport Config, TypeInfo, CContext
from pyhdk._ir cimport CNode, CScan, CExpr, CColumnRefExpr
from pyhdk._sql cimport CExecutionResult, ExecutionResult, CQueryDag, QueryDag, RelAlgExecutor

from collections.abc import Iterable

cdef class QueryExpr:
  cdef CBuilderExpr c_expr

  @property
  def is_ref(self):
    return self.c_expr.expr().get().isExpr[CColumnRefExpr]()

  @property
  def index(self):
    if not self.is_ref:
      raise RuntimeError(f"Only column references implement 'index' property. Actual expression: {self}")
    return self.c_expr.expr().get().asExpr[CColumnRefExpr]().index()

  @property
  def type(self):
    res = TypeInfo()
    res.c_type_info = self.c_expr.expr().get().type()
    return res

  @property
  def name(self):
    return self.c_expr.name()

  def rename(self, name):
    if not isinstance(name, str):
      raise TypeError(f"Only string names are allowed for expressions. Provided: {type(name)}.")
    res = QueryExpr();
    res.c_expr = self.c_expr.rename(name)
    return res

  def avg(self):
    res = QueryExpr();
    res.c_expr = self.c_expr.avg()
    return res

  def min(self):
    res = QueryExpr();
    res.c_expr = self.c_expr.min()
    return res

  def max(self):
    res = QueryExpr();
    res.c_expr = self.c_expr.max()
    return res

  def sum(self):
    res = QueryExpr();
    res.c_expr = self.c_expr.sum()
    return res

  def count(self, is_distinct=False, approx=False):
    if not isinstance(is_distinct, int):
      raise TypeError("Use True or False for 'is_distinct' arg.")
    if not isinstance(approx, int):
      raise TypeError("Use True or False for 'approx' arg.")

    res = QueryExpr()
    if approx and is_distinct:
      res.c_expr = self.c_expr.approxCountDist()
    else:
      res.c_expr = self.c_expr.count(is_distinct)
    return res

  def approx_quantile(self, prob):
    if not isinstance(prob, (float, int)):
      raise TypeError(f"Float number expected for 'prob' argument. Provided: {type(prob)}.")
    prob = float(prob)
    if prob < 0.0 or prob > 1.0:
      raise ValueError(f"Expected 'prob' to be in [0, 1] range. Provided: {prob}.")
    res = QueryExpr();
    res.c_expr = self.c_expr.approxQuantile(prob)
    return res

  def sample(self):
    res = QueryExpr();
    res.c_expr = self.c_expr.sample()
    return res

  def single_value(self):
    res = QueryExpr();
    res.c_expr = self.c_expr.singleValue()
    return res

  def top_k(self, count):
    if not isinstance(count, int) or count == 0:
      raise TypeError(f"Expected non-zero integer value as count arg. Provided: {type(count)}.")
    res = QueryExpr()
    res.c_expr = self.c_expr.topK(count)
    return res

  def bottom_k(self, count):
    if not isinstance(count, int) or count == 0:
      raise TypeError(f"Expected non-zero integer value as count arg. Provided: {type(count)}.")
    res = QueryExpr()
    res.c_expr = self.c_expr.bottomK(count)
    return res

  def stddev(self):
    res = QueryExpr();
    res.c_expr = self.c_expr.stdDev()
    return res

  def corr(self, arg):
    if not isinstance(arg, QueryExpr):
      raise TypeError(f"Expected QueryExpr for corr arg. Provided: {type(arg)}.")
    res = QueryExpr();
    res.c_expr = self.c_expr.corr((<QueryExpr>arg).c_expr)
    return res

  def lag(self, int n = 1):
    res = QueryExpr();
    res.c_expr = self.c_expr.lag(n)
    return res

  def lead(self, int n = 1):
    res = QueryExpr();
    res.c_expr = self.c_expr.lead(n)
    return res

  def first_value(self):
    res = QueryExpr();
    res.c_expr = self.c_expr.firstValue()
    return res

  def last_value(self):
    res = QueryExpr();
    res.c_expr = self.c_expr.lastValue()
    return res

  def logical_and(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.logicalAnd((<QueryExpr>value).c_expr)
    return res

  def logical_or(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.logicalOr((<QueryExpr>value).c_expr)
    return res

  def logical_not(self):
    res = QueryExpr()
    res.c_expr = self.c_expr.logicalNot()
    return res

  def bw_and(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.bwAnd((<QueryExpr>value).c_expr)
    return res

  def bw_or(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.bwOr((<QueryExpr>value).c_expr)
    return res

  def bw_xor(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.bwXor((<QueryExpr>value).c_expr)
    return res

  def bw_not(self):
    res = QueryExpr()
    res.c_expr = self.c_expr.bwNot()
    return res

  def is_null(self):
    res = QueryExpr()
    res.c_expr = self.c_expr.isNull()
    return res

  def is_not_null(self):
    return self.is_null().logical_not()

  def eq(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.eq((<QueryExpr>value).c_expr)
    return res

  def ne(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.ne((<QueryExpr>value).c_expr)
    return res

  def lt(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.lt((<QueryExpr>value).c_expr)
    return res

  def le(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.le((<QueryExpr>value).c_expr)
    return res

  def gt(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.gt((<QueryExpr>value).c_expr)
    return res

  def ge(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.ge((<QueryExpr>value).c_expr)
    return res

  def uminus(self):
    res = QueryExpr()
    res.c_expr = self.c_expr.uminus()
    return res

  def add(self, value, field=None):
    value = self._process_op_expr(value)
    res = QueryExpr()
    if field is None:
      res.c_expr = self.c_expr.add((<QueryExpr>value).c_expr)
    else:
      if not isinstance(field, str):
        raise TypeError(f"Expected str for 'field' arg. Provided: {type(field)}.")
      res.c_expr = self.c_expr.addDate((<QueryExpr>value).c_expr, field)
    return res

  def sub(self, value, field=None):
    value = self._process_op_expr(value)
    res = QueryExpr()
    if field is None:
      res.c_expr = self.c_expr.sub((<QueryExpr>value).c_expr)
    else:
      if not isinstance(field, str):
        raise TypeError(f"Expected str for 'field' arg. Provided: {type(field)}.")
      res.c_expr = self.c_expr.subDate((<QueryExpr>value).c_expr, field)
    return res

  def mul(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.mul((<QueryExpr>value).c_expr)
    return res

  def truediv(self, value):
    value = self._process_op_expr(value)
    if self.c_expr.expr().get().type().isInteger() and (<QueryExpr>value).c_expr.expr().get().type().isInteger():
      value = value.cast("fp64")
    res = QueryExpr()
    res.c_expr = self.c_expr.div((<QueryExpr>value).c_expr)
    return res

  def floordiv(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.div((<QueryExpr>value).c_expr)
    if not res.c_expr.expr().get().type().isInteger():
      res.c_expr = res.c_expr.floor()
    return res

  def div(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.div((<QueryExpr>value).c_expr)
    return res

  def mod(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.mod((<QueryExpr>value).c_expr)
    return res

  def cast(self, new_type):
    res = QueryExpr()
    if isinstance(new_type, str):
      res.c_expr = self.c_expr.castByStr(new_type)
    elif isinstance(new_type, TypeInfo):
      res.c_expr = self.c_expr.cast((<TypeInfo>new_type).c_type_info)
    else:
      raise TypeError(f"Expected TypeInfo or str for 'new_type' arg. Provided: {type(new_type)}.")
    return res

  def extract(self, field):
    if not isinstance(field, str):
      raise TypeError(f"Expected str for 'field' arg. Provided: {type(field)}.")

    res = QueryExpr()
    res.c_expr = self.c_expr.extract(field)
    return res

  def unnest(self):
    res = QueryExpr()
    res.c_expr = self.c_expr.unnest()
    return res

  def pow(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.pow((<QueryExpr>value).c_expr)
    return res

  def at(self, value):
    value = self._process_op_expr(value)
    res = QueryExpr()
    res.c_expr = self.c_expr.at((<QueryExpr>value).c_expr)
    return res

  def cardinality(self):
    res = QueryExpr()
    res.c_expr = self.c_expr.cardinality()
    return res

  def over(self, *args):
    cdef vector[CBuilderExpr] keys
    for arg in args:
      if not isinstance(arg, QueryExpr):
        raise TypeError(f"Expected QueryExpr arg for 'over' method. Provided: {type(arg)}.")
      keys.push_back((<QueryExpr>arg).c_expr)
    res = QueryExpr()
    res.c_expr = self.c_expr.over(keys)
    return res

  def order_by(self, *args):
    cdef vector[CBuilderOrderByKey] keys
    for arg in args:
      if isinstance(arg, tuple):
        key = arg[0]
        if not isinstance(key, QueryExpr):
          raise TypeError(f"Expected QueryExpr for order_by key. Provided: {type(key)}.")
        order = arg[1] if len(arg) > 1 else "asc"
        if not isinstance(order, str):
          raise TypeError(f"Expected 'asc' or 'desc' for sort order. Provided: {order}.")
        null_pos = arg[2] if len(arg) > 2 else "last"
        if not isinstance(null_pos, str):
          raise TypeError(f"Expected 'first' or 'last' for nulls position. Provided: {null_pos}.")
      else:
        if not isinstance(arg, QueryExpr):
          raise TypeError(f"Expected QueryExpr or tuple arg for 'order_by' method. Provided: {type(arg)}.")
        key = arg
        order = "asc"
        null_pos = "last"
      keys.push_back(CBuilderOrderByKey((<QueryExpr>key).c_expr, order, null_pos))
    res = QueryExpr()
    res.c_expr = self.c_expr.orderBy(keys)
    return res

  def __eq__(self, value):
    return self.eq(value)

  def __ne__(self, value):
    return self.ne(value)

  def __lt__(self, value):
    return self.lt(value)

  def __le__(self, value):
    return self.le(value)

  def __gt__(self, value):
    return self.gt(value)

  def __ge__(self, value):
    return self.ge(value)

  def __neg__(self):
    return self.uminus()

  def __add__(self, value):
    return self.add(value)

  def __sub__(self, value):
    return self.sub(value)

  def __mul__(self, value):
    return self.mul(value)

  def __floordiv__(self, value):
    return self.floordiv(value)

  def __truediv__(self, value):
    return self.truediv(value)

  def __mod__(self, value):
    return self.mod(value)

  def __getitem__(self, value):
    return self.at(value)

  def _process_op_expr(self, op):
    if isinstance(op, QueryExpr):
      return op

    res = QueryExpr()
    if type(op) == type(True):
      if op:
        res.c_expr = self.c_expr.builder().trueCst()
      else:
        res.c_expr = self.c_expr.builder().falseCst()
    elif isinstance(op, int):
      res.c_expr = self.c_expr.builder().cstFromIntNoType(op)
    elif isinstance(op, float):
      res.c_expr = self.c_expr.builder().cstFromFpNoType(op)
    elif isinstance(op, str):
      res.c_expr = self.c_expr.builder().cstFromStrNoType(op)
    else:
      raise TypeError(f"Expected operand of type int, float, bool, str or QueryExpr. Provided: {type(op)}.")

    return res

  def __repr__(self):
    return self.c_expr.expr().get().toString()

cdef class QueryNode:
  cdef CBuilderNode c_node
  cdef object _hdk

  def ref(self, col):
    res = QueryExpr()
    if isinstance(col, int):
      # skip virtual 'rowid' column
      if self.is_scan and col < 0:
        col = col - 1
      res.c_expr = self.c_node.refByIndex(col)
    elif isinstance(col, str):
      res.c_expr = self.c_node.refByName(col)
    else:
      raise TypeError(f"Expected int or str for 'col' argument. Provided: {col}")
    return res

  def __getitem__(self, col):
    return self.ref(col)

  def proj(self, *args, exprs=None, **kwargs):
    cdef vector[CBuilderExpr] proj_exprs

    for arg in args:
      expr = self._process_expr_arg(arg)
      proj_exprs.push_back((<QueryExpr>expr).c_expr)

    if exprs is not None:
      if not isinstance(exprs, dict):
        raise TypeError(f"Dict expected for 'expr' argument. Provided: {type(exprs)}.")

      for key, arg in exprs.items():
        if not isinstance(key, str):
          raise TypeError(f"String expected for expr key. Provided: {type(key)}.")
        expr = self._process_expr_arg(arg).rename(key)
        proj_exprs.push_back((<QueryExpr>expr).c_expr)

    for key, arg in kwargs.items():
      expr = self._process_expr_arg(arg).rename(key)
      proj_exprs.push_back((<QueryExpr>expr).c_expr)

    res = QueryNode()
    res.c_node = self.c_node.proj(proj_exprs)
    res._hdk = self._hdk
    return res

  def agg(self, group_keys, *args, aggs=None, **kwargs):
    cdef vector[CBuilderExpr] key_exprs
    cdef vector[CBuilderExpr] agg_exprs

    if isinstance(group_keys, (int, str, QueryExpr)):
      expr = self._process_expr_arg(group_keys)
      key_exprs.push_back((<QueryExpr>expr).c_expr)
    elif isinstance(group_keys, Iterable):
      for key in group_keys:
        expr = self._process_expr_arg(key)
        key_exprs.push_back((<QueryExpr>expr).c_expr)
    else:
      raise TypeError(f"Expected int, str, QueryExpr or iterable for 'group_keys' arg. Provided: {type(group_keys)}")

    for arg in args:
      expr = self._process_agg_expr_arg(arg)
      agg_exprs.push_back((<QueryExpr>expr).c_expr)

    if aggs is not None:
      if not isinstance(aggs, dict):
        raise TypeError(f"Dict expected for 'aggs' argument. Provided: {type(aggs)}.")

      for key, arg in aggs.items():
        if not isinstance(key, str):
          raise TypeError(f"String expected for expr key. Provided: {type(key)}.")
        expr = self._process_agg_expr_arg(arg).rename(key)
        agg_exprs.push_back((<QueryExpr>expr).c_expr)

    for key, arg in kwargs.items():
      expr = self._process_agg_expr_arg(arg).rename(key)
      agg_exprs.push_back((<QueryExpr>expr).c_expr)

    res = QueryNode()
    res.c_node = self.c_node.agg(key_exprs, agg_exprs)
    res._hdk = self._hdk
    return res

  def sort(self, *args, fields=None, limit=0, offset=0, **kwargs):
    cdef vector[CBuilderSortField] sort_fields

    if not isinstance(limit, int):
      raise TypeError(f"Non-negative integer is expected for 'limit' argument. Provided: {type(limit)}")
    if limit < 0:
      raise ValueError(f"Non-negative integer is expected for 'limit' argument. Provided: {limit}")
    if not isinstance(offset, int):
      raise TypeError(f"Non-negative integer is expected for 'offset' argument. Provided: {type(offset)}")
    if offset < 0:
      raise ValueError(f"Non-negative integer is expected for 'offset' argument. Provided: {offset}")

    for arg in args:
      expr, sort_dir, null_pos = self._process_sort_field_arg(arg)
      sort_fields.push_back(CBuilderSortField((<QueryExpr>expr).c_expr, <str>sort_dir, <str>null_pos))

    if fields is not None:
      if not isinstance(fields, dict):
        raise TypeError(f"Dict expected for 'fields' argument. Provided: {type(fields)}.")

      for key, arg in fields.items():
        if not isinstance(key, str):
          raise TypeError(f"String expected for field key. Provided: {type(key)}.")
        expr, sort_dir, null_pos = self._process_sort_field_arg(arg, key)
        sort_fields.push_back(CBuilderSortField((<QueryExpr>expr).c_expr, <str>sort_dir, <str>null_pos))

    for key, arg in kwargs.items():
      expr, sort_dir, null_pos = self._process_sort_field_arg(arg, key)
      sort_fields.push_back(CBuilderSortField((<QueryExpr>expr).c_expr, <str>sort_dir, <str>null_pos))

    res = QueryNode()
    res.c_node = self.c_node.sort(sort_fields, limit, offset)
    res._hdk = self._hdk
    return res

  def filter(self, *args):
    if len(args) == 0:
      raise ValueError("You should provide at least one filter condition.")

    cond = None
    for arg in args:
      if not isinstance(arg, QueryExpr):
        raise TypeError(f"Filter condition should be QueryExpr. Provided: {type(arg)}")
      if cond is None:
        cond = arg
      else:
        cond = cond.logical_and(arg)

    res = QueryNode()
    res.c_node = self.c_node.filter((<QueryExpr>cond).c_expr)
    res._hdk = self._hdk
    return res

  def join(self, rhs_node, lhs_cols=None, rhs_cols=None, cond=None, how="inner"):
    if isinstance(rhs_node, ExecutionResult):
      rhs_node = rhs_node.scan
    if not isinstance(rhs_node, QueryNode):
      raise TypeError(f"Expected QueryNode for 'rhs_node' arg. Provided: {type(rhs_node)}.")
    if not isinstance(how, str):
      raise TypeError(f"Expected str for 'how' arg. Provided: {type(how)}.")

    cdef vector[string] lhs_col_names
    cdef vector[string] rhs_col_names

    res = QueryNode()
    if lhs_cols is None:
      if rhs_cols is not None:
        raise ValueError("Mismatch length of 'lhs_cols' and 'rhs_cols'.")

      if cond is None:
        res.c_node = self.c_node.joinBySameCols((<QueryNode>rhs_node).c_node, how)
      else:
        if not isinstance(cond, QueryExpr):
          raise TypeError(f"Expected QueryExpr for 'cond' arg. Provided: {type(cond)}.")
        res.c_node = self.c_node.joinByCond((<QueryNode>rhs_node).c_node, (<QueryExpr>cond).c_expr, how)
    else:
      if isinstance(lhs_cols, str) or not isinstance(lhs_cols, Iterable):
        lhs_cols = [lhs_cols]

      for lhs_col in lhs_cols:
        if not isinstance(lhs_col, str):
          raise TypeError("Only strings are allowed in lhs_cols. Provided: {type(lhs_col)}.")
        lhs_col_names.push_back(lhs_col)

      if rhs_cols is None:
        res.c_node = self.c_node.joinByCols((<QueryNode>rhs_node).c_node, lhs_col_names, how)
      else:
        if isinstance(rhs_cols, str) or not isinstance(rhs_cols, Iterable):
          rhs_cols = [rhs_cols]

        if len(lhs_cols) != len(rhs_cols):
          raise ValueError("Mismatch length of 'lhs_cols' and 'rhs_cols'.")

        for rhs_col in rhs_cols:
          if not isinstance(rhs_col, str):
            raise TypeError("Only strings are allowed in rhs_cols. Provided: {type(rhs_col)}.")
          rhs_col_names.push_back(rhs_col)

        res.c_node = self.c_node.joinByColPairs((<QueryNode>rhs_node).c_node, lhs_col_names, rhs_col_names, how)

    res._hdk = self._hdk
    return res


  def _process_expr_arg(self, expr):
    if isinstance(expr, (int, str)):
      return self.ref(expr)
    elif isinstance(expr, QueryExpr):
      return expr
    else:
      raise TypeError(f"Unexpected query expression. Expected int, str or QueryExpr. Provided: {type(expr)}")

  def _process_agg_expr_arg(self, expr):
    if isinstance(expr, str):
      res = QueryExpr()
      res.c_expr = self.c_node.parseAggString(expr)
      return res
    elif isinstance(expr, QueryExpr):
      return expr
    else:
      raise TypeError(f"Unexpected aggregate query expression. Expected str or QueryExpr. Provided: {type(expr)}")

  def _process_sort_field_arg(self, field, key=None):
    sort_dir = "asc"
    null_pos = "last"

    if key is None:
      if isinstance(field, tuple):
        if len(field) < 1 or len(field) > 3:
          raise TypeError(f"Sort field tuple should have 1 to 3 elements but got {len(field)}.")
        expr = self._process_expr_arg(field[0])
        if len(field) > 1:
          sort_dir = field[1]
        if len(field) > 2:
          null_pos = field[2]
      else:
        expr = self._process_expr_arg(field)
    else:
      expr = self._process_expr_arg(key)
      if isinstance(field, tuple):
        if len(field) < 1 or len(field) > 2:
          raise TypeError(f"Sort direction tuple should have 1 to 2 elements but got {len(field)}.")
        sort_dir = field[0]
        if len(field) > 1:
          null_pos = field[1]
      else:
        sort_dir = field

    if not isinstance(sort_dir, str):
      raise TypeError(f"Sort direction should be a string. Provided: {sort_dir}.")

    if not isinstance(null_pos, str):
      raise TypeError(f"Nulls position should be a string. Provided: {null_pos}.")

    return expr, sort_dir, null_pos

  @property
  def is_scan(self):
    return self.c_node.node().get().isNode[CScan]()

  @property
  def size(self):
    res = self.c_node.size()
    # For scans we hide virtual column
    if self.is_scan:
      assert res > 0
      res = res - 1
    return res

  @property
  def shape(self):
    if not self.is_scan:
      raise RuntimeError("Only scan nodes provide shape.")
    return (self.c_node.rowCount(), self.size)

  def column_info(self, col):
    cdef CExpr *c_expr
    if isinstance(col, QueryExpr) and col.is_ref:
      col = col.index

    res = ColumnInfo()
    if isinstance(col, int):
      # skip virtual 'rowid' column
      if self.is_scan and col < 0:
        col = col - 1
      res.c_column_info = self.c_node.columnInfoByIndex(col)
    elif isinstance(col, str):
      res.c_column_info = self.c_node.columnInfoByName(col)
    else:
      raise TypeError(f"Only int, str and column references are allowed for 'col' arg. Provided: {col}")

    return res

  @property
  def table_name(self):
    if self.is_scan:
      return self.c_node.node().get().asNode[CScan]().getTableInfo().get().name
    return None

  @property
  def schema(self):
    res = dict()
    for col_idx in range(self.size):
      col_info = self.column_info(col_idx)
      res[col_info.name] = col_info
    return res

  def run(self, **kwargs):
    assert self._hdk is not None
    # Cannot assign unique_ptr here because Cython uses additional
    # intermediate variable and copy assignment for that. Use
    # release + reset instead.
    cdef CQueryDag* c_dag = self.c_node.finalize().release()
    dag = QueryDag()
    dag.c_dag.reset(c_dag)
    rel_alg_executor = RelAlgExecutor(self._hdk._executor, self._hdk._storage, self._hdk._data_mgr, dag=dag)
    res = rel_alg_executor.execute(**kwargs)
    res.scan = self._hdk.scan(res.table_name)
    return res

  def finalize(self):
    cdef CQueryDag* c_dag = self.c_node.finalize().release()
    dag = QueryDag()
    dag.c_dag.reset(c_dag)
    return dag

  @property
  def hdk(self):
    return self._hdk

  def __repr__(self):
    return self.c_node.node().get().toString()

cdef class QueryBuilder:
  cdef unique_ptr[CQueryBuilder] c_builder
  cdef object _hdk

  def __cinit__(self, SchemaProvider schema_provider, Config config, object hdk=None):
    self.c_builder = make_unique[CQueryBuilder](CContext.defaultCtx(), schema_provider.c_schema_provider, config.c_config)
    self._hdk = hdk

  def scan(self, table_name):
    res = QueryNode()
    res.c_node = self.c_builder.get().scan(table_name)
    res._hdk = self._hdk
    return res

  def typeFromString(self, type_str):
    if not isinstance(type_str, str):
      raise TypeError(f"Only strings are supported for 'type_str' arg. Provided: {type_str}")

    res = TypeInfo()
    res.c_type_info = CContext.defaultCtx().typeFromString(type_str)
    return res

  def count(self):
    res = QueryExpr()
    res.c_expr = self.c_builder.get().count()
    return res

  def row_number(self):
    res = QueryExpr()
    res.c_expr = self.c_builder.get().rowNumber()
    return res

  def rank(self):
    res = QueryExpr()
    res.c_expr = self.c_builder.get().rank()
    return res

  def dense_rank(self):
    res = QueryExpr()
    res.c_expr = self.c_builder.get().denseRank()
    return res

  def percent_rank(self):
    res = QueryExpr()
    res.c_expr = self.c_builder.get().percentRank()
    return res

  def ntile(self, int tile_count):
    res = QueryExpr()
    res.c_expr = self.c_builder.get().nTile(tile_count)
    return res

  def cst(self, value, cst_type=None, scale_decimal=True):
    if isinstance(cst_type, str):
      cst_type = self.typeFromString(cst_type)
    if cst_type is not None and not isinstance(cst_type, TypeInfo):
      raise TypeError("Arguments 'cst_type' doesn't represent a type.")

    if not isinstance(scale_decimal, int):
      raise TypeError("Use True or False for 'scale_decimal' arg.")

    cdef const CType *c_type = NULL
    cdef vector[CBuilderExpr] elems

    if cst_type is not None:
      c_type = (<TypeInfo>cst_type).c_type_info

    res = QueryExpr()
    if value is None:
      if cst_type is None:
        res.c_expr = self.c_builder.get().nullCstNoType()
      else:
        res.c_expr = self.c_builder.get().nullCst(c_type)
    elif type(value) == type(True):
      if cst_type is None:
        if value:
          res.c_expr = self.c_builder.get().trueCst()
        else:
          res.c_expr = self.c_builder.get().falseCst()
      else:
        res.c_expr = self.c_builder.get().cstFromInt(value, c_type)
    elif isinstance(value, int):
      if cst_type is None:
        res.c_expr = self.c_builder.get().cstFromIntNoType(value)
      else:
        if scale_decimal:
          res.c_expr = self.c_builder.get().cstFromInt(value, c_type)
        else:
          res.c_expr = self.c_builder.get().cstFromIntNoScale(value, c_type)
    elif isinstance(value, float):
      if cst_type is None:
        res.c_expr = self.c_builder.get().cstFromFpNoType(value)
      else:
        res.c_expr = self.c_builder.get().cstFromFp(value, c_type)
    elif isinstance(value, str):
      if cst_type is None:
        res.c_expr = self.c_builder.get().cstFromStrNoType(value)
      else:
        res.c_expr = self.c_builder.get().cstFromStr(value, c_type)
    elif isinstance(value, list):
      if cst_type is not None and not cst_type.is_array:
        raise TypeError("Array constant should have array type.")

      elem_type = cst_type.elem_type if cst_type is not None else None
      for elem_value in value:
        elem = self.cst(elem_value, elem_type, scale_decimal)
        elems.push_back((<QueryExpr>elem).c_expr)
        if elem_type is None:
          elem_type = elem.type

      if elem_type is None:
        raise RuntimeError("Constant type should be provided for empty arrays.")

      if cst_type is None:
        c_type = CContext.defaultCtx().arrayVarLen((<TypeInfo>elem_type).c_type_info, 4, False)

      res.c_expr = self.c_builder.get().cstArray(elems, c_type)
    else:
      raise TypeError("Only None, bool, int, fp, and str types are supported for constant value.")

    return res

  def date(self, value):
    res = QueryExpr()
    res.c_expr = self.c_builder.get().date(value)
    return res

  def time(self, value):
    res = QueryExpr()
    res.c_expr = self.c_builder.get().time(value)
    return res

  def timestamp(self, value):
    res = QueryExpr()
    res.c_expr = self.c_builder.get().timestamp(value)
    return res
