#
# Copyright 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from pyhdk._common import buildConfig
from pyhdk._storage import TableOptions, ArrowStorage, DataMgr
from pyhdk._sql import Calcite, RelAlgExecutor
from pyhdk._execute import Executor
from pyhdk._builder import QueryBuilder, QueryExpr, QueryNode

import pyarrow
import uuid
from collections.abc import Iterable


def not_implemented(func):
    def wrapper(*args, **kwargs):
        raise NotImplementedError(f"{func.__qualname__} is not yet implemented.")

    return wrapper


class QueryExprAPI:
    def rename(self, name):
        """
        Create a copy of the expression with a new assigned name.

        Expression names are used by projection and aggregation nodes as
        the resulting column names. If name is not explicitly specified,
        then it is automatically chosen basing on the expression.

        Parameters
        ----------
        name : str
            New expression name.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> import pyhdk
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"a": [1, 1, 2], "b": [1, 2, 3]})
        >>> ht.proj("a", "b", (ht["a"] + ht["b"]).rename("sum")).run()
        Schema:
          a: INT64
          b: INT64
          sum: INT64
        Data:
        1|1|2
        1|2|3
        2|3|5

        >>> ht.agg(["a"], hdk.count().rename("c")).run()
        Schema:
          a: INT64
          c: INT32[NN]
        Data:
        1|2
        2|1
        """
        pass

    def avg(self):
        """
        Create AVG aggregate expression with the current expression as its
        argument.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, 11, 5]})
        >>> ht.agg(["id"], ht["x"].avg()).run()
        Schema:
          id: INT64
          x_avg: FP64
        Data:
        1|6
        2|9
        """
        pass

    def min(self):
        """
        Create MIN aggregate expression with the current expression as its
        argument.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, 11, 5]})
        >>> ht.agg(["id"], ht["x"].min()).run()
        Schema:
          id: INT64
          x_min: INT64
        Data:
        1|4
        2|7
        """
        pass

    def max(self):
        """
        Create MAX aggregate expression with the current expression as its
        argument.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, 11, 5]})
        >>> ht.agg(["id"], ht["x"].max()).run()
        Schema:
          id: INT64
          x_max: INT64
        Data:
        1|9
        2|11
        """
        pass

    def sum(self):
        """
        Create SUM aggregate expression with the current expression as its
        argument.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, 11, 5]})
        >>> ht.agg(["id"], ht["x"].sum()).run()
        Schema:
          id: INT64
          x_sum: INT64
        Data:
        1|18
        2|18
        """
        pass

    def count(self, is_distinct=False, approx=False):
        """
        Create COUNT, COUNT DISTINCT or APPROX COUNT DISTINCT aggregate expression
        with the current expression as its argument.

        Parameters
        ----------
        is_distinct : bool, default: False
            Set to True for the DISTINCT COUNT variant of the aggregate.
        approx : bool, default: False
            Set to True when approximation should be used. Allowed only for DISTINCT
            COUNT aggregate. Provides less acuracy but better performance.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, None, 5]})
        >>> ht.agg(["id"], ht["x"].count()).run()
        Schema:
          id: INT64
          x_count: INT32[NN]
        Data:
        1|3
        2|1
        """
        pass

    def approx_quantile(self, prob):
        """
        Create APROX QUANTILE aggregate expression with the current expression as
        its argument.

        Parameters
        ----------
        prob : float
            Quantile probability. Should be in [0, 1] range.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"id": [1, 2, 1, 2, 1, 2], "x": [4, 7, 9, 11, 5, 13]})
        >>> ht.agg(["id"], ht["x"].approx_quantile(0.5), ht["x"].approx_quantile(1.0)).run()
        Schema:
          id: INT64
          x_approx_quantile: FP64
          x_approx_quantile_1: FP64
        Data:
        1|5|9
        2|11|13
        """
        pass

    def sample(self):
        """
        Create SAMPLE aggregate expression with the current expression as its
        argument.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, 11, 5]})
        >>> ht.agg(["id"], ht["x"].sample()).run()
        Schema:
          id: INT64
          x_sample: INT64
        Data:
        1|5
        2|11
        """
        pass

    def single_value(self):
        """
        Create SINGLE VALUE aggregate expression with the current expression as its
        argument.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 5, 4, 5, 4]})
        >>> ht.agg(["id"], ht["x"].single_value()).run()
        Schema:
          id: INT64
          x_single_value: INT64
        Data:
        1|4
        2|5
        """
        pass

    def extract(self, field):
        """
        Create EXTRACT expression to extract a part of date from the current expression.

        Parameters
        ----------
        field : str
            The part of date to extract. Supported values:
            - "year"
            - "quarter"
            - "month"
            - "day"
            - "hour"
            - "min", "minute"
            - "sec", "second"
            - "ms", "milli", "millisecond"
            - "us", "micro", "microsecond"
            - "ns", "nano", "nanosecond"
            - "dow", "dayofweek", "day of week"
            - "isodow", "isodayofweek", "iso day of week"
            - "doy", "dayofyear", "day of year"
            - "epoch"
            - "quarterday", "quarter day"
            - "week"
            - "weeksunday", "week sunday"
            - "weeksaturday", "week saturday"
            - "dateepoch", "date epoch"

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"a": pandas.to_datetime(["20230207", "20220308"], format="%Y%m%d")})
        >>> ht.proj(
        ...     y=ht["a"].extract("year"),
        ...     m=ht["a"].extract("month"),
        ...     d=ht["a"].extract("day"),
        ... ).run()
        Schema:
          y: INT64
          m: INT64
          d: INT64
        Data:
        2023|2|7
        2022|3|8
        """
        pass

    def cast(self, new_type):
        """
        Create CAST expression for the current expression.

        For literal expressions cast operation is executed immediately and new
        literal expression is returned as a result. In other cases cast expression
        is created and returned.

        Parameters
        ----------
        new_type : str or TypeInfo
            Type expression should be casted to.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"a": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]})
        >>> ht.proj(ht["a"].cast("int")).run()
        Schema:
          expr_1: INT64
        Data:
        1
        2
        3
        4
        6
        7
        >>> hdk.cst("1970-01-01 01:00:00").cast("timestamp[ms]")
        (Const 1970-01-01 01:00:00.000)
        >>> hdk.cst("1970-01-01 01:00:00").cast("timestamp[s]").cast("int")
        (Const 3600)
        >>> hdk.cst("1970-01-01 01:00:00").cast("timestamp[ms]").cast("int")
        (Const 3600000)
        """
        pass

    def uminus(self):
        """
        Create unary minus expression for the current expression.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, 11, 5]})
        >>> ht.proj(ht["id"].uminus(), -ht["x"]).run()
        Schema:
          expr_1: INT64
          expr_2: INT64
        Data:
        -1|-4
        -2|-7
        -1|-9
        -2|-11
        -1|-5
        """
        pass

    def is_null(self):
        """
        Create IS NULL expression for the current expression.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"id": [1, 2, 3, 4, 5], "x": [4, None, 9, None, 5]})
        >>> ht.proj("id", ht["x"].is_null()).run()
        Schema:
          id: INT64
          expr_1: BOOL[NN]
        Data:
        1|0
        2|1
        3|0
        4|1
        5|0
        """
        pass

    def is_not_null(self):
        """
        Create IS NOT NULL expression for the current expression.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"id": [1, 2, 3, 4, 5], "x": [4, None, 9, None, 5]})
        >>> ht.proj("id", ht["x"].is_not_null()).run()
        Schema:
          id: INT64
          expr_1: BOOL[NN]
        Data:
        1|1
        2|0
        3|1
        4|0
        5|1
        """
        pass

    def unnest(self):
        """
        Create UNNEST expression for the current expression.

        UNNEST expression is used to flatten an array producing a row for each element in
        the array.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.create_table("test", [("a", "array(int)")])
        >>> hdk.import_pydict({"a": [[1, 2], [1, 2, 3, 4]]}, ht)
        >>> ht.proj(a=ht["a"].unnest()).agg(["a"], "count").run()
        Schema:
          a: INT64
          count: INT32[NN]
        Data:
        1|2
        2|2
        3|1
        4|1
        """
        pass

    def add(self, value, field=None):
        """
        Create ADD binary expression.

        Parameters
        ----------
        value : int, float, or QueryExpr
            Right-hand operand for the binary operation.
        field : str
            Used for datetime add operation to specify interval to add. Supported
            values:
            - "year", "years"
            - "quarter", "quarters"
            - "month", "months"
            - "day", "days"
            - "hour", "hours"
            - "min", "mins", "minute", "minutes"
            - "sec", "secs", "second", "seconds"
            - "millennium", "milleniums"
            - "century", "centuries"
            - "decade", "decades"
            - "ms", "milli", "millisecond", "milliseconds"
            - "us", "micro", "microsecond", "microseconds"
            - "ns", "nano", "nanosecond", "nanoseconds"
            - "week", "weeks"
            - "quarterday", "quarterdays", "quarter day", "quarter days"
            - "weekday", "weekdays", "week day", "week days"
            - "dayofyear", "day of year", "doy"

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"x": [1, 1, 1, 2, 2], "y": [5, 6, 7, 8, 9]})
        >>> ht.proj(ht["x"].add(ht["y"])).run()
        Schema:
          expr_1: INT64
        Data:
        6
        7
        8
        10
        11
        >>> ht.proj(ht["x"] + ht["y"]).run()
        Schema:
          expr_1: INT64
        Data:
        6
        7
        8
        10
        11
        >>> ht = hdk.import_pydict({"a": pandas.to_datetime(["20230207", "20220308"], format="%Y%m%d")})
        >>> ht.proj(ht["a"].add(1, "month")).run()
        Schema:
          expr_1: TIMESTAMP[ns]
        Data:
        2023-03-07 00:00:00.000000000
        2022-04-08 00:00:00.000000000
        """
        pass

    def sub(self, value, field=None):
        """
        Create SUB binary expression.

        Parameters
        ----------
        value : int, float, or QueryExpr
            Right-hand operand for the binary operation.
        field : str
            Used for datetime sub operation to specify interval to subtract.
            Supported values:
            - "year", "years"
            - "quarter", "quarters"
            - "month", "months"
            - "day", "days"
            - "hour", "hours"
            - "min", "mins", "minute", "minutes"
            - "sec", "secs", "second", "seconds"
            - "millennium", "milleniums"
            - "century", "centuries"
            - "decade", "decades"
            - "ms", "milli", "millisecond", "milliseconds"
            - "us", "micro", "microsecond", "microseconds"
            - "ns", "nano", "nanosecond", "nanoseconds"
            - "week", "weeks"
            - "quarterday", "quarterdays", "quarter day", "quarter days"
            - "weekday", "weekdays", "week day", "week days"
            - "dayofyear", "day of year", "doy"

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"x": [1, 1, 1, 2, 2], "y": [5, 6, 7, 8, 9]})
        >>> ht.proj(ht["y"].sub(ht["x"])).run()
        Schema:
          expr_1: INT64
        Data:
        4
        5
        6
        6
        7
        >>> ht.proj(ht["y"] - ht["x"]).run()
        Schema:
          expr_1: INT64
        Data:
        4
        5
        6
        6
        7
        >>> ht = hdk.import_pydict({"a": pandas.to_datetime(["20230207", "20220308"], format="%Y%m%d")})
        >>> ht.proj(ht["a"].sub(1, "hour")).run()
        Schema:
          expr_1: TIMESTAMP[ns]
        Data:
        2023-02-06 23:00:00.000000000
        2022-03-07 23:00:00.000000000
        """
        pass

    def mul(self, value):
        """
        Create MUL binary expression.

        Parameters
        ----------
        value : int, float, or QueryExpr
            Right-hand operand for the binary operation.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        >>> ht.proj(ht["a"].mul(ht["b"]), ht["a"] * 2, ht["a"] * 1.5).run()
        Schema:
          expr_1: INT64
          expr_2: INT64
          expr_3: FP64
        Data:
        5|2|1.5
        8|4|3
        9|6|4.5
        8|8|6
        5|10|7.5
        """
        pass

    def truediv(self, value):
        """
        Create DIV binary expression with a cast to float when required.

        If both operands have integer type then cast to fp64 is applied to the value
        and then DIV expression is generated. Otherwise, DIV expression is generated.

        Parameters
        ----------
        value : int, float, or QueryExpr
            Right-hand operand for the binary operation.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        >>> ht.proj(ht["a"].truediv(ht["b"]), ht["a"] / 2, ht["a"] / 2.0).run()
        Schema:
          expr_1: FP64
          expr_2: FP64
          expr_3: FP64
        Data:
        0.2|0.5|0.5
        0.5|1|1
        1|1.5|1.5
        2|2|2
        5|2.5|2.5
        """
        pass

    def floordiv(self, value):
        """
        Create DIV binary expression with an optional fllor operation.

        If both operands have integer type then DIV expression is generated. Otherwise,
        an additional floor operations is used for the result.

        Parameters
        ----------
        value : int, float, or QueryExpr
            Right-hand operand for the binary operation.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        >>> ht.proj(ht["a"].floordiv(ht["b"]), ht["a"] // 2, ht["a"] // 2.0).run()
        Schema:
          expr_1: INT64
          expr_2: INT64
          expr_3: FP64
        Data:
        0|0|0
        0|1|1
        1|1|1
        2|2|2
        5|2|2
        """
        pass

    def div(self, value):
        """
        Create DIV binary expression.

        Operation produces integer result when both operands are integers (floor div is applied).
        Otherwise, float result is produced (true div is applied).

        Parameters
        ----------
        value : int, float, or QueryExpr
            Right-hand operand for the binary operation.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        >>> ht.proj(ht["a"].div(ht["b"]), ht["a"].div(2), ht["a"].div(2.0)).run()
        Schema:
          expr_1: INT64
          expr_2: INT64
          expr_3: FP64
        Data:
        0|0|0.5
        0|1|1
        1|1|1.5
        2|2|2
        5|2|2.5
        """
        pass

    def mod(self, value):
        """
        Create MOD binary expression.

        Parameters
        ----------
        value : int, or QueryExpr
            Right-hand operand for the binary operation.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        >>> ht.proj(ht["a"].mod(ht["b"]), ht["a"] % 2).run()
        Schema:
          expr_1: INT64
          expr_2: INT64
        Data:
        1|1
        2|0
        0|1
        0|0
        0|1
        """
        pass

    def logical_not(self):
        """
        Create logical NOT expression.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"x": [True, False, True, False]})
        >>> ht.proj(ht["x"].logical_not()).run()
        Schema:
          expr_1: BOOL
        Data:
        0
        1
        0
        1
        """
        pass

    def logical_and(self, value):
        """
        Create logical AND expression.

        Parameters
        ----------
        value : QueryExpr
            Right-hand operand for the binary operation.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict(
        ...     {"x": [True, False, True, False], "y": [True, True, False, False]}
        ... )
        >>> ht.proj(ht["x"].logical_and(ht["y"])).run()
        Schema:
          expr_1: BOOL
        Data:
        1
        0
        0
        0
        """
        pass

    def logical_or(self, value):
        """
        Create logical OR expression.

        Parameters
        ----------
        value : QueryExpr
            Right-hand operand for the binary operation.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict(
        ...     {"x": [True, False, True, False], "y": [True, True, False, False]}
        ... )
        >>> ht.proj(ht["x"].logical_or(ht["y"])).run()
        Schema:
          expr_1: BOOL
        Data:
        1
        1
        1
        0
        """
        pass

    def eq(self, value):
        """
        Create EQUAL comparison expression.

        Parameters
        ----------
        value : int, float, str or QueryExpr
            Right-hand operand for the binary operation.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
        >>> ht.filter((ht["x"] % 2).eq(1)).run()
        Schema:
          x: INT64
          y: INT64
        Data:
        1|5
        3|3
        5|1
        >>> ht.filter(ht["x"] == ht["y"]).run()
        Schema:
          x: INT64
          y: INT64
        Data:
        3|3
        """
        pass

    def ne(self, value):
        """
        Create UNEQUAL comparison expression.

        Parameters
        ----------
        value : int, float, str or QueryExpr
            Right-hand operand for the binary operation.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
        >>> ht.filter((ht["x"] % 2).ne(1)).run()
        Schema:
          x: INT64
          y: INT64
        Data:
        2|4
        4|2
        >>> ht.filter(ht["x"] != ht["y"]).run()
        Schema:
          x: INT64
          y: INT64
        Data:
        1|5
        2|4
        4|2
        5|1
        """
        pass

    def lt(self, value):
        """
        Create LESS comparison expression.

        Parameters
        ----------
        value : int, float, str or QueryExpr
            Right-hand operand for the binary operation.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
        >>> ht.filter(ht["x"].lt(3)).run()
        Schema:
          x: INT64
          y: INT64
        Data:
        1|5
        2|4
        >>> ht.filter(ht["x"] < ht["y"]).run()
        Schema:
          x: INT64
          y: INT64
        Data:
        1|5
        2|4
        """
        pass

    def le(self, value):
        """
        Create LESS OR EQUAL comparison expression.

        Parameters
        ----------
        value : int, float, str or QueryExpr
            Right-hand operand for the binary operation.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
        >>> ht.filter(ht["x"].le(3)).run()
        Schema:
          x: INT64
          y: INT64
        Data:
        1|5
        2|4
        3|3
        >>> ht.filter(ht["x"] <= ht["y"]).run()
        Schema:
          x: INT64
          y: INT64
        Data:
        1|5
        2|4
        3|3
        """
        pass

    def gt(self, value):
        """
        Create GREATER comparison expression.

        Parameters
        ----------
        value : int, float, str or QueryExpr
            Right-hand operand for the binary operation.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
        >>> ht.filter(ht["x"].gt(4)).run()
        Schema:
          x: INT64
          y: INT64
        Data:
        5|1
        >>> ht.filter(ht["x"] > ht["y"]).run()
        Schema:
          x: INT64
          y: INT64
        Data:
        4|2
        5|1
        """
        pass

    def ge(self, value):
        """
        Create GREATER OR EQUAL comparison expression.

        Parameters
        ----------
        value : int, float, str or QueryExpr
            Right-hand operand for the binary operation.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
        >>> ht.filter(ht["x"].ge(4)).run()
        Schema:
          x: INT64
          y: INT64
        Data:
        4|2
        5|1
        >>> ht.filter(ht["x"] >= ht["y"]).run()
        Schema:
          x: INT64
          y: INT64
        Data:
        3|3
        4|2
        5|1
        """
        pass

    def at(self, index):
        """
        Create subscript expression to extract array element by index.

        The indexing is 1-based. Out-of-bound access results in NULL.

        Parameters
        ----------
        index : int or QueryExpr
            Index of element to extract.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.create_table("test1", [("a", "array(int)"), ("b", "int")])
        >>> hdk.import_pydict({"a": [[1, 2], [2, 3, 4]], "b": [2, 3]}, ht)
        >>> ht.proj(ht["a"].at(1), ht["a"][ht["b"]], ht["a"].at(0)).run()
        Schema:
          expr_1: INT64
          expr_2: INT64
          expr_3: INT64
        Data:
        1|2|null
        2|4|null
        """
        pass

    __neg__ = uminus
    __add__ = add
    __sub__ = sub
    __mul__ = mul
    __floordiv__ = floordiv
    __truediv__ = truediv
    __mod__ = mod

    __eq__ = eq
    __ne__ = ne
    __lt__ = lt
    __le__ = le
    __gt__ = gt
    __ge__ = ge

    __getitem__ = at


class QueryNodeAPI:
    def proj(self, *args, exprs=None, **kwargs):
        """
        Create a projection node with the current node as its input.

        Parameters
        ----------
        *args : list
            Each element is a column reference (though its index or name) or a QueryExpr.
        exprs : dict, default: None
            Keys determine resulting column names and values are interpreted similar
            to *args.
        **kwargs : dict
            Used similar to exprs.

        Returns
        -------
        QueryNode
            Created projection node.

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"id": [1, 2, 3], "x": [10, 20, 30], "y": [0, -10, 10]})
        >>> ht.proj("x", -1).run()
        Schema:
          x: INT64
          y: INT64
        Data:
        10|0
        20|-10
        30|10
        >>> ht.proj(sum=ht["x"] + ht["y"]).run()
        Schema:
          sum: INT64
        Data:
        10
        10
        40
        >>> ht.proj(exprs={"neg_x" : -ht["x"]}).run()
        Schema:
          neg_x: INT64
        Data:
        -10
        -20
        -30
        """
        pass

    def agg(self, group_keys, *args, aggs=None, **kwargs):
        """
        Create an aggregation node with the current node as its input.

        Parameters
        ----------
        group_keys : int, str, QueryExpr or iterable
            Group key used fro aggregation. Integer and string values can be used
            to reference input columns by its index or name. QueryExpr expressions
            can be used to simply reference input columns or build more complex
            group keys.
        *args : list
            Each element is either a string with aggregte name or QueryExpr holding
            an aggregation expression. Supported aggregtates: count, count distinct,
            max, min, sum, svg, approx count dist, approx quantile, sample, single
            value. When aggregate assumes a column or a number parameter, it's
            specified in parentheses and columns are always referenced by their names.
        aggs : dict, default: None
            Keys determine resulting colum nanmes and values are interpreted similar
            to *args.
        **kwargs : dict
            Used similar to aggs.

        Returns
        -------
        QueryNode
            Created aggregation node.

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict(
        ...     {"id1": [1, 2, 1], "id2": [1, 1, 2], "x": [10, 20, 30], "y": [0, -10, 10]}
        ... )
        >>> ht.agg([0, 1], "count", "sum(x)", "min(y)").run()
        Schema:
          id1: INT64
          id2: INT64
          count: INT32[NN]
          x_sum: INT64
          y_min: INT64
        Data:
        1|1|1|10|0
        2|1|1|20|-10
        1|2|1|30|10
        >>> ht.agg(
        ...     ["id1", "id2"],
        ...     aggs={"cnt": "count", "xs": "sum(x)", "ym": ht["y"].min()},
        ... ).run()
        Schema:
          id1: INT64
          id2: INT64
          cnt: INT32[NN]
          xs: INT64
          ym: INT64
        Data:
        1|1|1|10|0
        2|1|1|20|-10
        1|2|1|30|10
        >>> ht.agg(["id1", "id2"], cnt="count", x_sum=ht["x"].sum(), y_min=ht["y"].min()).run()
        Schema:
          id1: INT64
          id2: INT64
          cnt: INT32[NN]
          x_sum: INT64
          y_min: INT64
        Data:
        1|1|1|10|0
        2|1|1|20|-10
        1|2|1|30|10
        """
        pass

    def sort(self, *args, fields=None, limit=0, offset=0, **kwargs):
        """
        Create a sort node with the current node as its input.

        Parameters
        ----------
        *args : list
            Each element is a reference to input column (through index, name or QueryExpr)
            or a tuple. In addition to column reference, tuples hold a sort order ("asc" or
            "desc") and optionally a NULLs position ("first" or "last"). By default values
            are sorted in ascending order and NULLs have last position.
        fields : dict, default: None
            Column names mapped to a sort order or a tuple holding a sort order and a NULLs
            position.
        limit : int, default: 0
            Limit number of output rows. If 0 then unlimited.
        offset : int, default: 0
            Number of rows to skip in the result.
        **kwargs : dict
            Used similar to fields.

        Returns
        -------
        QueryNode
            Created sort node.

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict(
        >>>     {"x": [1, 2, 1, 2, 1], "y": [1, 1, 2, None, 3], "z": [10, 20, 30, 40, 50]}
        >>> )
        >>> ht.sort("x", ("y", "asc", "first")).run()
        Schema:
          x: INT64
          y: INT64
          z: INT64
        Data:
        1|1|10
        1|2|30
        1|3|50
        2|null|40
        2|1|20
        >>> ht.sort(fields={"x" : "desc", "y" : ("asc", "first")}).run()
        Schema:
          x: INT64
          y: INT64
          z: INT64
        Data:
        2|null|40
        2|1|20
        1|1|10
        1|2|30
        1|3|50
        >>> ht.sort(x="desc", y=("desc", "last")).run()
        Schema:
          x: INT64
          y: INT64
          z: INT64
        Data:
        2|1|20
        2|null|40
        1|3|50
        1|2|30
        1|1|10
        """
        pass

    def join(self, rhs_node, lhs_cols=None, rhs_cols=None, cond=None, how="inner"):
        """
        Create a join node with the current node as its left input.

        By default, column with equal names are searched in input nodes and appropriate
        equi-join is created. User can specify column names to use for equi-join or use
        an arbitrary condition. When equi-join is generate through implicit or explicit
        column lists, only left input key columns go to the result. When join condition
        expression is directly provided, then all input columns go to the result.

        Parameters
        ----------
        rhs_node : QueryNode
            A right input node for the join.
        lhs_cols : str or list of str, default: None
            A column name or a list of column names in the left input nodes to be used for
            equi-join.
        rhs_cols : str or list of str, default: None
            A column name or a list of column names in the right input nodes to be used for
            equi-join. If not specified then lhs_cols is used instead.
        cond : QueryExpr
            Join condition.
        how : str, default: "inner"
            Join type. Supported values are "inner", "left", "semi", "anti".

        Returns
        -------
        QueryNode
            Created join node.

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht1 = hdk.import_pydict(
        ...             {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "x": [1.1, 2.2, 3.3, 4.4, 5.5]}
        ...         )
        >>> ht1 = hdk.import_pydict(
        ...     {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "x": [1.1, 2.2, 3.3, 4.4, 5.5]}
        ... )
        >>> ht2 = hdk.import_pydict(
        ...     {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5], "y": [5.5, 4.4, 3.3, 2.2, 1.1]}
        ... )
        >>> ht1.join(ht2).run()
        Schema:
          a: INT64
          b: INT64
          x: FP64
          y: FP64
        Data:
        3|3|3.3|3.3
        >>> ht1.join(ht2, "a").run()
        Schema:
          a: INT64
          b: INT64
          x: FP64
          b_1: INT64
          y: FP64
        Data:
        1|5|1.1|1|5.5
        2|4|2.2|2|4.4
        3|3|3.3|3|3.3
        4|2|4.4|4|2.2
        5|1|5.5|5|1.1
        >>> ht1.join(ht2, ["a", "b"], ["b", "a"]).run()
        Schema:
          a: INT64
          b: INT64
          x: FP64
          y: FP64
        Data:
        3|3|3.3|3.3
        >>> ht1.join(ht2, cond=ht1["a"] == ht2["b"] + 3).run()
        Schema:
          a: INT64
          b: INT64
          x: FP64
          a_1: INT64
          b_1: INT64
          y: FP64
        Data:
        4|2|4.4|1|1|5.5
        5|1|5.5|2|2|4.4
        """
        pass

    def filter(self, *args):
        """
        Create a filter node with the current node as its input.

        *args : list of QueryExpr
            Filter conditions.

        Returns
        -------
        QueryNode
            Created filter node.

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"a": [1, 2, 3, 4 ,5], "b": [5, 4, 3, 2, 1]})
        >>> ht.filter((ht["a"] > 1).logical_and(ht["b"] > 2)).run()
        Schema:
          a: INT64
          b: INT64
        Data:
        2|4
        3|3
        >>> ht.filter(ht["a"] < 4, ht["b"] < 5).run()
        Schema:
          a: INT64
          b: INT64
        Data:
        2|4
        3|3
        """
        pass

    def ref(self, col):
        """
        Create a column reference expression for the current node.

        col : int or str
            Column index (negative indexing is supported) or name.

        Returns
        -------
        QueryExpr
            Created column reference expresion.

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"id": [1, 2, 3], "x": [10, 20, 30], "y": [0, -10, 10]})
        >>> ht.proj(ht.ref(0), ht.ref("x"), ht.ref(-1)).run()
        Schema:
          id: INT64
          x: INT64
          y: INT64
        Data:
        1|10|0
        2|20|-10
        3|30|10
        >>> ht.proj(ht[0], ht["x"], ht[-1]).run()
        Schema:
          id: INT64
          x: INT64
          y: INT64
        Data:
        1|10|0
        2|20|-10
        3|30|10
        """
        pass

    __getitem__ = ref

    @property
    def size(self):
        """
        Get a number of columns in the node.

        Returns
        -------
        int

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"id": [1, 2, 3], "x": [10, 20, 30], "y": [0, -10, 10]})
        >>> ht.size
        3
        >>> ht.proj(0, 1).size
        2
        """
        pass

    def column_info(self, col):
        """
        Get column info.

        Parameters
        ----------
        col : int, str or QueryExpr
            Column index, name or reference.

        Returns
        -------
        ColumnInfo

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"id": [1, 2, 3], "x": [10, 20, 30], "y": [0, -10, 10]})
        >>> ht.column_info("id")
        id(db_id=16777217, table_id=1, column_id=1 type=INT64)
        >>> ht.column_info(-1)
        y(db_id=16777217, table_id=1, column_id=3 type=INT64)
        """
        pass

    @property
    def table_name(self):
        """
        Return a referenced table name.

        For scan nodes a name of a referenced physical table is returned.
        For other nodes None is returned.

        Returns
        -------
        str or None

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"a": [1, 2, 3], "b": [3, 2, 1]})
        >>> ht.table_name
        'tabe_984f9e2346844ab4ae48aba14e1af8d4'
        >>> ht = hdk.import_pydict({"a": [1, 2, 3], "b": [3, 2, 1]}, "table1")
        >>> ht.table_name
        'table1'
        """
        pass

    @property
    def schema(self):
        """
        Return a scheme of a table represented by this node.

        Scheme is a dictionary mapping column names to ColumnInfo objects.

        Returns
        -------
        dict

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"a": [1, 2, 3], "b": [3, 2, 1]})
        >>> ht.schema
        {'a': a(db_id=16777217, table_id=4, column_id=1 type=INT64), 'b': b(db_id=16777217, table_id=4, column_id=2 type=INT64)}
        """
        pass

    def run(self):
        """
        Run query with the current node as a query root node.

        Returns
        -------
        ExecutionResult
            The result of query execution.

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"a": [1, 2, 3], "b": [3, 2, 1]})
        >>> res = ht.proj(sum=ht["a"] + ht["b"]).run()
        >>> res
        Schema:
        sum: INT64
        Data:
        4
        4
        4
        """
        pass


class QueryOptions:
    def __init__(self, config):
        self._config = config
        self._opts = {}

    @property
    def enable_lazy_fetch(self):
        return self._opts.get("enable_lazy_fetch", self._config.rs.enable_lazy_fetch)

    @enable_lazy_fetch.setter
    def enable_lazy_fetch(self, value):
        if type(value) != type(True):
            raise TypeError(
                f"Expected bool value for 'enable_lazy_fetch' option. Got: {type(value)}."
            )
        self._opts["enable_lazy_fetch"] = value

    @property
    def enable_dynamic_watchdog(self):
        return self._opts.get(
            "enable_dynamic_watchdog", self._config.exec.watchdog.enable_dynamic
        )

    @enable_dynamic_watchdog.setter
    def enable_dynamic_watchdog(self, value):
        if type(value) != type(True):
            raise TypeError(
                f"Expected bool value for 'enable_dynamic_watchdog' option. Got: {type(value)}."
            )
        self._opts["enable_dynamic_watchdog"] = value

    @property
    def enable_columnar_output(self):
        return self._opts.get(
            "enable_columnar_output", self._config.rs.enable_columnar_output
        )

    @enable_columnar_output.setter
    def enable_columnar_output(self, value):
        if type(value) != type(True):
            raise TypeError(
                f"Expected bool value for 'enable_columnar_output' option. Got: {type(value)}."
            )
        self._opts["enable_columnar_output"] = value

    @property
    def enable_watchdog(self):
        return self._opts.get("enable_watchdog", self._config.exec.watchdog.enable)

    @enable_watchdog.setter
    def enable_watchdog(self, value):
        if type(value) != type(True):
            raise TypeError(
                f"Expected bool value for 'enable_watchdog' option. Got: {type(value)}."
            )
        self._opts["enable_watchdog"] = value

    @property
    def enable_dynamic_watchdog(self):
        return self._opts.get(
            "enable_dynamic_watchdog", self._config.exec.watchdog.enable_dynamic
        )

    @enable_dynamic_watchdog.setter
    def enable_dynamic_watchdog(self, value):
        if type(value) != type(True):
            raise TypeError(
                f"Expected bool value for 'enable_dynamic_watchdog' option. Got: {type(value)}."
            )
        self._opts["enable_dynamic_watchdog"] = value

    @property
    def just_explain(self):
        return self._opts.get("just_explain", False)

    @just_explain.setter
    def just_explain(self, value):
        if type(value) != type(True):
            raise TypeError(
                f"Expected bool value for 'just_explain' option. Got: {type(value)}."
            )
        self._opts["just_explain"] = value

    @property
    def device_type(self):
        return self._opts.get("device_type", "auto")

    @device_type.setter
    def device_type(self, value):
        if value.upper() not in ("CPU", "GPU", "AUTO"):
            raise ValueError(
                "Expected 'CPU', 'GPU' or 'auto' device type. Got: {value}."
            )
        self._opts["device_type"] = value


class HDK:
    def __init__(self, **kwargs):
        self._config = buildConfig(**kwargs)
        self._storage = ArrowStorage(1)
        self._data_mgr = DataMgr(self._config)
        self._data_mgr.registerDataProvider(self._storage)
        self._calcite = Calcite(self._storage, self._config)
        self._executor = Executor(self._data_mgr, self._config)
        self._builder = QueryBuilder(self._storage, self._config, self)

    def create_table(self, table_name, scheme, fragment_size=None):
        """
        Create an empty table in HDK in-memory storage. Data can be appended to
        existing tables using data import methods.

        Parameters
        ----------
        table_name : str
            Name of the new table. Shouldn't match name of any previously created
            table.
        scheme : list of tuples or dict
            A list of tuples holding column names and types or a dictionary mapping
            column names to their types.
        fragment_size : int, default: None
            Number of rows in each table fragment. Total fragments count in a
            table may affect table processing parallelism level and performance.
            If not set, then fragment size is chosen automatically.

        Returns
        -------
        QueryExpr
            Scan expression referencing created table.

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht1 = hdk.create_table("test1", [("id", "int"), ("val1", "int64"), ("val2", "text")])
        >>> ht1
        hdk::ir::Scan#1(test1, ["id", "val1", "val2", "rowid"])
        >>> ht2 = hdk.create_table("test2", {"id": "int", "val1": "int64", "val2": "text"})
        >>> ht2
        hdk::ir::Scan#2(test2, ["id", "val1", "val2", "rowid"])
        """
        opts = TableOptions()
        if fragment_size is not None:
            opts.fragment_size = fragment_size
        self._storage.createTable(table_name, scheme, opts)
        return self.scan(table_name)

    def drop_table(self, table):
        """
        Drop existing table from HDK in-memory storage.

        Parameters
        ----------
        table : str or QueryNode
            Name of the table to drop or a scan node referencing the table to drop.

        Returns
        -------

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.create_table("test1", [("id", "int"), ("val1", "int64"), ("val2", "text")])
        >>> hdk.scan("test1")
        hdk::ir::Scan#2(test1, ["id", "val1", "val2", "rowid"])
        >>> hdk.drop_table(ht)
        >>> hdk.scan("test1")
        RuntimeError: Unknown table: test1
        >>> 
        """
        if isinstance(table, QueryNode) and table.is_scan:
            table = table.table_name

        if isinstance(table, str):
            self._storage.dropTable(table)
        else:
            raise TypeError(
                f"Only str and QueryNode scans are allowed for 'table' arg. Provided: {table}"
            )

    @not_implemented
    def import_csv(
        self,
        file_name,
        table_name=None,
        scheme=None,
        delim=",",
        header=True,
        skip_rows=0,
        fragment_size=None,
        block_size=None,
    ):
        """
        Import CSV file(s) into HDK in-memory storage.

        Parameters
        ----------
        file_name : str or list of str
            CSV file name or a list of file names. Unix style pathname patterns
            are allowed to reference multiple files.
        table_name : str, default: None
            Destination table name. If not specified, then unique table name is
            generated and used. If table with specified name already exists,
            then imported data is appended to the existing table.
        scheme : list of str, list of tuples, or dict, default: None
            List of strings simply lists column names and can be used for CSV files
            with no headers. In this case, types are auto-detected. Tuple of two
            values can be used to cpecify both column name and type. Alternatively,
            a dictionary can be provided to map column names to their types. When
            header is not used, then scheme should list all columns. When header is
            used, scheme can be partial to enforce required data types for specific
            columns. If the destination table already exists then mismatch between
            provided scheme and table's scheme would cause an error.
        delim : str, default: ","
            Delimiter symbol.
        header : bool, default: True
            Use the first CSV file as a header to define column names.
        skip_rows : int, default: 0
            Number of rows to skip.
        fragment_size : int, default: None
            Number of rows in each table fragment. Total fragments count in a
            table may affect table processing parallelism level and performance.
            If not set, then fragment size is chosen automatically.
        block_size : int, default: None
            The size of data chunks used by the parser for multi-threaded parsing.
            Auto-detected by default.

        Returns
        -------
        QueryExpr
            Scan expression referencing created table.
        """
        pass

    @not_implemented
    def import_parquet(self, file_name, table_name=None, fragment_size=None):
        """
        Import Parquet file(s) into HDK in-memory storage.

        Parameters
        ----------
        file_name : str or list of str
            Parquet file name or a list of file names. Unix style pathname patterns
            are allowed to reference multiple files.
        table_name : str, default: None
            Destination table name. If not specified, then unique table name is
            generated and used. If table with specified name already exists,
            then imported data is appended to the existing table.
        fragment_size : int, default: None
            Number of rows in each table fragment. Total fragments count in a
            table may affect table processing parallelism level and performance.
            If not set, then fragment size is chosen automatically.

        Returns
        -------
        QueryExpr
            Scan expression referencing created table.
        """
        pass

    def import_arrow(self, at, table_name=None, fragment_size=None):
        """
        Import Arrow table into HDK in-memory storage.

        Parameters
        ----------
        at : pyarrow.Table
            Arrow table to import.
        table_name : str, default: None
            Destination table name. If not specified, then unique table name is
            generated and used. If table with specified name already exists,
            then imported data is appended to the existing table.
        fragment_size : int, default: None
            Number of rows in each table fragment. Total fragments count in a
            table may affect table processing parallelism level and performance.
            If not set, then fragment size is chosen automatically.

        Returns
        -------
        QueryExpr
            Scan expression referencing created table.
        """
        append = False
        real_name = table_name
        if isinstance(table_name, QueryNode):
            if not table_name.is_scan:
                raise TypeError("Non-scan QueryNode is not allowed as a table name.")
            real_name = table_name.table_name
            if self._storage.tableInfo(real_name) is None:
                raise RuntimeError(
                    "Table referred by scan QueryNode does not exist anymore: {real_name}."
                )
            append = True
        elif isinstance(table_name, str):
            append = self._storage.tableInfo(table_name) is not None
        elif table_name is None:
            real_name = "tabe_" + uuid.uuid4().hex
        else:
            raise TypeError(
                f"Expected str or QueryNode for 'table_name' arg. Got: {type(table_name)}."
            )

        if append:
            self._storage.appendArrowTable(at, real_name)
        else:
            opts = TableOptions()
            if fragment_size is not None:
                opts.fragment_size = fragment_size
            self._storage.importArrowTable(at, real_name, opts)

        return table_name if isinstance(table_name, QueryNode) else self.scan(real_name)

    def import_pydict(self, values, table_name=None, fragment_size=None):
        """
        Import Python dictionary into HDK in-memory storage.

        Parameters
        ----------
        values : dict
            Python dictionary to import.
        table_name : str, default: None
            Destination table name. If not specified, then unique table name is
            generated and used. If table with specified name already exists,
            then imported data is appended to the existing table.
        fragment_size : int, default: None
            Number of rows in each table fragment. Total fragments count in a
            table may affect table processing parallelism level and performance.
            If not set, then fragment size is chosen automatically.

        Returns
        -------
        QueryExpr
            Scan expression referencing created table.

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"a": [1, 2, 3], "b": [10.1, 20.2, 30.3]}, "t1")
        >>> ht.table_name
        't1'
        >>> ht.schema
        {'a': a(db_id=16777217, table_id=9, column_id=1 type=INT64), 'b': b(db_id=16777217, table_id=9, column_id=2 type=FP64)}
        """
        return self.import_arrow(
            pyarrow.Table.from_pydict(values),
            table_name=table_name,
            fragment_size=fragment_size,
        )

    def query_opts(self):
        return QueryOptions(self._config)

    def sql(self, sql_query, query_opts=None, **kwargs):
        """
        Execute SQL query.

        Parameters
        ----------
        sql_query : str
            SQL query to execute.
        query_opts : QueryOptions or dict, default: None
            Query execution options.
        **kwargs : dict
            Table aliases for the query. Keys are alises for tables referenced
            by values. Each value should be either a string or a scan expression.

        Returns
        -------
        ExecutionResult
            The result of query execution.

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>>
        >>> hdk.import_csv("test.csv", "test")
        >>> res = hdk.sql("SELCT type, count(*) FROM test GROUP BY type;")
        >>>
        >>> test = hdk.import_csv("test.csv")
        >>> res = hdk.sql("SELCT type, count(*) FROM test GROUP BY type;", test=test)
        """
        if query_opts is None:
            query_opts = {}
        elif isinstance(query_opts, QueryOptions):
            query_opts = query_opts._opts
        elif not isinstance(query_opts, dict):
            raise TypeError(
                f"Expected dict or QueryOptions for 'query_opts' arg. Got: {type(query_opts)}."
            )

        parts = []
        for name, orig_table in kwargs.items():
            if isinstance(orig_table, QueryNode) and orig_table.is_scan:
                orig_table = orig_table.table_name
            if not isinstance(orig_table, str):
                raise TypeError(
                    f"Expected str or table scan QueryNode for a table name alias. Got: {type(orig_table)}."
                )

            if len(parts) == 0:
                parts.append("WITH\n  ")
            else:
                parts.append(", ")
            parts.append(f"{name} AS (SELECT * FROM {orig_table})\n")

        sql_query = "".join(parts) + sql_query
        ra = self._calcite.process(sql_query)
        ra_executor = RelAlgExecutor(self._executor, self._storage, self._data_mgr, ra)
        return ra_executor.execute(**query_opts)

    def scan(self, table_name):
        """
        Create a scan query node referencing specified table.

        Parameters
        ----------
        table_name : str
            A name of a referenced table.

        Returns
        -------
        QueryNode

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"a": [1, 2, 3], "b": [10.1, 20.2, 30.3]}, "t1")
        >>> hdk.scan("t1")
        hdk::ir::Scan#2(t1, ["a", "b", "rowid"])
        """
        return self._builder.scan(table_name)

    def type(self, type_str):
        """
        Parse string type representation into TypeInfo object.

        Parameters
        ----------
        type_str : str
            String type representation. Type strings are case insensitive. It usually
            consists of type name, optional length in bits, optional parameter,
            optional unit parameter, and optional nullability suffix.

            By default types are nullable, "[nn]" suffix is used for non-nullable types.

            Datetime types support time unit specification. Supported suffixes:
            [m] = month, [d] = day, [s] = second, [ms] = millisecond, [us] = microsecond,
            [ns] = nanosecond.

            Supported types:

            Null type: "nullt"

            Integer type examples: "int8", "int16", "int32[nn]", "int64", "int" (equal
            to "int64")

            Floating point type examples: "fp32", "fp64[nn]", "fp" (equal to "fp64")

            Decimal type examples: "dec(10, 2)" (equal to "dec64(10, 2)"), "dec64(12, 1)",
            "DECIMAL64(10, 2)[NN]"

            Boolean type examples: "bool", "bool[nn]"

            Varchar type examples: "varcahr(10)[nn]", "varchar(0)"

            Text type examples: "text", "TEXT[NN]"

            Date type examples: "date32" (equal to "date32[s]"), "date64[s][nn]",
            "date" (equal to "date64[s]")

            Time type examples: "time16[s]", "time32[ms]", "time" (equal to "time64[us]")

            Timestamp type examples: "timestamp64[s]", "timestamp" (equal to "timestamp64[ms]")

            Interval type examples: "interval32[ms]", "interval" (equal to "interval64[ms])

            Dictionary type exampless: "dict32(text[nn])[10]" (10 - dictionary ID), "dict"
            (equal to "dict32(text)[0]), "dict16[5]" (equal to "dict16(text)[5]")

            Fixed length array type examples: array(int)(2), "array(fp64[nn])(8)[nn]",
            "array(dict(text[nn])[10])(4)[nn]"

            Variable length array type examples: "array(int)", "array(fp64[nn])[nn]"

        Returns
        -------
        TypeInfo

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> hdk.type("int")
        INT64
        >>> hdk.type("int32")
        INT32
        >>> hdk.type("fp")
        FP64
        >>> hdk.type("dec(10,2)")
        DEC64(10,2)
        >>> hdk.type("text[nn]")
        TEXT[NN]
        >>> hdk.type("array(int)")
        ARRAY32(INT64)
        """
        return self._builder.typeFromString(type_str)

    def const(self, value, cst_type=None, scale_decimal=True):
        """
        Create an expression representing a constant value.

        Parameters
        ----------
        value : None, int, float, bool, str or list
            Constant value.
        cst_type : str or TypeInfo, default: None
            Constant type. If not specified, then inferenced from value.
        scale_decimal : bool, default: True
            If true, then integer value passed for decimal literal will
            be scaled according to the type scale. Otherwise, it will be
            used as is for decimal value representation. E.g. value 1557
            for decimal(10, 2) will result in literal 1557.00 with enabled
            scaling and 15.57 with disabled scaling.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> # int64 constant
        >>> hdk.cst(10)
        (Const 10)
        >>> # fp64 constant
        >>> hdk.cst(10.1)
        (Const 10.100000)
        >>> hdk.cst(10, "fp64")
        (Const 10.000000)
        >>> # decimal constant
        >>> hdk.cst(1234, "dec(10,2)")
        (Const 1234.00)
        >>> hdk.cst(1234, "dec(10,2)", scale_decimal=False)
        (Const 12.34)
        >>> # bool constant
        >>> hdk.cst(True)
        (Const t)
        >>> hdk.cst(1, "bool")
        (Const t)
        >>> # string constant
        >>> hdk.cst("value")
        (Const value)
        >>> # date constant
        >>> hdk.cst("2001-02-03", "date")
        (Const 2001-02-03)
        >>> # timestamp constant
        >>> hdk.cst("2001-02-03 15:00:00", "timestamp")
        (Const 2001-02-03 15:00:00.000000)
        """
        return self._builder.cst(value, cst_type, scale_decimal)

    cst = const

    def date(self, value):
        """
        Create a date literal from string.

        Parameters
        ----------
        value : str
            Literal value.

        Returns
        -------
        QueryExpr
        >>> hdk = pyhdk.init()
        >>> hdk.date("2001-02-03")
        (Const 2001-02-03)
        """
        return self._builder.date(value)

    def time(self, value):
        """
        Create a time literal from string.

        Parameters
        ----------
        value : str
            Literal value.

        Returns
        -------
        QueryExpr
        >>> hdk = pyhdk.init()
        >>> hdk.time("15:00:00")
        (Const 15:00:00)
        """
        return self._builder.time(value)

    def timestamp(self, value):
        """
        Create a timestamp literal from string.

        Parameters
        ----------
        value : str
            Literal value.

        Returns
        -------
        QueryExpr
        >>> hdk = pyhdk.init()
        >>> hdk.timestamp("2001-02-03 15:00:00")
        (Const 2001-02-03 15:00:00.000000)
        """
        return self._builder.timestamp(value)

    def count(self):
        """
        Create a count agrregation expression.

        Returns
        -------
        QueryExpr
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_pydict({"a": [1, 2, 3, 4, 5]})
        >>> ht.agg([], hdk.count()).run()
        Schema:
          count: INT32[NN]
        Data:
        5
        """
        return self._builder.count()


def init(**kwargs):
    if init._instance is None:
        init._instance = HDK(**kwargs)
    return init._instance


init._instance = None
