#
# Copyright 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from pyhdk._common import buildConfig
from pyhdk._storage import ArrowStorage, DataMgr
from pyhdk._sql import Calcite, RelAlgExecutor
from pyhdk._execute import Executor


def not_implemented(func):
    def wrapper(*args, **kwargs):
        raise NotImplementedError(f"{func.__qualname__} is not yet implemented.")

    return wrapper


class QueryExpr:
    @not_implemented
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
        >>> hdk = pyhdk.init()
        >>> scan = hdk.from_pydict({"x": [1, 4, 7], "y": [9, 3, 1]})
        >>> proj = scan.proj((scan["x"] + scan["y"]).rename("sum"))
        >>> agg = proj.agg([], "max(sum)")
        >>> # Alternative option
        >>> proj = scan.proj(sum=scan["x"] + scan["y"])
        >>> agg = proj.agg([], "max(sum)")
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, 11, 5]})
        >>> agg = scan.agg(["id"], scan["x"].avg())
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, 11, 5]})
        >>> agg = scan.agg(["id"], scan["x"].min())
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, 11, 5]})
        >>> agg = scan.agg(["id"], scan["x"].max())
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, 11, 5]})
        >>> agg = scan.agg(["id"], scan["x"].sum())
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, None, 5]})
        >>> agg = scan.agg(["id"], scan["x"].count())
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, 11, 5]})
        >>> agg = scan.agg(["id"], scan["x"].approx_quantile(0.5))
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, 11, 5]})
        >>> agg = scan.agg(["id"], sample["x"].sample())
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, 11, 5]})
        >>> agg = scan.agg(["id"], scan["x"].single_value())
        """
        pass

    @not_implemented
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
        >>> day = hdk.date("1983-12-01").extract("dayofweek")
        """
        pass

    @not_implemented
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
        >>> day = hdk.cst("1983-12-01").cast("date")
        """
        pass

    @not_implemented
    def uminus(self):
        """
        Create unary minus expression for the current expression.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> scan = hdk.from_pydict({"id": [1, 2, 1, 2, 1], "x": [4, 7, 9, 11, 5]})
        >>> node = scan.proj("id", scan["x"].uminus())
        """
        pass

    @not_implemented
    def is_null(self):
        """
        Create IS NULL expression for the current expression.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> scan = hdk.from_pydict({"id": [1, 2, 1, 2, 1], "x": [4, None, 9, None, 5]})
        >>> node = scan.proj("id", scan["x"].is_null())
        """
        pass

    @not_implemented
    def is_not_null(self):
        """
        Create IS NOT NULL expression for the current expression.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> scan = hdk.from_pydict({"id": [1, 2, 1, 2, 1], "x": [4, None, 9, None, 5]})
        >>> node = scan.proj("id", scan["x"].is_not_null())
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"arr": [[1, 2], [3, 4]]})
        >>> node = scan.proj(scan["arr"].unnest())
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"x": [1, 2, 1, 2, 1], "y": [4, 7, 9, 11, 5]})
        >>> node = scan.proj(scan["x"].add(scan["y"]))
        >>> ...
        >>> expr = pyhdk.date("1983-12-01").add(5, "weeks")
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"x": [1, 2, 1, 2, 1], "y": [4, 7, 9, 11, 5]})
        >>> node = scan.proj(scan["x"].sub(scan["y"]))
        >>> ...
        >>> expr = pyhdk.date("1983-12-01").sub(5, "weeks")
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"x": [1, 2, 1, 2, 1], "y": [4, 7, 9, 11, 5]})
        >>> node = scan.proj(scan["x"].mul(scan["y"]))
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
        >>> scan = hdk.from_pydict({"x": [11, 14, 21, 28, 15], "y": [2, 3, 7, 4, 5]})
        >>> node = scan.proj(scan["x"].truediv(scan["y"]))
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
        >>> scan = hdk.from_pydict({"x": [11, 14, 21, 28, 15], "y": [2, 3, 7, 4, 5]})
        >>> node = scan.proj(scan["x"].truediv(scan["y"]))
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"x": [11, 14, 21, 28, 15], "y": [2, 3, 7, 4, 5]})
        >>> node = scan.proj(scan["x"].div(scan["y"]))
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"x": [11, 14, 21, 28, 15]})
        >>> node = scan.proj(scan["x"].mod(5))
        """
        pass

    @not_implemented
    def logical_not(self):
        """
        Create logical NOT expression.

        Returns
        -------
        QueryExpr

        Examples
        --------
        >>> hdk = pyhdk.init()
        >>> scan = hdk.from_pydict({"x": [True, False, True, False]})
        >>> node = scan.proj(scan["x"].logical_not())
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict(
        >>>     {"x": [True, False, True, False], "y": [True, True, False, False]}
        >>> )
        >>> node = scan.proj(scan["x"].logical_and(scan["y"]))
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict(
        >>>     {"x": [True, False, True, False], "y": [True, True, False, False]}
        >>> )
        >>> node = scan.proj(scan["x"].logical_or(scan["y"]))
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"x": [11, 14, 21, 28, 15]})
        >>> node = scan.proj(scan["x"].eq(14))
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"x": [11, 14, 21, 28, 15]})
        >>> node = scan.proj(scan["x"].ne(14))
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"x": [11, 14, 21, 28, 15]})
        >>> node = scan.proj(scan["x"].lt(15))
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"x": [11, 14, 21, 28, 15]})
        >>> node = scan.proj(scan["x"].le(15))
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"x": [11, 14, 21, 28, 15]})
        >>> node = scan.proj(scan["x"].gt(15))
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"x": [11, 14, 21, 28, 15]})
        >>> node = scan.proj(scan["x"].ge(15))
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"x": [[1, 2], [3, 4]]})
        >>> node = scan.proj(scan["x"].at(1))
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


class QueryNode:
    @not_implemented
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
        >>> scan = hdk.from_pydict({"id": [1, 2, 3], "x": [10, 20, 30], "y": [0, -10, 10]})
        >>> # All following projections select columns 'x' and 'y'.
        >>> node = scan.proj(1, 2)
        >>> node = scan.proj("x", "y")
        >>> node = scan.proj({"x": scan["x"], "y": "y"})
        >>> node = scan.proj(x="x", y=scan["y"])
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict(
        >>>     {"id1": [1, 2, 1], "id2": [1, 1, 2], "x": [10, 20, 30], "y": [0, -10, 10]}
        >>> )
        >>> node = scan.agg([0, 1], "count", "sum(x)", "min(y)")
        >>> node = scan.agg(
        >>>     ["id1", "id2"],
        >>>     aggs={"cnt": "count", "x_sum": "sum(x)", "y_min": scan["y"].min()},
        >>> )
        >>> node = scan.agg(
        >>>     ["id1", "id2"], cnt="count", x_sum=scan["x"].sum(), y_min=scan["y"].min()
        >>> )
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict(
        >>>     {"x": [1, 2, 1, 2, 1], "y": [1, 1, 2, None, 3], "z": [10, 20, 30, 40, 50]}
        >>> )
        >>> node = scan.sort("x", ("y", "asc", "first"))
        >>> node = scan.sort(fields={"x" : "asc", "y" : ("asc", "first")})
        >>> node = scan.sort(x="asc", y=("asc", "first"))
        """
        pass

    @not_implemented
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
        >>> lhs = hdk.from_pydict({"id": [1, 2, 7, 9, 15], "val1": [1, 2, 4, 7, 9]})
        >>> rhs = hdk.from_pydict({"id": [1, 3, 4, 9, 10], "val2": [1, 2, 4, 7, 9]})
        >>> # Following join expressions are equal
        >>> node = lhs.join(rhs)
        >>> node = lhs.join(rhs, "id")
        >>> node = lhs.join(rhs, "id", "id")
        >>> # This join expression would perform the same equi-join but would have
        >>> # duplicated join key column "x" in the result (as 'x' and "x_1").
        >>> node = lhs.join(rhs, cond=lhs["id"].eq(rhs["id"]), how="inner")
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"id": [1, 2, 3], "x": [10, 20, 30], "y": [0, -10, 10]})
        >>> # Following filter nodes are equal
        >>> node = scan.filter((scan["x"] > 10).logicalAnd(scan["y"] < 10))
        >>> node = scan.filter(scan["x"] > 10, scan["y"] < 10)
        """
        pass

    @not_implemented
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
        >>> scan = hdk.from_pydict({"id": [1, 2, 3], "x": [10, 20, 30], "y": [0, -10, 10]})
        >>> # Different ways to get a reference to the "id" column.
        >>> ref = scan.ref(0)
        >>> ref = scan.ref(-3)
        >>> ref = scan.ref("id")
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
        """
        pass

    @not_implemented
    def run(self):
        """
        Run query with the current node as a query root node.

        Returns
        -------
        ExecutionResult
            The result of query execution.
        """
        pass


class HDK:
    def __init__(self, **kwargs):
        self._config = buildConfig(**kwargs)
        self._storage = ArrowStorage(1)
        self._data_mgr = DataMgr(self._config)
        self._data_mgr.registerDataProvider(self._storage)
        self._calcite = Calcite(self._storage, self._config)
        self._executor = Executor(self._data_mgr, self._config)
        # self._builder = QueryBuilder(self._storage, self._config)

    @not_implemented
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
        >>> hdk.create_table("test1", [("id", "int"), ("val1", "int64"), ("val2", "text")])
        >>> hdk.create_table("test2", {"id": "int", "val1": "int64", "val2": "text"})
        """
        pass

    @not_implemented
    def drop_table(self, table):
        """
        Drop existing table from HDK in-memory storage.

        Parameters
        ----------
        table : str or QueryNode
            Name of the table to drop or a scan node referencing the table to drop.

        Returns
        -------
        """
        pass

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

    @not_implemented
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
        pass

    @not_implemented
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
        """
        pass

    @not_implemented
    def sql(self, sql_query, **kwargs):
        """
        Execute SQL query.

        Parameters
        ----------
        sql_query : str
            SQL query to execute.
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
        ra = self._calcite.process(sql_query)
        ra_executor = RelAlgExecutor(self._executor, self._storage, self._data_mgr, ra)
        return ra_executor.execute()

    @not_implemented
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
        """
        pass

    @not_implemented
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
        """
        pass

    @not_implemented
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
        >>> cst = hdk.cst(10)
        >>> # fp64 constant
        >>> cst = hdk.cst(10.1)
        >>> cst = hdk.cst(10, "fp64")
        >>> # decimal constant
        >>> cst = hdk.cst(10, "dec(10,2))
        >>> cst = hdk.cst(1000, "dec(10,2), scale_decimal=False)
        >>> cst = hdk.cst(10.00, "dec(10,2))
        >>> cst = hdk.cst("10.00", "dec(10,2)")
        >>> # bool constant
        >>> cst = hdk.cst(True)
        >>> cst = hdk.cst(1, "bool")
        >>> # string constant
        >>> cst = hdk.cst("value")
        >>> # date constant
        >>> cst = hdk.cst("2001-02-03", "date")
        >>> # timestamp constant
        >>> cst = hdk.cst("2001-02-03 15:00:00", "date")
        """
        pass

    cst = const

    @not_implemented
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
        >>> cst = hdk.date("2001-02-03")
        """
        pass

    @not_implemented
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
        >>> cst = hdk.time("15:00:00")
        """
        pass

    @not_implemented
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
        >>> cst = hdk.timestamp("2001-02-03 15:00:00")
        """
        pass

    def count(self):
        """
        Create a count agrregation expression.

        Returns
        -------
        QueryExpr
        >>> hdk = pyhdk.init()
        >>> ht = hdk.import_csv("test.csv")
        >>> res = ht.proj(hdk.count())
        """
        pass


def init(**kwargs):
    if init._instance is None:
        init._instance = HDK(**kwargs)
    return init._instance


init._instance = None
