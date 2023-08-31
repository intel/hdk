package org.apache.calcite.rel.externalize;

import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlAggFunction;

import java.util.List;

public class QuantileAggregateCall extends AggregateCall {
    private final String interpolation;

    public QuantileAggregateCall(SqlAggFunction aggFunction, boolean distinct, List<Integer> argList,
                                 RelDataType type, String interpolation) {
        super(aggFunction, distinct, argList, type, null);
        this.interpolation = interpolation;
    }

    public String getInterpolation() {
        return interpolation;
    }
}
