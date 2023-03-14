/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.calcite.prepare;

import com.google.common.collect.ImmutableSet;
import com.mapd.calcite.parser.MapDParserOptions;
import com.mapd.calcite.parser.MapDSchema;
import com.mapd.calcite.parser.ProjectProjectRemoveRule;

import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.config.CalciteConnectionConfigImpl;
import org.apache.calcite.config.CalciteConnectionProperty;
import org.apache.calcite.jdbc.CalciteSchema;
import org.apache.calcite.linq4j.function.Functions;
import org.apache.calcite.plan.Context;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptCostImpl;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgram;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.plan.volcano.VolcanoPlanner;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.RelShuttleImpl;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.externalize.MapDRelJsonReader;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.advise.SqlAdvisor;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.validate.SqlConformanceEnum;
import org.apache.calcite.sql.validate.SqlMoniker;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.tools.FrameworkConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

/**
 * Customised version of the PlannerImpl for MapD. Used to be a copy of
 * PlannerImpl, refactored now to use inheritance to minimize maintenance
 * efforts. Implementation of {@link org.apache.calcite.tools.Planner}.
 */
public class MapDPlanner extends PlannerImpl {
  FrameworkConfig config;
  private List<MapDParserOptions.FilterPushDownInfo> filterPushDownInfo =
          new ArrayList<>();
  private Restriction restriction = null;
  final static Logger MAPDLOGGER = LoggerFactory.getLogger(MapDPlanner.class);

  public MapDPlanner(FrameworkConfig config) {
    super(config);
    this.config = config;
  }

  public boolean isExpand() {
    return config.getSqlToRelConverterConfig().isExpand();
  }

  private static SchemaPlus rootSchema(SchemaPlus schema) {
    for (;;) {
      if (schema.getParentSchema() == null) {
        return schema;
      }
      schema = schema.getParentSchema();
    }
  }

  private CalciteCatalogReader createCatalogReader() {
    final SchemaPlus rootSchema = rootSchema(config.getDefaultSchema());
    final Context context = config.getContext();
    final CalciteConnectionConfig connectionConfig;

    if (context != null) {
      connectionConfig = context.unwrap(CalciteConnectionConfig.class);
    } else {
      Properties properties = new Properties();
      properties.setProperty(CalciteConnectionProperty.CASE_SENSITIVE.camelName(),
              String.valueOf(config.getParserConfig().caseSensitive()));
      connectionConfig = new CalciteConnectionConfigImpl(properties);
    }

    return new CalciteCatalogReader(CalciteSchema.from(rootSchema),
            CalciteSchema.from(config.getDefaultSchema()).path(null),
            getTypeFactory(),
            connectionConfig);
  }

  public void advanceToValidate() {
    try {
      String dummySql = "SELECT 1";
      super.parse(dummySql);
    } catch (SqlParseException e) {
      throw new RuntimeException(e);
    }
  }

  public void ready() {
    // need to call ready on the super class, but that method is marked private
    // circumventing via reflection for now
    try {
      Method readyMethod = getClass().getSuperclass().getDeclaredMethod("ready");
      readyMethod.setAccessible(true);
      readyMethod.invoke(this);
    } catch (InvocationTargetException e) {
      if (e.getCause() instanceof RuntimeException) {
        throw(RuntimeException) e.getCause();
      } else {
        throw new RuntimeException(e.getCause());
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public static HepPlanner getHepPlanner(HepProgram hepProgram, boolean noDag) {
    if (noDag) {
      return new HepPlanner(
              hepProgram, null, true, Functions.ignore2(), RelOptCostImpl.FACTORY);
    } else {
      return new HepPlanner(hepProgram);
    }
  }

  @Override
  public RelRoot rel(SqlNode sql) {
    RelRoot root = super.rel(sql);
    if (restriction != null) {
      root = applyInjectFilterRule(root, restriction);
    }
    root = applyQueryOptimizationRules(root);
    root = applyFilterPushdown(root);
    return root;
  }

  private RelRoot applyInjectFilterRule(RelRoot root, Restriction restriction) {
    // TODO consider doing these rules in one preplan pass

    final InjectFilterRule injectFilterRule =
            InjectFilterRule.Config.DEFAULT.toRule(restriction);

    final HepProgram program =
            HepProgram.builder().addRuleInstance(injectFilterRule).build();
    HepPlanner prePlanner = MapDPlanner.getHepPlanner(program, false);
    prePlanner.setRoot(root.rel);
    final RelNode rootRelNode = prePlanner.findBestExp();
    return root.withRel(rootRelNode);
  }

  private RelRoot applyFilterPushdown(RelRoot root) {
    if (filterPushDownInfo.isEmpty()) {
      return root;
    }
    final DynamicFilterJoinRule dynamicFilterJoinRule = new DynamicFilterJoinRule(true,
            RelFactories.LOGICAL_BUILDER,
            FilterJoinRule.TRUE_PREDICATE,
            filterPushDownInfo);
    final HepProgram program =
            HepProgram.builder().addRuleInstance(dynamicFilterJoinRule).build();
    HepPlanner prePlanner = MapDPlanner.getHepPlanner(program, false);
    prePlanner.setRoot(root.rel);
    final RelNode rootRelNode = prePlanner.findBestExp();
    filterPushDownInfo.clear();
    return root.withRel(rootRelNode);
  }

  private RelRoot applyQueryOptimizationRules(RelRoot root) {
    QueryOptimizationRules outerJoinOptRule =
            new OuterJoinOptViaNullRejectionRule(RelFactories.LOGICAL_BUILDER);

    HepProgram program = HepProgram.builder()
                                 .addRuleInstance(CoreRules.AGGREGATE_MERGE)
                                 .addRuleInstance(outerJoinOptRule)
                                 .build();
    HepPlanner prePlanner = MapDPlanner.getHepPlanner(program, true);
    prePlanner.setRoot(root.rel);
    final RelNode rootRelNode = prePlanner.findBestExp();
    return root.withRel(rootRelNode);
  }

  private RelRoot applyOptimizationsRules(RelRoot root, ImmutableSet<RelOptRule> rules) {
    HepProgramBuilder programBuilder = new HepProgramBuilder();
    for (RelOptRule rule : rules) {
      programBuilder.addRuleInstance(rule);
    }
    HepPlanner hepPlanner = MapDPlanner.getHepPlanner(programBuilder.build(), true);
    hepPlanner.setRoot(root.rel);
    return root.withRel(hepPlanner.findBestExp());
  }

  public RelRoot optimizeRaQuery(String query, MapDSchema schema) throws IOException {
    ready();
    RexBuilder builder = new RexBuilder(getTypeFactory());
    RelOptCluster cluster = RelOptCluster.create(new VolcanoPlanner(), builder);
    CalciteCatalogReader catalogReader = createCatalogReader();
    MapDRelJsonReader reader = new MapDRelJsonReader(cluster, catalogReader, schema);

    RelRoot relR = RelRoot.of(reader.read(query), SqlKind.SELECT);

    if (restriction != null) {
      relR = applyInjectFilterRule(relR, restriction);
    }

    relR = applyQueryOptimizationRules(relR);
    relR = applyFilterPushdown(relR);
    relR = applyOptimizationsRules(relR,
            ImmutableSet.of(CoreRules.JOIN_PROJECT_BOTH_TRANSPOSE_INCLUDE_OUTER,
                    CoreRules.FILTER_REDUCE_EXPRESSIONS,
                    ProjectProjectRemoveRule.INSTANCE,
                    CoreRules.PROJECT_FILTER_TRANSPOSE));
    relR = applyOptimizationsRules(relR, ImmutableSet.of(CoreRules.PROJECT_MERGE));
    relR = applyOptimizationsRules(relR,
            ImmutableSet.of(
                    CoreRules.FILTER_PROJECT_TRANSPOSE, CoreRules.PROJECT_REMOVE));
    return RelRoot.of(relR.project(), relR.kind);
  }

  public void setFilterPushDownInfo(
          final List<MapDParserOptions.FilterPushDownInfo> filterPushDownInfo) {
    this.filterPushDownInfo = filterPushDownInfo;
  }

  public void setRestriction(Restriction restriction) {
    this.restriction = restriction;
  }

  private RelRoot acceptShuttleOnTree(RexShuttle shuttle, RelRoot root) {
    RelNode node = root.rel.accept(new RelShuttleImpl() {
      @Override
      protected RelNode visitChild(RelNode parent, int i, RelNode child) {
        RelNode node = super.visitChild(parent, i, child);
        return node.accept(shuttle);
      }
    });

    return new RelRoot(node,
            root.validatedRowType,
            root.kind,
            root.fields,
            root.collation,
            Collections.emptyList());
  }

  public RelRoot expandSearch(RelRoot root) {
    final RexShuttle callShuttle = new RexShuttle() {
      RexBuilder builder = new RexBuilder(getTypeFactory());

      public RexNode visitCall(RexCall call) {
        call = (RexCall) super.visitCall(call);
        if (call.getKind() == SqlKind.SEARCH) {
          return RexUtil.expandSearch(builder, null, call);
        }
        return call;
      }
    };

    return acceptShuttleOnTree(callShuttle, root);
  }

  public RelRoot replaceIsTrue(RelRoot root) {
    final RexShuttle callShuttle = new RexShuttle() {
      RexBuilder builder = new RexBuilder(getTypeFactory());

      public RexNode visitCall(RexCall call) {
        call = (RexCall) super.visitCall(call);
        if (call.getKind() == SqlKind.IS_TRUE) {
          return builder.makeCall(SqlStdOperatorTable.AND,
                  builder.makeCall(
                          SqlStdOperatorTable.IS_NOT_NULL, call.getOperands().get(0)),
                  call.getOperands().get(0));
        } else if (call.getKind() == SqlKind.IS_NOT_TRUE) {
          return builder.makeCall(SqlStdOperatorTable.OR,
                  builder.makeCall(
                          SqlStdOperatorTable.IS_NULL, call.getOperands().get(0)),
                  builder.makeCall(SqlStdOperatorTable.NOT, call.getOperands().get(0)));
        } else if (call.getKind() == SqlKind.IS_FALSE) {
          return builder.makeCall(SqlStdOperatorTable.AND,
                  builder.makeCall(
                          SqlStdOperatorTable.IS_NOT_NULL, call.getOperands().get(0)),
                  builder.makeCall(SqlStdOperatorTable.NOT, call.getOperands().get(0)));
        } else if (call.getKind() == SqlKind.IS_NOT_FALSE) {
          return builder.makeCall(SqlStdOperatorTable.OR,
                  builder.makeCall(
                          SqlStdOperatorTable.IS_NULL, call.getOperands().get(0)),
                  call.getOperands().get(0));
        }

        return call;
      }
    };

    return acceptShuttleOnTree(callShuttle, root);
  }
}

// End MapDPlanner.java
