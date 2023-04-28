/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "IR/ExprRewriter.h"

#include <boost/functional/hash.hpp>

#include <unordered_map>

namespace hdk::ir {

class InputRewriter final : public ExprRewriter {
 public:
  InputRewriter() = default;

  InputRewriter(const Node* old_input, const Node* new_input) {
    addNodeMapping(old_input, new_input);
  }

  InputRewriter(const Node* old_input,
                const Node* new_input,
                const std::unordered_map<unsigned, unsigned>& old_to_new_index_map) {
    addNodeMapping(old_input, new_input, old_to_new_index_map);
  }

  void addNodeMapping(const Node* old_input, const Node* new_input) {
    node_map_[old_input] = new_input;
  }

  void addNodeMapping(
      const Node* old_input,
      const Node* new_input,
      const std::unordered_map<unsigned, unsigned>& old_to_new_index_map) {
    node_map_[old_input] = new_input;
    for (auto& pr : old_to_new_index_map) {
      index_map_[std::make_pair(old_input, pr.first)] = pr.second;
    }
  }

  ExprPtr visitColumnRef(const ColumnRef* col_ref) override {
    auto node_it = node_map_.find(col_ref->node());
    if (node_it != node_map_.end()) {
      unsigned index = col_ref->index();
      auto idx_it = index_map_.find(std::make_pair(col_ref->node(), index));
      if (idx_it != index_map_.end()) {
        index = idx_it->second;
      }
      return makeExpr<ColumnRef>(col_ref->type(), node_it->second, index);
    }
    return ExprRewriter::visitColumnRef(col_ref);
  }

 protected:
  std::unordered_map<const Node*, const Node*> node_map_;
  std::unordered_map<std::pair<const Node*, unsigned>,
                     unsigned,
                     boost::hash<std::pair<const Node*, unsigned>>>
      index_map_;
};

}  // namespace hdk::ir
