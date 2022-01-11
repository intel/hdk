#include "Expression/Types.h"

#include <memory>
#include <string>

namespace hdk {

class Expression {
 public:
  Expression(std::unique_ptr<Type>&& type) : type_(std::move(type)) {}

  virtual ~Expression() = default;

  virtual std::string toString() const = 0;

  Type* getTypeInfo() const { return type_.get(); }

 protected:
  std::unique_ptr<Type> type_;
};

class Column : public Expression {
 public:
  Column(std::unique_ptr<Type>&& type) : Expression(std::move(type)) {}

  std::string toString() const override;
};

// TODO: consider computing the output type based on the target expression over which we
// are performing the aggregate
class Aggregate : public Expression {
 public:
  enum class AggType { kAVG, kMIN, kMAX, kSUM, kCOUNT };

  Aggregate(std::unique_ptr<Type>&& type,
            const AggType aggregate_type,
            std::unique_ptr<Expression>&& target_expression)
      : Expression(std::move(type))
      , agg_type_(aggregate_type)
      , target_expression_(std::move(target_expression)) {}

  std::string toString() const override;

  AggType getAggType() const { return agg_type_; }
  static std::string aggTypeToString(const AggType type);

 private:
  AggType agg_type_;
  std::unique_ptr<Expression> target_expression_;
};

}  // namespace hdk
