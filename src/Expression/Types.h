#include <cstdint>
#include <string>

namespace hdk {

class Type {
 public:
  virtual ~Type() = default;

  virtual int32_t size() const = 0;
  virtual std::string toString() const = 0;
};

class TinyInt : public Type {
 public:
  int32_t size() const override { return 1; }

  std::string toString() const override { return "TINYINT"; }
};

class SmallInt : public Type {
 public:
  int32_t size() const override { return 2; }

  std::string toString() const override { return "SMALLINT"; }
};

class Int : public Type {
 public:
  int32_t size() const override { return 4; }

  std::string toString() const override { return "INT"; }
};

class BigInt : public Type {
 public:
  int32_t size() const override { return 8; }

  std::string toString() const override { return "BIGINT"; }
};

class Float : public Type {
 public:
  int32_t size() const override { return 4; }

  std::string toString() const override { return "FLOAT"; }
};

class Double : public Type {
 public:
  int32_t size() const override { return 8; }

  std::string toString() const override { return "DOUBLE"; }
};

class String : public Type {
 public:
  int32_t size() const override { return -1; }

  std::string toString() const override { return "STRING"; }
};

}  // namespace hdk
