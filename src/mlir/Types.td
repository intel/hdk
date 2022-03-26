#ifndef HDK_TYPES
#define HDK_TYPES

include "HDKBase.td"

class HDK_Type<string name, string typeMnemonic,
                 string baseCppClass = "::mlir::Type">
    : TypeDef<HDK_Dialect, name, [], baseCppClass> {
  let mnemonic = typeMnemonic;
}

def HDK_KernelType : HDK_Type<"Kernel", "hdk.Kernel"> {
    let summary = "hdk.Kernel";
    let description = "Represents an HDK kernel.";
}

// SQL Types mapping to sqltypes.h
def HDK_NullType : HDK_Type<"Null", "null"> {
    let summary = "HDK Null Type";
}

def HDK_Boolean : HDK_Type<"Boolean", "boolean"> {
    let summary = "HDK Boolean Type";
}

def HDK_Decimal : HDK_Type<"Decimal", "decimal"> {
    let summary = "HDK Decimal/Numeric Type";
}

def HDK_Int : HDK_Type<"Int", "int"> {
    let summary = "HDK 32-bit Integer Type";
}

def HDK_TinyInt : HDK_Type<"TinyInt", "tinyint"> {
    let summary = "HDK 8-bit Integer Type";
}

def HDK_SmallInt : HDK_Type<"SmallInt", "smallint"> {
    let summary = "HDK 16-bit Integer Type";
}

def HDK_BigInt : HDK_Type<"BigInt", "bigint"> {
    let summary = "HDK 64-bit Integer Type";
}

def HDK_String : HDK_Type<"String", "string"> {
    let summary = "HDK String Type";
    let description = [{
        Currently a catch-all for text, dictionary encoded text,
        char, and varchar types.
    }];
}

def HDK_Date : HDK_Type<"Date", "date"> {
    let summary = "HDK Date Type";
}

def HDK_Array : HDK_Type<"Array", "array"> {
    let summary = "HDK Array Type";

    let parameters = (ins "::mlir::Type":$elementType);

    let printer = [{
        $_printer << "<" << getImpl()->elementType << ">";
      }];

      let parser = [{
        if ($_parser.parseLess())
            return Type();
        Type elementType;
        if ($_parser.parseType(elementType))
            return Type();
        if ($_parser.parseGreater())
          return Type();
        return get($_ctxt, elementType);
      }];

      let builders = [
        TypeBuilderWithInferredContext<(ins "::mlir::Type":$elementType), [{
         return Base::get(elementType.getContext(), elementType);
        }]>
      ];
}

def HDK_Datum : HDK_Type<"Datum", "datum"> {
    let summary = "HDK Datum Type";
    let description = "Container for Datum C++ union.";

    //let parameters = (
}

def HDK_AnySQLType : AnyTypeOf<[
    HDK_NullType,
    HDK_Boolean,
    HDK_Decimal,
    HDK_Int,
    HDK_TinyInt,
    HDK_SmallInt,
    HDK_BigInt,
    HDK_String,
    HDK_Date,
    HDK_Array
], "Any SQL Operator type.">;

#endif
