//===- Attributes.td - HDK dialect attribute definitions ----------*- tablegen -*-===//

#ifndef HDK_ATTRIBUTES
#define HDK_ATTRIBUTES

include "HDKBase.td"

// It's common to define a base classes for attributes in the same dialect. This
// removes the need to pass in the dialect for each attribute, and can also be used
// to define a few fields ahead of time.
class HDK_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<HDK_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}


def HDK_DatumAttr : HDK_Attr<"Datum", "datum"> {
    let summary = "An Attribute containing a Datum (union)";
    let description = [{
        A Datum represents a stored constant value.
    }];
    /// Here we've defined two parameters, one is the `self` type of the attribute
    /// (i.e. the type of the Attribute itself), and the other is the integer value
    /// of the attribute.
    let parameters = (ins AttributeSelfTypeParameter<"">:$type, "hdk::Datum":$value);

    /// Here we've defined a custom builder for the type, that removes the need to pass
    /// in an MLIRContext instance; as it can be infered from the `type`.
    let builders = [
        AttrBuilderWithInferredContext<(ins "mlir::Type":$type, "const hdk::Datum &":$value), [{
        return $_get(type.getContext(), type, value);
                      }]>
    ];

    /// Here we define the textual format of the attribute declaratively, which will
    /// automatically generate parser and printer logic. This will allow for
    /// instances of the attribute to be output as, for example:
    ///
    ///    #my.int<50> : !my.int<32> // a 32-bit integer of value 50.
    ///
    ///let assemblyFormat = "`<` $value `>`";

    /// Indicate that our attribute will add additional verification to the parameters.
    let genVerifyDecl = 1;

    /// Indicate to the ODS generator that we do not want the default builders,
    /// as we have defined our own simpler ones.
    let skipDefaultBuilders = 1;
}

#endif
