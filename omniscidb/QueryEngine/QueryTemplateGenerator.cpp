#include "QueryTemplateGenerator.h"

#include <glog/logging.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Verifier.h>

// This file was pretty much auto-generated by running:
//      llc -march=cpp RuntimeFunctions.ll
// and formatting the results to be more readable.

namespace {

llvm::Function* default_func_builder(llvm::Module* mod, const std::string& name) {
  using namespace llvm;

  std::vector<Type*> func_args;
  FunctionType* func_type = FunctionType::get(
      /*Result=*/IntegerType::get(mod->getContext(), 32),
      /*Params=*/func_args,
      /*isVarArg=*/false);

  auto func_ptr = mod->getFunction(name);
  if (!func_ptr) {
    func_ptr = Function::Create(
        /*Type=*/func_type,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/name,
        mod);  // (external, no body)
    func_ptr->setCallingConv(CallingConv::C);
  }

  AttributeSet func_pal;
  {
    SmallVector<AttributeSet, 4> Attrs;
    AttributeSet PAS;
    {
      AttrBuilder B;
      PAS = AttributeSet::get(mod->getContext(), ~0U, B);
    }

    Attrs.push_back(PAS);
    func_pal = AttributeSet::get(mod->getContext(), Attrs);
  }
  func_ptr->setAttributes(func_pal);

  return func_ptr;
}

llvm::Function* pos_start(llvm::Module* mod) {
  return default_func_builder(mod, "pos_start");
}

llvm::Function* group_buff_idx(llvm::Module* mod) {
  return default_func_builder(mod, "group_buff_idx");
}

llvm::Function* pos_step(llvm::Module* mod) {
  using namespace llvm;

  std::vector<Type*> func_args;
  FunctionType* func_type = FunctionType::get(
      /*Result=*/IntegerType::get(mod->getContext(), 32),
      /*Params=*/func_args,
      /*isVarArg=*/false);

  auto func_ptr = mod->getFunction("pos_step");
  if (!func_ptr) {
    func_ptr = Function::Create(
        /*Type=*/func_type,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/"pos_step",
        mod);  // (external, no body)
    func_ptr->setCallingConv(CallingConv::C);
  }

  AttributeSet func_pal;
  {
    SmallVector<AttributeSet, 4> Attrs;
    AttributeSet PAS;
    {
      AttrBuilder B;
      PAS = AttributeSet::get(mod->getContext(), ~0U, B);
    }

    Attrs.push_back(PAS);
    func_pal = AttributeSet::get(mod->getContext(), Attrs);
  }
  func_ptr->setAttributes(func_pal);

  return func_ptr;
}

llvm::Function* init_group_by_buffer(llvm::Module* mod) {
  using namespace llvm;

  auto i64_type = IntegerType::get(mod->getContext(), 64);
  auto pi64_type = PointerType::get(i64_type, 0);
  auto i32_type = IntegerType::get(mod->getContext(), 32);

  std::vector<Type*> func_args{pi64_type, pi64_type, i32_type, i32_type, i32_type};

  auto func_type = FunctionType::get(Type::getVoidTy(mod->getContext()), func_args, false);

  auto func_ptr = mod->getFunction("init_group_by_buffer");
  if (!func_ptr) {
    func_ptr = Function::Create(func_type, GlobalValue::ExternalLinkage, "init_group_by_buffer", mod);
    func_ptr->setCallingConv(CallingConv::C);
  }

  AttributeSet func_pal;
  {
    SmallVector<AttributeSet, 4> Attrs;
    AttributeSet PAS;
    {
      AttrBuilder B;
      PAS = AttributeSet::get(mod->getContext(), ~0U, B);
    }

    Attrs.push_back(PAS);
    func_pal = AttributeSet::get(mod->getContext(), Attrs);
  }
  func_ptr->setAttributes(func_pal);

  return func_ptr;
}

llvm::Function* row_process(llvm::Module* mod,
                            const size_t aggr_col_count,
                            const bool is_nested,
                            const bool hoist_literals) {
  using namespace llvm;

  std::vector<Type*> func_args;
  auto i8_type = IntegerType::get(mod->getContext(), 8);
  auto i32_type = IntegerType::get(mod->getContext(), 32);
  auto i64_type = IntegerType::get(mod->getContext(), 64);
  auto pi32_type = PointerType::get(i32_type, 0);
  auto pi64_type = PointerType::get(i64_type, 0);

  if (aggr_col_count) {
    for (size_t i = 0; i < aggr_col_count; ++i) {
      func_args.push_back(pi64_type);
    }
  } else {                           // group by query
    func_args.push_back(pi64_type);  // groups buffer
    func_args.push_back(pi64_type);  // small groups buffer
    func_args.push_back(pi32_type);  // 1 iff current row matched, else 0
    func_args.push_back(pi32_type);  // total rows matched from the caller
    func_args.push_back(pi32_type);  // total rows matched before atomic increment
  }

  func_args.push_back(pi64_type);  // aggregate init values

  func_args.push_back(i64_type);
  func_args.push_back(i64_type);
  func_args.push_back(pi64_type);
  if (hoist_literals) {
    func_args.push_back(PointerType::get(i8_type, 0));
  }
  FunctionType* func_type = FunctionType::get(
      /*Result=*/i32_type,
      /*Params=*/func_args,
      /*isVarArg=*/false);

  auto func_name = unique_name("row_process", is_nested);
  auto func_ptr = mod->getFunction(func_name);

  if (!func_ptr) {
    func_ptr = Function::Create(
        /*Type=*/func_type,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/func_name,
        mod);  // (external, no body)
    func_ptr->setCallingConv(CallingConv::C);

    AttributeSet func_pal;
    {
      SmallVector<AttributeSet, 4> Attrs;
      AttributeSet PAS;
      {
        AttrBuilder B;
        PAS = AttributeSet::get(mod->getContext(), ~0U, B);
      }

      Attrs.push_back(PAS);
      func_pal = AttributeSet::get(mod->getContext(), Attrs);
    }
    func_ptr->setAttributes(func_pal);
  }

  return func_ptr;
}

}  // namespace

llvm::Function* query_template(llvm::Module* mod,
                               const size_t aggr_col_count,
                               const bool is_nested,
                               const bool hoist_literals,
                               const bool is_estimate_query) {
  using namespace llvm;

  auto func_pos_start = pos_start(mod);
  CHECK(func_pos_start);
  auto func_pos_step = pos_step(mod);
  CHECK(func_pos_step);
  auto func_group_buff_idx = group_buff_idx(mod);
  CHECK(func_group_buff_idx);
  auto func_row_process = row_process(mod, is_estimate_query ? 1 : aggr_col_count, is_nested, hoist_literals);
  CHECK(func_row_process);

  auto i8_type = IntegerType::get(mod->getContext(), 8);
  auto i32_type = IntegerType::get(mod->getContext(), 32);
  auto i64_type = IntegerType::get(mod->getContext(), 64);
  auto pi8_type = PointerType::get(i8_type, 0);
  auto ppi8_type = PointerType::get(pi8_type, 0);
  auto pi32_type = PointerType::get(i32_type, 0);
  auto pi64_type = PointerType::get(i64_type, 0);
  auto ppi64_type = PointerType::get(pi64_type, 0);

  std::vector<Type*> query_args;
  query_args.push_back(ppi8_type);
  if (hoist_literals) {
    query_args.push_back(pi8_type);
  }
  query_args.push_back(pi64_type);
  query_args.push_back(pi64_type);
  query_args.push_back(pi32_type);

  query_args.push_back(pi64_type);
  query_args.push_back(ppi64_type);
  query_args.push_back(ppi64_type);
  query_args.push_back(i32_type);
  query_args.push_back(i64_type);
  query_args.push_back(pi32_type);
  query_args.push_back(pi32_type);

  FunctionType* query_func_type = FunctionType::get(
      /*Result=*/Type::getVoidTy(mod->getContext()),
      /*Params=*/query_args,
      /*isVarArg=*/false);

  auto query_template_name = unique_name("query_template", is_nested);
  auto query_func_ptr = mod->getFunction(query_template_name);
  CHECK(!query_func_ptr);

  query_func_ptr = Function::Create(
      /*Type=*/query_func_type,
      /*Linkage=*/GlobalValue::ExternalLinkage,
      /*Name=*/query_template_name,
      mod);
  query_func_ptr->setCallingConv(CallingConv::C);

  AttributeSet query_func_pal;
  {
    SmallVector<AttributeSet, 4> Attrs;
    AttributeSet PAS;
    {
      AttrBuilder B;
      B.addAttribute(Attribute::NoCapture);
      PAS = AttributeSet::get(mod->getContext(), 1U, B);
    }

    Attrs.push_back(PAS);
    {
      AttrBuilder B;
      B.addAttribute(Attribute::NoCapture);
      PAS = AttributeSet::get(mod->getContext(), 2U, B);
    }

    Attrs.push_back(PAS);

    {
      AttrBuilder B;
      B.addAttribute(Attribute::NoCapture);
      Attrs.push_back(AttributeSet::get(mod->getContext(), 3U, B));
    }

    {
      AttrBuilder B;
      B.addAttribute(Attribute::NoCapture);
      Attrs.push_back(AttributeSet::get(mod->getContext(), 4U, B));
    }

    Attrs.push_back(PAS);

    query_func_pal = AttributeSet::get(mod->getContext(), Attrs);
  }
  query_func_ptr->setAttributes(query_func_pal);

  Function::arg_iterator query_arg_it = query_func_ptr->arg_begin();
  Value* byte_stream = query_arg_it;
  byte_stream->setName("byte_stream");
  Value* literals{nullptr};
  if (hoist_literals) {
    literals = ++query_arg_it;
    literals->setName("literals");
  }
  Value* row_count_ptr = ++query_arg_it;
  row_count_ptr->setName("row_count_ptr");
  Value* frag_row_off_ptr = ++query_arg_it;
  frag_row_off_ptr->setName("frag_row_off_ptr");
  Value* max_matched_ptr = ++query_arg_it;
  max_matched_ptr->setName("max_matched_ptr");
  Value* agg_init_val = ++query_arg_it;
  agg_init_val->setName("agg_init_val");
  Value* out = ++query_arg_it;
  out->setName("out");
  Value* unused = ++query_arg_it;
  unused->setName("unused");
  Value* frag_idx = ++query_arg_it;
  frag_idx->setName("frag_idx");
  Value* join_hash_table = ++query_arg_it;
  join_hash_table->setName("join_hash_table");
  Value* total_matched = ++query_arg_it;
  total_matched->setName("total_matched");
  Value* error_code = ++query_arg_it;
  error_code->setName("error_code");

  auto bb_entry = BasicBlock::Create(mod->getContext(), ".entry", query_func_ptr, 0);
  auto bb_preheader = BasicBlock::Create(mod->getContext(), ".loop.preheader", query_func_ptr, 0);
  auto bb_forbody = BasicBlock::Create(mod->getContext(), ".for.body", query_func_ptr, 0);
  auto bb_crit_edge = BasicBlock::Create(mod->getContext(), "._crit_edge", query_func_ptr, 0);
  auto bb_exit = BasicBlock::Create(mod->getContext(), ".exit", query_func_ptr, 0);

  // Block  (.entry)
  std::vector<Value*> result_ptr_vec;
  if (!is_estimate_query) {
    for (size_t i = 0; i < aggr_col_count; ++i) {
      auto result_ptr = new AllocaInst(i64_type, "result", bb_entry);
      result_ptr->setAlignment(8);
      result_ptr_vec.push_back(result_ptr);
    }
  }

  LoadInst* row_count = new LoadInst(row_count_ptr, "row_count", false, bb_entry);
  row_count->setAlignment(8);
  LoadInst* frag_row_off = new LoadInst(frag_row_off_ptr, "frag_row_off", false, bb_entry);
  frag_row_off->setAlignment(8);

  std::vector<Value*> agg_init_val_vec;
  if (!is_estimate_query) {
    for (size_t i = 0; i < aggr_col_count; ++i) {
      auto idx_lv = ConstantInt::get(i32_type, i);
      auto agg_init_gep = GetElementPtrInst::CreateInBounds(agg_init_val, idx_lv, "", bb_entry);
      auto agg_init_val = new LoadInst(agg_init_gep, "", false, bb_entry);
      agg_init_val->setAlignment(8);
      agg_init_val_vec.push_back(agg_init_val);
      auto init_val_st = new StoreInst(agg_init_val, result_ptr_vec[i], false, bb_entry);
      init_val_st->setAlignment(8);
    }
  }

  CallInst* pos_start = CallInst::Create(func_pos_start, "pos_start", bb_entry);
  pos_start->setCallingConv(CallingConv::C);
  pos_start->setTailCall(true);
  AttributeSet pos_start_pal;
  pos_start->setAttributes(pos_start_pal);

  CallInst* pos_step = CallInst::Create(func_pos_step, "pos_step", bb_entry);
  pos_step->setCallingConv(CallingConv::C);
  pos_step->setTailCall(true);
  AttributeSet pos_step_pal;
  pos_step->setAttributes(pos_step_pal);

  CallInst* group_buff_idx = nullptr;
  if (!is_estimate_query) {
    group_buff_idx = CallInst::Create(func_group_buff_idx, "group_buff_idx", bb_entry);
    group_buff_idx->setCallingConv(CallingConv::C);
    group_buff_idx->setTailCall(true);
    AttributeSet group_buff_idx_pal;
    group_buff_idx->setAttributes(group_buff_idx_pal);
  }

  CastInst* pos_start_i64 = new SExtInst(pos_start, i64_type, "", bb_entry);
  ICmpInst* enter_or_not = new ICmpInst(*bb_entry, ICmpInst::ICMP_SLT, pos_start_i64, row_count, "");
  BranchInst::Create(bb_preheader, bb_exit, enter_or_not, bb_entry);

  // Block .loop.preheader
  CastInst* pos_step_i64 = new SExtInst(pos_step, i64_type, "", bb_preheader);
  BranchInst::Create(bb_forbody, bb_preheader);

  // Block  .forbody
  Argument* pos_inc_pre = new Argument(i64_type);
  PHINode* pos = PHINode::Create(i64_type, 2, "pos", bb_forbody);
  pos->addIncoming(pos_start_i64, bb_preheader);
  pos->addIncoming(pos_inc_pre, bb_forbody);

  std::vector<Value*> row_process_params;
  row_process_params.insert(row_process_params.end(), result_ptr_vec.begin(), result_ptr_vec.end());
  if (is_estimate_query) {
    row_process_params.push_back(new LoadInst(out, "", false, bb_forbody));
  }
  row_process_params.push_back(agg_init_val);
  row_process_params.push_back(pos);
  row_process_params.push_back(frag_row_off);
  row_process_params.push_back(row_count_ptr);
  if (hoist_literals) {
    CHECK(literals);
    row_process_params.push_back(literals);
  }
  CallInst* row_process = CallInst::Create(func_row_process, row_process_params, "", bb_forbody);
  row_process->setCallingConv(CallingConv::C);
  row_process->setTailCall(false);
  AttributeSet row_process_pal;
  row_process->setAttributes(row_process_pal);

  BinaryOperator* pos_inc = BinaryOperator::CreateNSW(Instruction::Add, pos, pos_step_i64, "", bb_forbody);
  ICmpInst* loop_or_exit = new ICmpInst(*bb_forbody, ICmpInst::ICMP_SLT, pos_inc, row_count, "");
  BranchInst::Create(bb_forbody, bb_crit_edge, loop_or_exit, bb_forbody);

  // Block ._crit_edge
  std::vector<Instruction*> result_vec_pre;
  if (!is_estimate_query) {
    for (size_t i = 0; i < aggr_col_count; ++i) {
      auto result = new LoadInst(result_ptr_vec[i], ".pre.result", false, bb_crit_edge);
      result->setAlignment(8);
      result_vec_pre.push_back(result);
    }
  }

  BranchInst::Create(bb_exit, bb_crit_edge);

  // Block  .exit
  std::vector<PHINode*> result_vec;
  if (!is_estimate_query) {
    for (int64_t i = aggr_col_count - 1; i >= 0; --i) {
      auto result = PHINode::Create(IntegerType::get(mod->getContext(), 64), 2, "", bb_exit);
      result->addIncoming(result_vec_pre[i], bb_crit_edge);
      result->addIncoming(agg_init_val_vec[i], bb_entry);
      result_vec.insert(result_vec.begin(), result);
    }
  }

  if (!is_estimate_query) {
    for (size_t i = 0; i < aggr_col_count; ++i) {
      auto col_idx = ConstantInt::get(i32_type, i);
      auto out_gep = GetElementPtrInst::CreateInBounds(out, col_idx, "", bb_exit);
      auto col_buffer = new LoadInst(out_gep, "", false, bb_exit);
      col_buffer->setAlignment(8);
      auto slot_idx = BinaryOperator::CreateAdd(
          group_buff_idx, BinaryOperator::CreateMul(frag_idx, pos_step, "", bb_exit), "", bb_exit);
      auto target_addr = GetElementPtrInst::CreateInBounds(col_buffer, slot_idx, "", bb_exit);
      StoreInst* result_st = new StoreInst(result_vec[i], target_addr, false, bb_exit);
      result_st->setAlignment(8);
    }
  }

  ReturnInst::Create(mod->getContext(), bb_exit);

  // Resolve Forward References
  pos_inc_pre->replaceAllUsesWith(pos_inc);
  delete pos_inc_pre;

  if (verifyFunction(*query_func_ptr)) {
    LOG(FATAL) << "Generated invalid code. ";
  }

  return query_func_ptr;
}

llvm::Function* query_group_by_template(llvm::Module* mod,
                                        const bool is_nested,
                                        const bool hoist_literals,
                                        const QueryMemoryDescriptor& query_mem_desc,
                                        const ExecutorDeviceType device_type,
                                        const bool check_scan_limit) {
  using namespace llvm;

  auto func_pos_start = pos_start(mod);
  CHECK(func_pos_start);
  auto func_pos_step = pos_step(mod);
  CHECK(func_pos_step);
  auto func_group_buff_idx = group_buff_idx(mod);
  CHECK(func_group_buff_idx);
  auto func_row_process = row_process(mod, 0, is_nested, hoist_literals);
  CHECK(func_row_process);
  auto func_init_shared_mem = query_mem_desc.sharedMemBytes(device_type) ? mod->getFunction("init_shared_mem")
                                                                         : mod->getFunction("init_shared_mem_nop");
  CHECK(func_init_shared_mem);
  auto func_write_back =
      query_mem_desc.sharedMemBytes(device_type) ? mod->getFunction("write_back") : mod->getFunction("write_back_nop");
  CHECK(func_write_back);

  auto i32_type = IntegerType::get(mod->getContext(), 32);
  auto i64_type = IntegerType::get(mod->getContext(), 64);
  auto pi8_type = PointerType::get(IntegerType::get(mod->getContext(), 8), 0);
  auto pi32_type = PointerType::get(i32_type, 0);
  auto pi64_type = PointerType::get(i64_type, 0);
  auto ppi64_type = PointerType::get(pi64_type, 0);
  auto ppi8_type = PointerType::get(pi8_type, 0);

  std::vector<Type*> query_args;
  query_args.push_back(ppi8_type);
  if (hoist_literals) {
    query_args.push_back(pi8_type);
  }
  query_args.push_back(pi64_type);
  query_args.push_back(pi64_type);
  query_args.push_back(pi32_type);
  query_args.push_back(pi64_type);

  query_args.push_back(ppi64_type);
  query_args.push_back(ppi64_type);
  query_args.push_back(i32_type);
  query_args.push_back(i64_type);
  query_args.push_back(pi32_type);
  query_args.push_back(pi32_type);

  FunctionType* query_func_type = FunctionType::get(
      /*Result=*/Type::getVoidTy(mod->getContext()),
      /*Params=*/query_args,
      /*isVarArg=*/false);

  auto query_name = unique_name("query_group_by_template", is_nested);
  auto query_func_ptr = mod->getFunction(query_name);
  CHECK(!query_func_ptr);

  query_func_ptr = Function::Create(
      /*Type=*/query_func_type,
      /*Linkage=*/GlobalValue::ExternalLinkage,
      /*Name=*/"query_group_by_template",
      mod);

  query_func_ptr->setCallingConv(CallingConv::C);

  AttributeSet query_func_pal;
  {
    SmallVector<AttributeSet, 4> Attrs;
    AttributeSet PAS;
    {
      AttrBuilder B;
      B.addAttribute(Attribute::ReadNone);
      B.addAttribute(Attribute::NoCapture);
      PAS = AttributeSet::get(mod->getContext(), 1U, B);
    }

    Attrs.push_back(PAS);
    {
      AttrBuilder B;
      B.addAttribute(Attribute::ReadOnly);
      B.addAttribute(Attribute::NoCapture);
      PAS = AttributeSet::get(mod->getContext(), 2U, B);
    }

    Attrs.push_back(PAS);
    {
      AttrBuilder B;
      B.addAttribute(Attribute::ReadNone);
      B.addAttribute(Attribute::NoCapture);
      PAS = AttributeSet::get(mod->getContext(), 3U, B);
    }

    Attrs.push_back(PAS);
    {
      AttrBuilder B;
      B.addAttribute(Attribute::ReadOnly);
      B.addAttribute(Attribute::NoCapture);
      PAS = AttributeSet::get(mod->getContext(), 4U, B);
    }

    Attrs.push_back(PAS);
    {
      AttrBuilder B;
      B.addAttribute(Attribute::UWTable);
      PAS = AttributeSet::get(mod->getContext(), ~0U, B);
    }

    Attrs.push_back(PAS);

    query_func_pal = AttributeSet::get(mod->getContext(), Attrs);
  }
  query_func_ptr->setAttributes(query_func_pal);

  Function::arg_iterator query_arg_it = query_func_ptr->arg_begin();
  Value* byte_stream = query_arg_it;
  byte_stream->setName("byte_stream");
  Value* literals{nullptr};
  if (hoist_literals) {
    literals = ++query_arg_it;
    ;
    literals->setName("literals");
  }
  Value* row_count_ptr = ++query_arg_it;
  row_count_ptr->setName("row_count_ptr");
  Value* frag_row_off_ptr = ++query_arg_it;
  frag_row_off_ptr->setName("frag_row_off_ptr");
  Value* max_matched_ptr = ++query_arg_it;
  max_matched_ptr->setName("max_matched_ptr");
  Value* agg_init_val = ++query_arg_it;
  agg_init_val->setName("agg_init_val");
  Value* group_by_buffers = ++query_arg_it;
  group_by_buffers->setName("group_by_buffers");
  Value* small_groups_buffer = ++query_arg_it;
  small_groups_buffer->setName("small_groups_buffer");
  Value* frag_idx = ++query_arg_it;
  frag_idx->setName("frag_idx");
  Value* join_hash_table = ++query_arg_it;
  join_hash_table->setName("join_hash_table");
  Value* total_matched = ++query_arg_it;
  total_matched->setName("total_matched");
  Value* error_code = ++query_arg_it;
  error_code->setName("error_code");

  auto bb_entry = BasicBlock::Create(mod->getContext(), ".entry", query_func_ptr, 0);
  auto bb_preheader = BasicBlock::Create(mod->getContext(), ".loop.preheader", query_func_ptr, 0);
  auto bb_forbody = BasicBlock::Create(mod->getContext(), ".forbody", query_func_ptr, 0);
  auto bb_crit_edge = BasicBlock::Create(mod->getContext(), "._crit_edge", query_func_ptr, 0);
  auto bb_exit = BasicBlock::Create(mod->getContext(), ".exit", query_func_ptr, 0);

  // Block  .entry
  LoadInst* row_count = new LoadInst(row_count_ptr, "", false, bb_entry);
  row_count->setAlignment(8);
  LoadInst* frag_row_off = new LoadInst(frag_row_off_ptr, "", false, bb_entry);
  frag_row_off->setAlignment(8);
  LoadInst* max_matched = new LoadInst(max_matched_ptr, "", false, bb_entry);
  max_matched->setAlignment(4);
  auto crt_matched_ptr = new AllocaInst(i32_type, "crt_matched", bb_entry);
  auto old_total_matched_ptr = new AllocaInst(i32_type, "old_total_matched", bb_entry);
  CallInst* pos_start = CallInst::Create(func_pos_start, "", bb_entry);
  pos_start->setCallingConv(CallingConv::C);
  pos_start->setTailCall(true);
  AttributeSet pos_start_pal;
  pos_start->setAttributes(pos_start_pal);

  CallInst* pos_step = CallInst::Create(func_pos_step, "", bb_entry);
  pos_step->setCallingConv(CallingConv::C);
  pos_step->setTailCall(true);
  AttributeSet pos_step_pal;
  pos_step->setAttributes(pos_step_pal);

  CallInst* group_buff_idx = CallInst::Create(func_group_buff_idx, "", bb_entry);
  group_buff_idx->setCallingConv(CallingConv::C);
  group_buff_idx->setTailCall(true);
  AttributeSet group_buff_idx_pal;
  group_buff_idx->setAttributes(group_buff_idx_pal);

  CastInst* pos_start_i64 = new SExtInst(pos_start, i64_type, "", bb_entry);
  const PointerType* Ty = dyn_cast<PointerType>(group_by_buffers->getType());
  CHECK(Ty);
  GetElementPtrInst* group_by_buffers_gep = GetElementPtrInst::Create(
#if !(LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5)
      Ty->getArrayElementType(),
#endif
      group_by_buffers,
      group_buff_idx,
      "",
      bb_entry);
  LoadInst* col_buffer = new LoadInst(group_by_buffers_gep, "", false, bb_entry);
  col_buffer->setAlignment(8);
  LoadInst* small_buffer{nullptr};
  if (query_mem_desc.getSmallBufferSizeBytes()) {
    auto small_buffer_gep = GetElementPtrInst::Create(
#if !(LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5)
        Ty->getArrayElementType(),
#endif
        small_groups_buffer,
        group_buff_idx,
        "",
        bb_entry);
    small_buffer = new LoadInst(small_buffer_gep, "", false, bb_entry);
    small_buffer->setAlignment(8);
  }
  if (query_mem_desc.lazyInitGroups(device_type) && query_mem_desc.hash_type == GroupByColRangeType::MultiCol) {
    CHECK(!query_mem_desc.output_columnar);
    CallInst::Create(
        init_group_by_buffer(mod),
        std::vector<llvm::Value*>{
            col_buffer,
            agg_init_val,
            ConstantInt::get(IntegerType::get(mod->getContext(), 32), query_mem_desc.entry_count),
            ConstantInt::get(IntegerType::get(mod->getContext(), 32), query_mem_desc.group_col_widths.size()),
            ConstantInt::get(IntegerType::get(mod->getContext(), 32), query_mem_desc.getRowSize() / sizeof(int64_t)),
        },
        "",
        bb_entry);
  }
  auto shared_mem_bytes_lv = ConstantInt::get(i32_type, query_mem_desc.sharedMemBytes(device_type));
  auto result_buffer =
      CallInst::Create(func_init_shared_mem, std::vector<llvm::Value*>{col_buffer, shared_mem_bytes_lv}, "", bb_entry);
  ICmpInst* enter_or_not = new ICmpInst(*bb_entry, ICmpInst::ICMP_SLT, pos_start_i64, row_count, "");
  BranchInst::Create(bb_preheader, bb_exit, enter_or_not, bb_entry);

  // Block .loop.preheader
  CastInst* pos_step_i64 = new SExtInst(pos_step, i64_type, "", bb_preheader);
  BranchInst::Create(bb_forbody, bb_preheader);

  // Block .forbody
  Argument* pos_pre = new Argument(i64_type);
  PHINode* pos = PHINode::Create(i64_type, check_scan_limit ? 3 : 2, "pos", bb_forbody);

  std::vector<Value*> row_process_params;
  row_process_params.push_back(result_buffer);
  if (query_mem_desc.getSmallBufferSizeBytes()) {
    row_process_params.push_back(small_buffer);
  } else {
    row_process_params.push_back(Constant::getNullValue(pi64_type));
  }
  row_process_params.push_back(crt_matched_ptr);
  row_process_params.push_back(total_matched);
  row_process_params.push_back(old_total_matched_ptr);
  row_process_params.push_back(agg_init_val);
  row_process_params.push_back(pos);
  row_process_params.push_back(frag_row_off);
  row_process_params.push_back(row_count_ptr);
  if (hoist_literals) {
    CHECK(literals);
    row_process_params.push_back(literals);
  }
  if (check_scan_limit) {
    new StoreInst(ConstantInt::get(IntegerType::get(mod->getContext(), 32), 0), crt_matched_ptr, bb_forbody);
  }
  CallInst* row_process = CallInst::Create(func_row_process, row_process_params, "", bb_forbody);
  row_process->setCallingConv(CallingConv::C);
  row_process->setTailCall(true);
  AttributeSet row_process_pal;
  row_process->setAttributes(row_process_pal);

  BinaryOperator* pos_inc = BinaryOperator::Create(Instruction::Add, pos, pos_step_i64, "", bb_forbody);
  ICmpInst* loop_or_exit = new ICmpInst(*bb_forbody, ICmpInst::ICMP_SLT, pos_inc, row_count, "");
  if (check_scan_limit) {
    auto crt_matched = new LoadInst(crt_matched_ptr, "", false, bb_forbody);
    auto filter_match = BasicBlock::Create(mod->getContext(), "filter_match", query_func_ptr, bb_crit_edge);
    llvm::Value* new_total_matched = new LoadInst(old_total_matched_ptr, "", false, filter_match);
    new_total_matched = BinaryOperator::CreateAdd(new_total_matched, crt_matched, "", filter_match);
    CHECK(new_total_matched);
    ICmpInst* limit_not_reached = new ICmpInst(*filter_match, ICmpInst::ICMP_SLT, new_total_matched, max_matched, "");
    BranchInst::Create(bb_forbody,
                       bb_crit_edge,
                       BinaryOperator::Create(BinaryOperator::And, loop_or_exit, limit_not_reached, "", filter_match),
                       filter_match);
    auto filter_nomatch = BasicBlock::Create(mod->getContext(), "filter_nomatch", query_func_ptr, bb_crit_edge);
    BranchInst::Create(bb_forbody, bb_crit_edge, loop_or_exit, filter_nomatch);
    ICmpInst* crt_matched_nz =
        new ICmpInst(*bb_forbody, ICmpInst::ICMP_NE, crt_matched, ConstantInt::get(i32_type, 0), "");
    BranchInst::Create(filter_match, filter_nomatch, crt_matched_nz, bb_forbody);
    pos->addIncoming(pos_start_i64, bb_preheader);
    pos->addIncoming(pos_pre, filter_match);
    pos->addIncoming(pos_pre, filter_nomatch);
  } else {
    pos->addIncoming(pos_start_i64, bb_preheader);
    pos->addIncoming(pos_pre, bb_forbody);
    BranchInst::Create(bb_forbody, bb_crit_edge, loop_or_exit, bb_forbody);
  }

  // Block ._crit_edge
  BranchInst::Create(bb_exit, bb_crit_edge);

  // Block .exit
  CallInst::Create(func_write_back, std::vector<Value*>{col_buffer, result_buffer, shared_mem_bytes_lv}, "", bb_exit);
  ReturnInst::Create(mod->getContext(), bb_exit);

  // Resolve Forward References
  pos_pre->replaceAllUsesWith(pos_inc);
  delete pos_pre;

  if (verifyFunction(*query_func_ptr)) {
    LOG(FATAL) << "Generated invalid code. ";
  }

  return query_func_ptr;
}

std::string unique_name(const char* base_name, const bool is_nested) {
  char full_name[128] = {0};
  snprintf(full_name, sizeof(full_name), "%s_%u", base_name, static_cast<unsigned>(is_nested));
  return full_name;
}
