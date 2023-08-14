target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "spir64-unknown-unknown"

declare i64 @__spirv_BuiltInGlobalInvocationId(i32 %dimention)
declare i64 @__spirv_BuiltInWorkgroupSize(i32 %dimention)

declare i64 @__spirv_BuiltInLocalInvocationId(i32 %dimention)
declare i64 @__spirv_BuiltInWorkgroupId(i32 %dimention)
declare i64 @__spirv_BuiltInNumWorkgroups(i32 %dimention)

declare i64 @__spirv_BuiltInSubgroupSize(i32 %dimention)

declare void @__spirv_ControlBarrier(i32 %execution_scope, i32 %memory_scope, i32 %memory_semantics)

@slm.buf.i64 = internal local_unnamed_addr addrspace(3) global [4096 x i64] zeroinitializer, align 4

define i64 addrspace(4)* @declare_dynamic_shared_memory() {
    %res.share = bitcast [4096 x i64] addrspace(3)* @slm.buf.i64 to i64 addrspace(3)*
    %res = addrspacecast i64 addrspace(3)* %res.share to i64 addrspace(4)*
    ret i64 addrspace(4)* %res
}

define void @sync_threadblock() {
    call void @__spirv_ControlBarrier(i32 2, i32 2, i32 u0x100)
    ret void
}

define i64 @get_block_dim() {
    %gsize = call i64 @__spirv_BuiltInWorkgroupSize(i32 0)
    ret i64 %gsize
}

define i32 @pos_start_impl(i32 addrspace(4)* %0)  readnone nounwind alwaysinline {
    %gid = call i64 @__spirv_BuiltInWorkgroupId(i32 0)
    %gsize = call i64 @__spirv_BuiltInWorkgroupSize(i32 0)
    %tid = call i64 @__spirv_BuiltInLocalInvocationId(i32 0)
    %gid.i32 = trunc i64 %gid to i32
    %gsize.i32 = trunc i64 %gsize to i32
    %tid.i32 = trunc i64 %tid to i32
    %group_offset = mul i32 %gid.i32, %gsize.i32
    %pos_start = add i32 %group_offset, %tid.i32
    ret i32 %pos_start
}

define i32 @pos_step_impl() {
    %gid = call i64 @__spirv_BuiltInNumWorkgroups(i32 0)
    %gsize = call i64 @__spirv_BuiltInWorkgroupSize(i32 0)
    %gid.i32 = trunc i64 %gid to i32
    %gsize.i32 = trunc i64 %gsize to i32
    %res = mul i32 %gid.i32, %gsize.i32
    ret i32 %res
}

define i64 @get_thread_index() {
    %tid = call i64 @__spirv_BuiltInLocalInvocationId(i32 0)
    ret i64 %tid
}

define i64 @get_block_index() {
    %gid = call i64 @__spirv_BuiltInWorkgroupId(i32 0)
    ret i64 %gid
}

define i8 @thread_warp_idx(i8 noundef %warp_sz) {
    ret i8 0
}

define i64 @agg_count_shared(i64 addrspace(4)* %agg, i64 noundef %val) {
    %ld = load i64, i64 addrspace(4)* %agg
    %old = atomicrmw add i64 addrspace(4)* %agg, i64 1 monotonic
    ret i64 %old
}

define i32 @agg_count_int32_shared(i32 addrspace(4)* %agg, i32 noundef %val) {
    %ld = load i32, i32 addrspace(4)* %agg
    %old = atomicrmw add i32 addrspace(4)* %agg, i32 1 monotonic
    ret i32 %old
}

define i64 @agg_count_skip_val_shared(i64 addrspace(4)* %agg, i64 noundef %val, i64 noundef %skip_val) {
    %no_skip = icmp ne i64 %val, %skip_val
    br i1 %no_skip, label %.noskip, label %.skip
.noskip:
    %old = call i64 @agg_count_shared(i64 addrspace(4)* %agg, i64 %val)
    ret i64 %old
.skip:
    ret i64 0
}

define i64 @agg_count_double_skip_val_shared(i64 addrspace(4)* %agg, double noundef %val, double noundef %skip_val) {
    %no_skip = fcmp one double %val, %skip_val
    br i1 %no_skip, label %.noskip, label %.skip
.noskip:
    %val_cst = bitcast double %val to i64
    %res = call i64 @agg_count_shared(i64 addrspace(4)* %agg, i64 %val_cst)
    ret i64 %res
.skip:
    %orig = load i64, i64 addrspace(4)* %agg
    ret i64 %orig
}

;; TODO: may cause a CPU fallback on codegen
define i64 @agg_sum_shared(i64 addrspace(4)* %agg, i64 noundef %val) {
    %old = atomicrmw add i64 addrspace(4)* %agg, i64 %val monotonic
    ret i64 %old
}

define i32 @agg_sum_int32_shared(i32 addrspace(4)* %agg, i32 noundef %val) {
    %old = atomicrmw add i32 addrspace(4)* %agg, i32 %val monotonic
    ret i32 %old
}

define void @agg_sum_float_shared(i32 addrspace(4)* %agg, float noundef %val) {
.entry:
    %orig = load atomic i32, i32 addrspace(4)* %agg unordered, align 4
    %cst = bitcast i32 %orig to float
    br label %.loop
.loop:
    %cmp = phi i32 [ %orig, %.entry ], [ %loaded, %.loop ]
    %cmp_cst = bitcast i32 %cmp to float
    %new_val = fadd float %cmp_cst, %val
    %new_val_cst = bitcast float %new_val to i32
    %val_success = cmpxchg i32 addrspace(4)* %agg, i32 %cmp, i32 %new_val_cst acq_rel monotonic
    %loaded = extractvalue {i32, i1} %val_success, 0
    %success = extractvalue {i32, i1} %val_success, 1
    br i1 %success, label %.exit, label %.loop
.exit:
    ret void
}

define void @atomic_or(i32 addrspace(4)* %addr, i32 noundef %val) {
.entry:
    %orig = load atomic i32, i32 addrspace(4)* %addr unordered, align 8
    br label %.loop
.loop:
    %cmp = phi i32 [ %orig, %.entry ], [ %loaded , %.loop ]
    %new_val = or i32 %cmp, %val
    %val_success = cmpxchg i32 addrspace(4)* %addr, i32 %cmp, i32 %new_val acq_rel monotonic
    %loaded = extractvalue {i32, i1} %val_success, 0
    %success = extractvalue {i32, i1} %val_success, 1
    br i1 %success, label %.exit, label %.loop
.exit:
    ret void
}

define double @atomic_max_float(float addrspace(4)* %addr, float noundef %val) {
.entry:
    %orig = load float, float addrspace(4)* %addr, align 8
    br label %.loop
.loop:
    %loaded = phi float [ %orig, %.entry], [ %old.cst, %.loop ]
    %isless = fcmp ogt float %val, %loaded
    %min = select i1 %isless, float %val, float %loaded
    %min.cst = bitcast float %min to i32
    %loaded.cst = bitcast float %loaded to i32
    %addr.cst = bitcast float addrspace(4)* %addr to i32 addrspace(4)*
    %old = call i32 @atomic_cas_int_32(i32 addrspace(4)* %addr.cst, i32 %loaded.cst, i32 %min.cst)
    %old.cst = bitcast i32 %old to float
    %success = icmp eq i32 %old, %loaded.cst
    br i1 %success, label %.exit, label %.loop
.exit:
    %res = fpext float %old.cst to double
    ret double %res
}

define double @atomic_max_double(double addrspace(4)* %addr, double noundef %val) {
.entry:
    %orig = load double, double addrspace(4)* %addr, align 8
    br label %.loop
.loop:
    %loaded = phi double [ %orig, %.entry], [ %old.cst, %.loop ]
    %isless = fcmp ogt double %val, %loaded
    %min = select i1 %isless, double %val, double %loaded
    %min.cst = bitcast double %min to i64
    %loaded.cst = bitcast double %loaded to i64
    %addr.cst = bitcast double addrspace(4)* %addr to i64 addrspace(4)*
    %old = call i64 @atomic_cas_int_64(i64 addrspace(4)* %addr.cst, i64 %loaded.cst, i64 %min.cst)
    %old.cst = bitcast i64 %old to double
    %success = icmp eq i64 %old, %loaded.cst
    br i1 %success, label %.exit, label %.loop
.exit:
    ret double %old.cst
}

define void @write_back_non_grouped_agg(i64 addrspace(4)* %input_buffer, i64 addrspace(4)* %output_buffer, i32 noundef %agg_idx) {
    %tid = call i64 @get_thread_index()
    %agg_idx.i64 = sext i32 %agg_idx to i64
    %cmp = icmp eq i64 %tid, %agg_idx.i64
    br i1 %cmp, label %.agg, label %.exit
.agg:
    %gep = getelementptr inbounds i64, i64 addrspace(4)* %input_buffer, i64 %agg_idx.i64
    %val = load i64, i64 addrspace(4)* %gep
    %old = call i64 @agg_sum_shared(i64 addrspace(4)* %output_buffer, i64 %val)
    br label %.exit
.exit:
    ret void
}

define i64 @atomic_cas_int_64(i64 addrspace(4)* %p, i64 %cmp, i64 %val) {
    %val_success = cmpxchg i64 addrspace(4)* %p, i64 %cmp, i64 %val acq_rel monotonic
    %old = extractvalue {i64, i1} %val_success, 0
    ret i64 %old
}

define i32 @atomic_cas_int_32(i32 addrspace(4)* %p, i32 %cmp, i32 %val) {
    %val_success = cmpxchg i32 addrspace(4)* %p, i32 %cmp, i32 %val acq_rel monotonic
    %old = extractvalue {i32, i1} %val_success, 0
    ret i32 %old
}

define i64 @atomic_xchg_int_64(i64 addrspace(4)* %p, i64 %val) {
    %old = atomicrmw xchg i64 addrspace(4)* %p, i64 %val monotonic
    ret i64 %old
}

define i32 @atomic_xchg_int_32(i32 addrspace(4)* %p, i32 %val) {
    %old = atomicrmw xchg i32 addrspace(4)* %p, i32 %val monotonic
    ret i32 %old
}

define void @agg_max_int32_shared(i32 addrspace(4)* %agg, i32 noundef %val) {
    %old = atomicrmw max i32 addrspace(4)* %agg, i32 %val monotonic
    ret void
}

define void @agg_max_shared(i64 addrspace(4)* %agg, i64 noundef %val) {
    %old = atomicrmw max i64 addrspace(4)* %agg, i64 %val monotonic
    ret void
}

define void @agg_max_skip_val_shared(i64 addrspace(4)* %agg, i64 noundef %val, i64 noundef %skip_val) {
    %no_skip = icmp ne i64 %val, %skip_val
    br i1 %no_skip, label %.noskip, label %.skip
.noskip:
    call void @agg_max_shared(i64 addrspace(4)* %agg, i64 noundef %val)
    br label %.skip
.skip:
    ret void
}

define void @agg_min_shared(i64 addrspace(4)* %agg, i64 noundef %val) {
    %old = atomicrmw min i64 addrspace(4)* %agg, i64 %val acq_rel
    ret void
}

declare i64 @llvm.smin.i64(i64, i64)
declare i32 @llvm.smin.i32(i32, i32)

define void @agg_min_skip_val_shared(i64 addrspace(4)* %agg, i64 noundef %val, i64 noundef %skip_val) {
    %no_skip = icmp ne i64 %val, %skip_val
    br i1 %no_skip, label %.noskip, label %.skip
.noskip:
    %orig = load atomic i64, i64 addrspace(4)* %agg unordered, align 8
    br label %.loop
.loop:
    %loaded = phi i64 [ %orig, %.noskip ], [ %old, %.loop ]
    %isnull = icmp eq i64 %loaded, %skip_val
    %min = call i64 @llvm.smin.i64(i64 %loaded, i64 %val)
    %st = select i1 %isnull, i64 %val, i64 %min
    %old = call i64 @atomic_cas_int_64(i64 addrspace(4)* %agg, i64 %loaded, i64 %st)
    %success = icmp eq i64 %old, %loaded
    br i1 %success, label %.skip, label %.loop
.skip:
    ret void
}

define void @agg_min_int32_skip_val_shared(i32 addrspace(4)* %agg, i32 noundef %val, i32 noundef %skip_val) {
    %no_skip = icmp ne i32 %val, %skip_val
    br i1 %no_skip, label %.noskip, label %.skip
.noskip:
    %orig = load atomic i32, i32 addrspace(4)* %agg unordered, align 8
    br label %.loop
.loop:
    %loaded = phi i32 [ %orig, %.noskip ], [ %old, %.loop ]
    %isnull = icmp eq i32 %loaded, %skip_val
    %min = call i32 @llvm.smin.i32(i32 %loaded, i32 %val)
    %st = select i1 %isnull, i32 %val, i32 %min
    %old = call i32 @atomic_cas_int_32(i32 addrspace(4)* %agg, i32 %loaded, i32 %st)
    %success = icmp eq i32 %old, %loaded
    br i1 %success, label %.skip, label %.loop
.skip:
    ret void
}
