target triple = "spir64-unknown-unknown"

declare i64 @__spirv_BuiltInGlobalInvocationId(i32 %dimention)
declare i64 @__spirv_BuiltInWorkgroupSize(i32 %dimention)

declare i64 @__spirv_BuiltInLocalInvocationId(i32 %dimention)
declare i64 @__spirv_BuiltInWorkgroupId(i32 %dimention)
declare i64 @__spirv_BuiltInNumWorkgroups(i32 %dimention)

declare i64 @__spirv_BuiltInSubgroupSize(i32 %dimention)

@slm.buf.i64 = internal local_unnamed_addr addrspace(3) global [1024 x i64] zeroinitializer, align 8

define i32 @pos_start_impl(i32* %0)  readnone nounwind alwaysinline {
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

define i64 @agg_count(i64* %agg, i64 %val) {
    %ld = load i64, i64* %agg
    %old = atomicrmw add i64* %agg, i64 1 monotonic
    ret i64 %old
}

define i64 @agg_count_skip_val(i64* %agg, i64 noundef %val, i64 noundef %skip_val) {
    %no_skip = icmp ne i64 %val, %skip_val
    br i1 %no_skip, label %.noskip, label %.skip
.noskip:
    %old = call i64 @agg_count(i64* %agg, i64 %val)
    ret i64 %old
.skip:
    ret i64 0
}

define i64 @agg_count_double_skip_val(i64* %agg, double noundef %val, double noundef %skip_val) {
    %no_skip = fcmp one double %val, %skip_val
    br i1 %no_skip, label %.noskip, label %.skip
.noskip:
    %val_cst = bitcast double %val to i64
    %res = call i64 @agg_count(i64* %agg, i64 %val_cst)
    ret i64 %res
.skip:
    %orig = load i64, i64* %agg
    ret i64 %orig
}

;; TODO: may cause a CPU fallback on codegen
define i64 @agg_sum(i64* %agg, i64 noundef %val) {
    %old = atomicrmw add i64* %agg, i64 %val monotonic
    ret i64 %old
}

define void @agg_sum_float(i32 addrspace(4)* %agg, float noundef %val) {
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

define void @agg_sum_float_skip_val(i32 addrspace(4)* %agg, float noundef %val, float noundef %skip_val) {
    %no_skip = fcmp one float %val, %skip_val
    br i1 %no_skip, label %.noskip, label %.skip
.noskip:
    call void @agg_sum_float(i32 addrspace(4)* %agg, float noundef %val)
    br label %.skip
.skip:
    ret void
}

define void @agg_sum_double(i64 addrspace(4)* %agg, double noundef %val) {
.entry:
    %orig = load atomic i64, i64 addrspace(4)* %agg unordered, align 8
    %cst = bitcast i64 %orig to double
    br label %.loop
.loop:
    %cmp = phi i64 [ %orig, %.entry ], [ %loaded, %.loop ]
    %cmp_cst = bitcast i64 %cmp to double
    %new_val = fadd double %cmp_cst, %val
    %new_val_cst = bitcast double %new_val to i64
    %val_success = cmpxchg i64 addrspace(4)* %agg, i64 %cmp, i64 %new_val_cst acq_rel monotonic
    %loaded = extractvalue {i64, i1} %val_success, 0
    %success = extractvalue {i64, i1} %val_success, 1
    br i1 %success, label %.exit, label %.loop
.exit:
    ret void
}

define void @agg_sum_double_skip_val(i64 addrspace(4)* %agg, double noundef %val, double noundef %skip_val) {
    %no_skip = fcmp one double %val, %skip_val
    br i1 %no_skip, label %.noskip, label %.skip
.noskip:
    call void @agg_sum_double(i64 addrspace(4)* %agg, double noundef %val)
    br label %.skip
.skip:
    ret void
}


define i64 @agg_sum_skip_val(i64* %agg, i64 noundef %val, i64 noundef %skip_val) {
    %no_skip = icmp ne i64 %val, %skip_val
    br i1 %no_skip, label %.noskip, label %.skip
.noskip:
    %old = atomicrmw xchg i64* %agg, i64 0 monotonic
    %isempty = icmp eq i64 %old, -9223372036854775808
    %sel = select i1 %isempty, i64 0, i64 %old
    %new_val = add nsw i64 %val, %sel
    %old2 = atomicrmw add i64* %agg, i64 %new_val monotonic
    ret i64 %old2
.skip:
    ret i64 0
}
