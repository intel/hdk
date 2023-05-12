target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "spir64-unknown-unknown"

declare i64 @__spirv_BuiltInGlobalInvocationId(i32 %dimention)
declare i64 @__spirv_BuiltInWorkgroupSize(i32 %dimention)

declare i64 @__spirv_BuiltInLocalInvocationId(i32 %dimention)
declare i64 @__spirv_BuiltInWorkgroupId(i32 %dimention)
declare i64 @__spirv_BuiltInNumWorkgroups(i32 %dimention)

declare i64 @__spirv_BuiltInSubgroupSize(i32 %dimention)

; TODO(Petr): this should be dynamically sized depending on the arch, but at least 64KB
@slm.buf.i64 = internal local_unnamed_addr addrspace(3) global [1024 x i64] zeroinitializer, align 8

; https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpControlBarrier
declare void @__spirv_ControlBarrier(i32 %execution_scope, i32 %memory_scope, i32 %memory_semantics)

; L0 zeroes out the memory by default, so an empty implementation can work in simple cases
define i64 addrspace(3)* @init_shared_mem(i64 addrspace(4)* %agg_init_val, i32 noundef %groups_buffer_size) {
    %res = bitcast [1024 x i64] addrspace(3)* @slm.buf.i64 to i64 addrspace(3)*
    ret i64 addrspace(3)* %res
}

define void @write_back_non_grouped_agg(i64 addrspace(3)* %input_buffer, i64 addrspace(4)* %output_buffer, i32 noundef %agg_idx) { 
    ; TODO
    ret void
}

define noundef i64 addrspace(3)* @declare_dynamic_shared_memory() {
    %res = bitcast [1024 x i64] addrspace(3)* @slm.buf.i64 to i64 addrspace(3)*
    ret i64 addrspace(3)* %res
}

define void @write_projection_int64(i8 addrspace(4)* nocapture noundef writeonly %0, i64 noundef %1, i64 noundef %2) {
  %4 = icmp eq i64 %1, %2
  br i1 %4, label %7, label %5

5:                                                ; preds = %3
  %6 = bitcast i8 addrspace(4)* %0 to i64 addrspace(4)*
  store i64 %1, i64 addrspace(4)* %6, align 8
  br label %7

7:                                                ; preds = %5, %3
  ret void
}

define void @write_projection_int32(i8 addrspace(4)* nocapture noundef writeonly %0, i32 noundef %1, i64 noundef %2) {
    ; TODO
    ret void
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind uwtable willreturn
define void @agg_sum_double_skip_val(i64 addrspace(4)* nocapture noundef %0, double noundef %1, double noundef %2) {
  %4 = fcmp une double %1, %2
  br i1 %4, label %5, label %15

5:                                                ; preds = %3
  %6 = load i64, i64 addrspace(4)* %0, align 8
  %7 = bitcast double %2 to i64
  %8 = icmp eq i64 %6, %7
  br i1 %8, label %13, label %9

9:                                                ; preds = %5
  %10 = bitcast i64 %6 to double
  %11 = bitcast i64 addrspace(4)* %0 to double addrspace(4)*
  %12 = fadd double %10, %1
  store double %12, double addrspace(4)* %11, align 8
  br label %15

13:                                               ; preds = %5
  %14 = bitcast i64 addrspace(4)* %0 to double addrspace(4)*
  store double %1, double addrspace(4)* %14, align 8
  br label %15

15:                                               ; preds = %13, %9, %3
  ret void
}


; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind uwtable willreturn
define dso_local void @agg_min_double_skip_val(i64 addrspace(4)* nocapture noundef %0, double noundef %1, double noundef %2) local_unnamed_addr #7 {
  %4 = fcmp une double %1, %2
  br i1 %4, label %5, label %14

5:                                                ; preds = %3
  %6 = load i64, i64 addrspace(4)* %0
  %7 = bitcast double %2 to i64
  %8 = icmp eq i64 %6, %7
  %9 = bitcast i64 %6 to double
  %10 = bitcast i64 addrspace(4)* %0 to double addrspace(4)*
  %11 = fcmp ogt double %9, %1
  %12 = or i1 %8, %11
  %13 = select i1 %12, double %1, double %9
  store double %13, double addrspace(4)* %10, align 8
  br label %14

14:                                               ; preds = %5, %3
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind uwtable willreturn
define dso_local void @agg_max_float_skip_val(i32 addrspace(4)* nocapture noundef %0, float noundef %1, float noundef %2) local_unnamed_addr #7 {
  %4 = fcmp une float %1, %2
  br i1 %4, label %5, label %14

5:                                                ; preds = %3
  %6 = load i32, i32 addrspace(4)* %0, align 4
  %7 = bitcast float %2 to i32
  %8 = icmp eq i32 %6, %7
  %9 = bitcast i32 %6 to float
  %10 = bitcast i32 addrspace(4)* %0 to float addrspace(4)*
  %11 = fcmp olt float %9, %1
  %12 = or i1 %8, %11
  %13 = select i1 %12, float %1, float %9
  store float %13, float addrspace(4)* %10, align 4
  br label %14

14:                                               ; preds = %5, %3
  ret void
}


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
