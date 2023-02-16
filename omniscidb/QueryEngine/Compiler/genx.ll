target triple = "spir64-unknown-unknown"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

declare i64 @__spirv_BuiltInGlobalInvocationId(i32 %dimention)
declare i64 @__spirv_BuiltInWorkgroupSize(i32 %dimention)

declare i64 @__spirv_BuiltInLocalInvocationId(i32 %dimention)
declare i64 @__spirv_BuiltInWorkgroupId(i32 %dimention)
declare i64 @__spirv_BuiltInNumWorkgroups(i32 %dimention)

declare i64 @__spirv_BuiltInSubgroupSize(i32 %dimention)

declare void @__spirv_ControlBarrier(i32 %execution_scope, i32 %memory_scope, i32 %memory_semantics)
declare i64 @__spirv_AtomicIAdd(i64 addrspace(1)* %ptr, i32 %execution_scope, i32 %memory_semantics, i64 %val)

@slm.buf = internal local_unnamed_addr addrspace(3) global [1024 x i64] zeroinitializer, align 8

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

define i32 @group_buff_idx_impl() {
    %pstart = call i32 @pos_start_impl(i32 addrspace(4)* null)
    ret i32 %pstart
}

declare i64 @agg_count(i64* %agg, i64 %val)

; define i64 @agg_count(i64* %agg, i64 %val) {
;     %ld = load i64, i64* %agg
;     %old = atomicrmw add i64* %agg, i64 1 monotonic
;     %add = add i64 %old, 1
;     store i64 %add, i64* %agg
;     ret i64 %old
; }

define i64 @agg_count_skip_val(i64* %agg, i64 noundef %val, i64 noundef %skip_val) {
    %no_skip = icmp ne i64 %val, %skip_val
    br i1 %no_skip, label %.noskip, label %.skip
.noskip:
    %old = call i64 @agg_count(i64* %agg, i64 %val)
    ret i64 %old
.skip:
    ret i64 0
}

define i64 @agg_sum_shared(i64 addrspace(4)* %agg, i64 noundef %val) {
    %old = atomicrmw add i64 addrspace(4)* %agg, i64 %val monotonic
    ret i64 %old
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

; define i64 addrspace(3)* @init_shared_mem(i64 addrspace(4)* %agg_init_val, i32 noundef %groups_buffer_size) {
;     %res.ptr = bitcast [1024 x i64] addrspace(3)* @slm.buf to i64 addrspace(3)*
;     %gep = getelementptr i64, i64 addrspace(3)* %res.ptr, i64 0
;     ; store i64 %groups_buffer_size, i64 addrspace(3)* %gep
;     ; store i64 5, i64 addrspace(3)* %gep
;     ; call void @sync_threadblock()
;     call void @__spirv_ControlBarrier(i32 1, i32 1, i32 u0x200)
;     ret i64 addrspace(3)* %res.ptr
; }

define i64 addrspace(3)* @init_shared_mem(i64 addrspace(4)* %agg_init_val, i32 noundef %groups_buffer_size) {
.entry:
    ; %buf.units = ashr i32 %groups_buffer_size, 3
    %buf.units.i64 = sext i32 %groups_buffer_size to i64
    %pos = call i64 @get_thread_index()
    %wgsize = call i64 @__spirv_BuiltInWorkgroupSize(i32 0)
    %res.ptr = bitcast [1024 x i64] addrspace(3)* @slm.buf to i64 addrspace(3)*
    %loop.cond = icmp slt i64 %pos, %buf.units.i64
    br i1 %loop.cond, label %.for_body, label %.exit
.for_body:
    %pos.idx = phi i64 [ %pos, %.entry ], [ %pos.idx.new, %.for_body ]
    %agg_init_val.idx = getelementptr inbounds i64, i64 addrspace(4)* %agg_init_val, i64 %pos.idx
    ; %slm.idx = getelementptr inbounds [1024 x i64], [1024 x i64] addrspace(3)* @slm.buf, i64 0, i64 %pos.idx
    %slm.idx = getelementptr inbounds i64, i64 addrspace(3)* %res.ptr, i64 %pos.idx
    %val = load i64, i64 addrspace(4)* %agg_init_val.idx
    ; store i64 0, i64 addrspace(3)* %slm.idx
    %pos.idx.new = add nsw i64 %pos.idx, %wgsize
    %cond = icmp slt i64 %pos.idx.new, %buf.units.i64
    br i1 %cond, label %.for_body, label %.exit
.exit:
    call void @sync_threadblock()
    ret i64 addrspace(3)* %res.ptr
}

@.str = private unnamed_addr constant [3 x i8] c"%d\00", align 1
declare dso_local i32 @printf(i8* noundef, ...)

define void @write_back_non_grouped_agg(i64 addrspace(3)* %input_buffer, i64 addrspace(4)* %output_buffer, i32 noundef %agg_idx) {
    %tid = call i64 @get_thread_index()
    %gid = call i64 @__spirv_BuiltInWorkgroupId(i32 0)
    ; store i64 0, i64 addrspace(4)* %output_buffer
    %agg_idx.i64 = sext i32 %agg_idx to i64
    %cmp = icmp eq i64 %tid, %agg_idx.i64
    ; %grcmp = icmp eq i64 %gid, 0
    ; %and = and i1 %cmp, %grcmp
    br i1 %cmp, label %.agg, label %.exit
.agg:
    %gep = getelementptr inbounds i64, i64 addrspace(3)* %input_buffer, i64 %tid
    %val = load i64, i64 addrspace(3)* %gep
    %out.cast = addrspacecast i64 addrspace(4)* %output_buffer to i64 addrspace(1)*
    ; Using an atomic produces weird behavior. Having a store in the write_back_non_grouped_agg entry block
    ; solves the problem for some reason. The memory seem to be initialized correctly.
    %old = call spir_func i64 @__spirv_AtomicIAdd(i64 addrspace(1)* %out.cast, i32 1, i32 u0x8, i64 %val)
    ; store i64 %val, i64 addrspace(4)* %output_buffer
    br label %.exit
.exit:
    ret void
}

define void @sync_threadblock() {
    call void @__spirv_ControlBarrier(i32 2, i32 2, i32 u0x100)
    ret void
}
