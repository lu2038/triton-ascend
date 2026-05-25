/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_DYNAMIC_CV_PIPELINE_PASSES
#define TRITON_DYNAMIC_CV_PIPELINE_PASSES

include "mlir/Pass/PassBase.td"

def AddDynamicCVPipeline : Pass<"add_dynamic_cv_pipeline", "mlir::ModuleOp"> {
    let summary = "Add dynamic cv pipeline based on ssbufer";
    let constructor = "triton::createAddDynamicCVPipelinePass()";
    let options = [
        Option<"compileOn91095", "compile-on-910-95",
        "bool", /*default*/"false",
        "compile on 910_95">
    ];
    let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect", "LLVM::LLVMDialect"];
}

def OpClassifierPass : Pass<"op-classifier", "mlir::ModuleOp"> {
  let summary = "Classify operations as CUBE or VECTOR";
  let constructor = "triton::createOpClassifierPass()";
}

def PlanComputeBlockPass : Pass<"plan-compute-block", "mlir::ModuleOp"> {
  let summary = "Plan compute block";
  let constructor = "triton::createPlanComputeBlockPass()";
}

def AddBlockIdForControlOpsPass : Pass<"add-block-id-for-control-ops", "mlir::ModuleOp"> {
    let summary = "Add block_id attribute to control flow operations";
    let constructor = "triton::createAddBlockIdForControlOpsPass()";
}

def DataDependencyAnalysisPass : Pass<"data-dependency-analysis", "mlir::ModuleOp"> {
    let summary = "Analyze data dependencies between Vector and Cube blocks";
    let constructor = "triton::createDataDependencyAnalysisPass()";
}

def InterCoreTransferAndSyncPass : Pass<"inter-core-transfer-and-sync", "mlir::ModuleOp"> {
    let summary = "Insert inter-core transfer operations and PIPE_S synchronization between Vector and Cube cores";
    let constructor = "triton::createInterCoreTransferAndSyncPass()";
    let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def MarkMainLoopPass : Pass<"mark-main-loop", "mlir::ModuleOp"> {
    let summary = "Mark main computation loops containing FixpipeOp or CopyOp with main_loop attribute";
    let constructor = "triton::createMarkMainLoopPass()";
    let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def SeparateCVScopePass : Pass<"separate-cv-scope", "mlir::ModuleOp"> {
    let summary = "separate ops into cv scope";
    let constructor = "triton::createSeparateCVScopePass()";
    let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def PreserveControlAttrsCanonicalizePass : Pass<"preserve-control-attrs-canonicalize", "mlir::ModuleOp"> {
    let summary = "Canonicalize while preserving selected control-flow attrs";
    let constructor = "triton::createPreserveControlAttrsCanonicalizePass()";
}

def ComputeBlockOptPass : Pass<"compute-block-id-opt", "mlir::ModuleOp"> {
  let summary = "Block ID Optimization Pass";
  let constructor = "triton::createComputeBlockOptPass()";
}

def RemoveSsbufAttrPass : Pass<"remove-ssbuf-attr", "mlir::ModuleOp"> {
  let summary = "Remove ssbuf attributes";
  let constructor = "triton::createRemoveSsbufAttrPass()";
}

// Sub-passes for PlanComputeBlock
def PlanCubeBlockPass : Pass<"plan-cube-block", "mlir::ModuleOp"> {
  let summary = "Plan compute blocks for CUBE core";
  let constructor = "triton::createPlanCubeBlockPass()";
}

def PlanVectorBlockPass : Pass<"plan-vector-block", "mlir::ModuleOp"> {
  let summary = "Plan compute blocks for VECTOR core";
  let constructor = "triton::createPlanVectorBlockPass()";
}

def ReorderOpsByBlockIdPass : Pass<"reorder-ops-by-block-id", "mlir::ModuleOp"> {
  let summary = "Reorder operations by block ID";
  let constructor = "triton::createReorderOpsByBlockIdPass()";
}

// Sub-passes for ComputeBlockOpt
def UnifyAllocBlockPass : Pass<"unify-alloc-block", "mlir::ModuleOp"> {
  let summary = "Unify block_id for memref.alloc, scf.if with linalg.fill, and memref.subview operations";
  let constructor = "triton::createUnifyAllocBlockPass()";
}

def UBUsageOptPass : Pass<"ub-usage-opt", "mlir::ModuleOp"> {
  let summary = "Optimize UB buffer usage";
  let constructor = "triton::createUBUsageOptPass()";
}

def FuseAdotBaddCPass : Pass<"fuse-adotbaddc", "mlir::ModuleOp"> {
  let summary = "Fuse A.dot(B) + C patterns";
  let constructor = "triton::createFuseAdotBaddCPass()";
}

// Sub-passes for SplitDataflow
def SplitDataflowPass : Pass<"split-dataflow", "mlir::ModuleOp"> {
  let summary = "Split dataflow between CUBE and VECTOR cores";
  let constructor = "triton::createSplitDataflowPass()";
}

// Sub-passes for SeparateMemoryFromCompute
def SeparateMemoryFromComputePass : Pass<"separate-memory-from-compute", "mlir::ModuleOp"> {
  let summary = "Separate memory operations from compute operations";
  let constructor = "triton::createSeparateMemoryFromComputePass()";
}

def AsyncLoadHoistingPass : Pass<"async-load-hoisting", "mlir::ModuleOp"> {
  let summary = "Hoist async load operations out of loops";
  let constructor = "triton::createAsyncLoadHoistingPass()";
}

def AddMultiBufferToGMLoadPass : Pass<"gm-load-multi-buffer", "mlir::ModuleOp"> {
  let summary = "Add multi-buffer to GM load operations";
  let constructor = "triton::createAddMultiBufferToGMLoadPass()";
}

// Sub-passes for AllocMultiCache
def AllocMultiCachePass : Pass<"alloc-multi-cache", "mlir::ModuleOp"> {
  let summary = "Allocate multi-cache buffers for cross-core communication";
  let constructor = "triton::createAllocMultiCachePass()";
}

def AddMultiBufferInnerScopePass : Pass<"add_multi_buffer_inner_scope", "mlir::ModuleOp"> {
  let summary = "Add multi-buffer within inner scope";
  let constructor = "triton::createAddMultiBufferInnerScopePass()";
}

def AddMultiBufferOuterScopePass : Pass<"add_multi_buffer_outer_scope", "mlir::ModuleOp"> {
  let summary = "Add multi-buffer within outer scope";
  let constructor = "triton::createAddMultiBufferOuterScopePass()";
}

// Sub-passes for AddControlFlowCondition
def InitDependentMapPass : Pass<"init-dependent-map", "mlir::ModuleOp"> {
  let summary = "Initialize dependent map for control flow condition";
  let constructor = "triton::createInitDependentMapPass()";
}

def ProcessArgsPass : Pass<"process-args", "mlir::ModuleOp"> {
  let summary = "Process arguments for control flow condition";
  let constructor = "triton::createProcessArgsPass()";
}

def CloneOpsPass : Pass<"clone-ops", "mlir::ModuleOp"> {
  let summary = "Clone operations for control flow condition";
  let constructor = "triton::createCloneOpsPass()";
}

def CreateIfOpsPass : Pass<"create-if-ops", "mlir::ModuleOp"> {
  let summary = "Create if operations for control flow condition";
  let constructor = "triton::createCreateIfOpsPass()";
}

def UpdateForOpsPass : Pass<"update-for-ops", "mlir::ModuleOp"> {
  let summary = "Update for operations for control flow condition";
  let constructor = "triton::createUpdateForOpsPass()";
}

def UpdateConditionInfoPass : Pass<"update-condition-info", "mlir::ModuleOp"> {
  let summary = "Update condition info for control flow condition";
  let constructor = "triton::createUpdateConditionInfoPass()";
}

def UpdateLoopIterTimesPass : Pass<"update-loop-iter-times", "mlir::ModuleOp"> {
  let summary = "Update loop iteration times for control flow condition";
  let constructor = "triton::createUpdateLoopIterTimesPass()";
}

#endif // TRITON_DYNAMIC_CV_PIPELINE_PASSES