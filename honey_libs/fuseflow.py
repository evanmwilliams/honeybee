import os
import datetime
import polars as pl

from honey_lang import Helper, Input, Output, Function, __hb_bash

@Helper
class Dir:
    stage = 1

    def make(name):
        time = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        dir = f"output-{time}/{Dir.stage * 10:03d}-{name}"
        os.makedirs(dir, exist_ok=True)
        Dir.stage += 1
        return dir


@Helper
def carry_over(src_object, dst_object, *, file=None):
    def carry_one(file):
        src = f"{src_object.path}/{file}"
        dst = f"{dst_object.path}/{file}"
        if os.path.islink(src):
            src = os.readlink(src)
        os.symlink(src=src, dst=dst)

    if file is None:
        for file in os.listdir(src_object.path):
            carry_one(file)
    else:
        carry_one(file)

################################################################################
# %% FuseFlow Schedule
@Input 
class MlirProgram: 
    """An MLIR program in your filesystem

    This is the program that we use to generate the schedule."""
    path: str 
    """Path to the MLIR program"""


@Output 
class VectorizationPass:
    """VectorizationPass

    The goal of this step is to produce the schedule for the parallelization 
    pass in the FuseFlow compiler"""
    path: str


@Output
class VecStreamLevelChoice:
    """VecStreamLevelChoice

    The goal of this step is to choose a stream level for parallelization
    when also selecting vectorization parameters."""
    path: str

    stream_level: int


@Output
class VecParFactorChoice:
    """VecParFactorChoice

    The goal of this step is to choose a parallelization factor when also
    selecting vectorization parameters."""
    path: str

    par_factor: int


@Output
class VecStreamShapeChoice:
    """VecStreamShapeChoice

    The goal of this step is to choose a stream shape when also selecting
    vectorization parameters."""
    path: str

    stream_shape: int


@Output
class VecBlockSparseChoice:
    """VecBlockSparseChoice

    The goal of this step is to choose whether block-sparse vectorization is
    enabled when also selecting vectorization parameters."""
    path: str

    block_sparse: bool


@Output
class StreamShapeChoice:
    """StreamShapeChoice

    The goal of this step is to choose a stream shape for vectorization."""
    path: str

    stream_shape: int


@Output
class BlockSparseChoice:
    """BlockSparseChoice

    The goal of this step is to choose whether block-sparse vectorization is enabled."""
    path: str

    block_sparse: bool


@Output
class StreamLevelChoice:
    """StreamLevelChoice

    The goal of this step is to choose a stream level for parallelization."""
    path: str

    stream_level: int


@Output
class ParallelizationPass:
    """ParallelizationPass

    The goal of this step is to produce the schedule for the parallelization 
    pass in the FuseFlow compiler"""
    path: str


@Output
class ParFactorChoice:
    """ParFactorChoice

    The goal of this step is to choose a parallelization factor."""
    path: str

    par_factor: int


@Output
class FuseFlowSchedule:
    """FuseFlowSchedule 

    The goal of this step is to produce a schedule for the FuseFlow compiler 
    that sets the parallelization and vectorization pass parameters"""
    path: str

@Function(
)
def default_schedule(__hb_ret: FuseFlowSchedule):
    """schedule 

    The function that produces a schedule."""
    print("This is a default schedule.")


@Function(
)
def build_schedule(
    __hb_pass: ParallelizationPass,
    __hb_ret: FuseFlowSchedule,
):
    print("Build the schedule")

@Function(
    "ret.stream_level = 0",
)
def choose_default_stream_level(
    __hb_par: ParFactorChoice,
    __hb_ret: StreamLevelChoice,
):
    print("Choose default stream level (0).")

@Function(
    "ret.stream_level = 1",
)
def choose_stream_level_1(__hb_par: ParFactorChoice, __hb_ret: StreamLevelChoice):
    print("Choose stream level 1.")

@Function(
    "ret.stream_level = 2",
)
def choose_stream_level_2(__hb_par: ParFactorChoice, __hb_ret: StreamLevelChoice):
    print("Choose stream level 2.")

@Function(
    "ret.stream_level = 4",
)
def choose_stream_level_4(__hb_par: ParFactorChoice, __hb_ret: StreamLevelChoice):
    print("Choose stream level 4.")

@Function(
    "ret.stream_level = 8",
)
def choose_stream_level_8(__hb_par: ParFactorChoice, __hb_ret: StreamLevelChoice):
    print("Choose stream level 8.")

@Function(
    "ret.stream_level = 16",
)
def choose_stream_level_16(__hb_par: ParFactorChoice, __hb_ret: StreamLevelChoice):
    print("Choose stream level 16.")

@Function(
    "ret.par_factor = 1",
)
def choose_default_par_factor(__hb_ret: ParFactorChoice):
    print("Choose default par factor (1).")

@Function(
    "ret.par_factor = 2",
)
def choose_par_factor_2(__hb_ret: ParFactorChoice):
    print("Choose par factor 2.")

@Function(
    "ret.par_factor = 4",
)
def choose_par_factor_4(__hb_ret: ParFactorChoice):
    print("Choose par factor 4.")

@Function(
    "ret.par_factor = 8",
)
def choose_par_factor_8(__hb_ret: ParFactorChoice):
    print("Choose par factor 8.")

@Function(
    "ret.par_factor = 16",
)
def choose_par_factor_16(__hb_ret: ParFactorChoice):
    print("Choose par factor 16.")

@Function(
    "ret.stream_level = 0",
)
def choose_default_par_stream_level(
    __hb_par: VecParFactorChoice,
    __hb_ret: VecStreamLevelChoice,
):
    print("Choose default stream level (0).")

@Function(
    "ret.stream_level = 1",
)
def choose_par_stream_level_1(
    __hb_par: VecParFactorChoice,
    __hb_ret: VecStreamLevelChoice,
):
    print("Choose stream level 1.")

@Function(
    "ret.stream_level = 2",
)
def choose_par_stream_level_2(
    __hb_par: VecParFactorChoice,
    __hb_ret: VecStreamLevelChoice,
):
    print("Choose stream level 2.")

@Function(
    "ret.stream_level = 4",
)
def choose_par_stream_level_4(
    __hb_par: VecParFactorChoice,
    __hb_ret: VecStreamLevelChoice,
):
    print("Choose stream level 4.")

@Function(
    "ret.stream_level = 8",
)
def choose_par_stream_level_8(
    __hb_par: VecParFactorChoice,
    __hb_ret: VecStreamLevelChoice,
):
    print("Choose stream level 8.")

@Function(
    "ret.stream_level = 16",
)
def choose_par_stream_level_16(
    __hb_par: VecParFactorChoice,
    __hb_ret: VecStreamLevelChoice,
):
    print("Choose stream level 16.")

@Function(
    "ret.par_factor = 1",
)
def choose_default_par_factor_for_vec(
    __hb_shape: VecStreamShapeChoice,
    __hb_ret: VecParFactorChoice,
):
    print("Choose default par factor (1).")

@Function(
    "ret.par_factor = 2",
)
def choose_par_factor_for_vec_2(
    __hb_shape: VecStreamShapeChoice,
    __hb_ret: VecParFactorChoice,
):
    print("Choose par factor 2.")

@Function(
    "ret.par_factor = 4",
)
def choose_par_factor_for_vec_4(
    __hb_shape: VecStreamShapeChoice,
    __hb_ret: VecParFactorChoice,
):
    print("Choose par factor 4.")

@Function(
    "ret.par_factor = 8",
)
def choose_par_factor_for_vec_8(
    __hb_shape: VecStreamShapeChoice,
    __hb_ret: VecParFactorChoice,
):
    print("Choose par factor 8.")

@Function(
    "ret.par_factor = 16",
)
def choose_par_factor_for_vec_16(
    __hb_shape: VecStreamShapeChoice,
    __hb_ret: VecParFactorChoice,
):
    print("Choose par factor 16.")

@Function(
    "ret.stream_shape = 16",
)
def choose_default_vec_stream_shape(
    __hb_block: VecBlockSparseChoice,
    __hb_ret: VecStreamShapeChoice,
):
    print("Choose default stream shape (16).")

@Function(
    "ret.stream_shape = 1",
)
def choose_vec_stream_shape_1(
    __hb_block: VecBlockSparseChoice,
    __hb_ret: VecStreamShapeChoice,
):
    print("Choose stream shape 1.")

@Function(
    "ret.stream_shape = 2",
)
def choose_vec_stream_shape_2(
    __hb_block: VecBlockSparseChoice,
    __hb_ret: VecStreamShapeChoice,
):
    print("Choose stream shape 2.")

@Function(
    "ret.stream_shape = 4",
)
def choose_vec_stream_shape_4(
    __hb_block: VecBlockSparseChoice,
    __hb_ret: VecStreamShapeChoice,
):
    print("Choose stream shape 4.")

@Function(
    "ret.stream_shape = 8",
)
def choose_vec_stream_shape_8(
    __hb_block: VecBlockSparseChoice,
    __hb_ret: VecStreamShapeChoice,
):
    print("Choose stream shape 8.")

@Function(
    "ret.stream_shape = 16",
)
def choose_vec_stream_shape_16(
    __hb_block: VecBlockSparseChoice,
    __hb_ret: VecStreamShapeChoice,
):
    print("Choose stream shape 16.")

@Function(
    "ret.block_sparse = false",
)
def choose_default_vec_block_sparse(__hb_ret: VecBlockSparseChoice):
    print("Choose default block sparse (false).")

@Function(
    "ret.block_sparse = true",
)
def choose_vec_block_sparse_true(__hb_ret: VecBlockSparseChoice):
    print("Choose block sparse (true).")

@Function(
    "ret.stream_shape = 16",
)
def choose_default_stream_shape(
    __hb_block: BlockSparseChoice,
    __hb_ret: StreamShapeChoice,
):
    print("Choose default stream shape (16).")

@Function(
    "ret.stream_shape = 1",
)
def choose_stream_shape_1(
    __hb_block: BlockSparseChoice,
    __hb_ret: StreamShapeChoice,
):
    print("Choose stream shape 1.")

@Function(
    "ret.stream_shape = 2",
)
def choose_stream_shape_2(
    __hb_block: BlockSparseChoice,
    __hb_ret: StreamShapeChoice,
):
    print("Choose stream shape 2.")

@Function(
    "ret.stream_shape = 4",
)
def choose_stream_shape_4(
    __hb_block: BlockSparseChoice,
    __hb_ret: StreamShapeChoice,
):
    print("Choose stream shape 4.")

@Function(
    "ret.stream_shape = 8",
)
def choose_stream_shape_8(
    __hb_block: BlockSparseChoice,
    __hb_ret: StreamShapeChoice,
):
    print("Choose stream shape 8.")

@Function(
    "ret.stream_shape = 16",
)
def choose_stream_shape_16(
    __hb_block: BlockSparseChoice,
    __hb_ret: StreamShapeChoice,
):
    print("Choose stream shape 16.")

@Function(
    "ret.block_sparse = false",
)
def choose_default_block_sparse(__hb_ret: BlockSparseChoice):
    print("Choose default block sparse (false).")

@Function(
    "ret.block_sparse = true",
)
def choose_block_sparse_true(__hb_ret: BlockSparseChoice):
    print("Choose block sparse (true).")

@Function(
)
def parallelization_only(
    __hb_level: StreamLevelChoice,
    __hb_ret: ParallelizationPass,
):
    print("Parallelization only.")

@Function(
)
def parallelization_and_vectorization(
    __hb_level: VecStreamLevelChoice,
    __hb_ret: ParallelizationPass,
):
    print("Parallelization and Vectorization.")


@Function(

)
def vectorization(__hb_shape: StreamShapeChoice, __hb_ret: VectorizationPass):
    print("Vectorization.")
