import os
import datetime
import polars as pl
import json

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

@Helper
def write_schedule_json(schedule_dict, output_dir, *, filename="schedule.json"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schedule_dict, f, indent=2, sort_keys=True)
    return out_path

@Helper
class ScheduleState:
    stream_level = None
    par_factor = None
    stream_shape = None
    block_sparse = None

    @classmethod
    def set_values(cls, *, stream_level=None, par_factor=None, stream_shape=None, block_sparse=None):
        if stream_level is not None:
            cls.stream_level = stream_level
        if par_factor is not None:
            cls.par_factor = par_factor
        if stream_shape is not None:
            cls.stream_shape = stream_shape
        if block_sparse is not None:
            cls.block_sparse = block_sparse

    @classmethod
    def as_dict(cls):
        return {
            "stream-parallelizer": {
                "stream-level": cls.stream_level,
                "par-factor": cls.par_factor,
            },
            "stream-vectorizer": {
                "stream-shape": cls.stream_shape,
                "enable-block-sparse": cls.block_sparse,
            },
        }

    @classmethod
    def missing_fields(cls):
        missing = []
        if cls.stream_level is None:
            missing.append("stream_level")
        if cls.par_factor is None:
            missing.append("par_factor")
        if cls.stream_shape is None:
            missing.append("stream_shape")
        if cls.block_sparse is None:
            missing.append("block_sparse")
        return missing

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
    ScheduleState.set_values(
        stream_level=0,
        par_factor=1,
        stream_shape=16,
        block_sparse=False,
    )
    schedule = ScheduleState.as_dict()
    write_schedule_json(schedule, __hb_ret.path)
    print(json.dumps(schedule, indent=2, sort_keys=True))


@Function(
)
def build_schedule(__hb_pass: ParallelizationPass, __hb_ret: FuseFlowSchedule):
    missing = ScheduleState.missing_fields()
    if missing:
        raise RuntimeError(
            "schedule state incomplete; missing: " + ", ".join(missing)
        )
    schedule = ScheduleState.as_dict()
    write_schedule_json(schedule, __hb_ret.path)
    print(json.dumps(schedule, indent=2, sort_keys=True))

@Function(
    "ret.stream_level = 0",
)
def choose_default_stream_level(__hb_vec: VectorizationPass, __hb_ret: StreamLevelChoice):
    ScheduleState.set_values(stream_level=0)
    print("Choose default stream level (0).")

@Function(
    "ret.stream_level = 1",
)
def choose_stream_level_1(__hb_vec: VectorizationPass, __hb_ret: StreamLevelChoice):
    ScheduleState.set_values(stream_level=1)
    print("Choose stream level 1.")

@Function(
    "ret.stream_level = 2",
)
def choose_stream_level_2(__hb_vec: VectorizationPass, __hb_ret: StreamLevelChoice):
    ScheduleState.set_values(stream_level=2)
    print("Choose stream level 2.")

@Function(
    "ret.stream_level = 4",
)
def choose_stream_level_4(__hb_vec: VectorizationPass, __hb_ret: StreamLevelChoice):
    ScheduleState.set_values(stream_level=4)
    print("Choose stream level 4.")

@Function(
    "ret.stream_level = 8",
)
def choose_stream_level_8(__hb_vec: VectorizationPass, __hb_ret: StreamLevelChoice):
    ScheduleState.set_values(stream_level=8)
    print("Choose stream level 8.")

@Function(
    "ret.stream_level = 16",
)
def choose_stream_level_16(__hb_vec: VectorizationPass, __hb_ret: StreamLevelChoice):
    ScheduleState.set_values(stream_level=16)
    print("Choose stream level 16.")

@Function(
    "ret.par_factor = 1",
)
def choose_default_par_factor(__hb_level: StreamLevelChoice, __hb_ret: ParFactorChoice):
    ScheduleState.set_values(par_factor=1)
    print("Choose default par factor (1).")

@Function(
    "ret.par_factor = 2",
)
def choose_par_factor_2(__hb_level: StreamLevelChoice, __hb_ret: ParFactorChoice):
    ScheduleState.set_values(par_factor=2)
    print("Choose par factor 2.")

@Function(
    "ret.par_factor = 4",
)
def choose_par_factor_4(__hb_level: StreamLevelChoice, __hb_ret: ParFactorChoice):
    ScheduleState.set_values(par_factor=4)
    print("Choose par factor 4.")

@Function(
    "ret.par_factor = 8",
)
def choose_par_factor_8(__hb_level: StreamLevelChoice, __hb_ret: ParFactorChoice):
    ScheduleState.set_values(par_factor=8)
    print("Choose par factor 8.")

@Function(
    "ret.par_factor = 16",
)
def choose_par_factor_16(__hb_level: StreamLevelChoice, __hb_ret: ParFactorChoice):
    ScheduleState.set_values(par_factor=16)
    print("Choose par factor 16.")

@Function(
    "ret.stream_shape = 16",
)
def choose_default_stream_shape(__hb_ret: StreamShapeChoice):
    ScheduleState.set_values(stream_shape=16)
    print("Choose default stream shape (16).")

@Function(
    "ret.stream_shape = 1",
)
def choose_stream_shape_1(__hb_ret: StreamShapeChoice):
    ScheduleState.set_values(stream_shape=1)
    print("Choose stream shape 1.")

@Function(
    "ret.stream_shape = 2",
)
def choose_stream_shape_2(__hb_ret: StreamShapeChoice):
    ScheduleState.set_values(stream_shape=2)
    print("Choose stream shape 2.")

@Function(
    "ret.stream_shape = 4",
)
def choose_stream_shape_4(__hb_ret: StreamShapeChoice):
    ScheduleState.set_values(stream_shape=4)
    print("Choose stream shape 4.")

@Function(
    "ret.stream_shape = 8",
)
def choose_stream_shape_8(__hb_ret: StreamShapeChoice):
    ScheduleState.set_values(stream_shape=8)
    print("Choose stream shape 8.")

@Function(
    "ret.stream_shape = 16",
)
def choose_stream_shape_16(__hb_ret: StreamShapeChoice):
    ScheduleState.set_values(stream_shape=16)
    print("Choose stream shape 16.")

@Function(
    "ret.block_sparse = false",
)
def choose_default_block_sparse(__hb_shape: StreamShapeChoice, __hb_ret: BlockSparseChoice):
    ScheduleState.set_values(block_sparse=False)
    print("Choose default block sparse (false).")

@Function(
    "ret.block_sparse = true",
)
def choose_block_sparse_true(__hb_shape: StreamShapeChoice, __hb_ret: BlockSparseChoice):
    ScheduleState.set_values(block_sparse=True)
    print("Choose block sparse (true).")

@Function(
)
def vectorization(__hb_block: BlockSparseChoice, __hb_ret: VectorizationPass):
    print("Vectorization.")

@Function(
)
def parallelization(__hb_par: ParFactorChoice, __hb_ret: ParallelizationPass):
    print("Parallelization.")
