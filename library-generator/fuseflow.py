from dataclasses import dataclass

from lib import Function, Prop, Type

'''
To update to the new library: 
- The new classes are just the "S" classes, not the D classes
- Each Python function would be pass
- Nore S's and D's as arguments
- Python functions never actually return anything (it saves to disk instead)
- @input will generate a @Prop and a @Type 
    - The prefix will have P_ (i.e. P_Mlir)
- Should be a pretty quick upgrade/refactor
'''


@Prop
@dataclass
class MlirInput:
    "Input MLIR metadata used to ground scheduling choices."

    id: str
    "Stable identifier for the MLIR file (usually the basename)."

    path: str
    "Path to the MLIR file."

    num_loops: int
    "Maximum loop depth across relevant Linalg ops."


@Prop
@dataclass
class ParConfig:
    "Allowed stream-parallelizer configuration for an MLIR input."

    mlir: str
    "MLIR identifier this config applies to."

    level: int
    "Stream level to parallelize."

    factor: int
    "Parallelization factor."


@Prop
@dataclass
class VecConfig:
    "Allowed stream-vectorizer configuration for an MLIR input."

    mlir: str
    "MLIR identifier this config applies to."

    shape: int
    "Vector length (stream_shape)."

    block_sparse: bool
    "Whether block-sparse vectorization is enabled."


@Type
@dataclass
class Mlir:
    """@intermediate:MLIR input"""

    @dataclass
    class S:
        id: str
        "Stable identifier for the MLIR file (usually the basename)."

        path: str
        "Path to the MLIR file."

        num_loops: int
        "Maximum loop depth across relevant Linalg ops."

    @dataclass
    class D:
        pass

    static: S
    dynamic: D


@Type
@dataclass
class Parallelization:
    """@intermediate:Parallelization schedule"""

    @dataclass
    class S:
        mlir: str
        "MLIR identifier this schedule applies to."

        stream_level: int
        "Stream level to parallelize."

        par_factor: int
        "Parallelization factor."

    @dataclass
    class D:
        pass

    static: S
    dynamic: D


@Type
@dataclass
class Vectorization:
    """@intermediate:Vectorization schedule"""

    @dataclass
    class S:
        mlir: str
        "MLIR identifier this schedule applies to."

        stream_shape: int
        "Vector length (stream_shape)."

        block_sparse: bool
        "Whether block-sparse vectorization is enabled."

    @dataclass
    class D:
        pass

    static: S
    dynamic: D


@Type
@dataclass
class Schedule:
    """@intermediate:FuseFlow schedule"""

    @dataclass
    class S:
        mlir: str
        "MLIR identifier this schedule applies to."

        stream_level: int
        "Stream level to parallelize."

        par_factor: int
        "Parallelization factor."

        stream_shape: int
        "Vector length (stream_shape)."

        block_sparse: bool
        "Whether block-sparse vectorization is enabled."

    @dataclass
    class D:
        pass

    static: S
    dynamic: D


@Type
@dataclass
class ScheduleChoice:
    """@intermediate:Schedule choice (mlir-scoped)"""

    @dataclass
    class S:
        mlir: str
        "MLIR identifier this schedule applies to."

    @dataclass
    class D:
        pass

    static: S
    dynamic: D


@Function(
    "MlirInput { id = ret.id, path = ret.path, num_loops = ret.num_loops }",
)
def load_mlir(ret: Mlir.S) -> Mlir.D:
    """Load MLIR metadata"""
    return Mlir.D()


@Function(
    "ret.mlir = mlir.id",
    "ret.stream_level < mlir.num_loops",
    "ParConfig { mlir = mlir.id, level = ret.stream_level, factor = ret.par_factor }",
)
def choose_parallel(mlir: Mlir, ret: Parallelization.S) -> Parallelization.D:
    """Choose a parallelization configuration"""
    return Parallelization.D()


@Function(
    "ret.mlir = mlir.id",
    "VecConfig { mlir = mlir.id, shape = ret.stream_shape, block_sparse = ret.block_sparse }",
)
def choose_vector(mlir: Mlir, ret: Vectorization.S) -> Vectorization.D:
    """Choose a vectorization configuration"""
    return Vectorization.D()


@Function(
    "ret.mlir = mlir.id",
    "ret.stream_level < mlir.num_loops",
    "ParConfig { mlir = mlir.id, level = ret.stream_level, factor = ret.par_factor }",
    "VecConfig { mlir = mlir.id, shape = ret.stream_shape, block_sparse = ret.block_sparse }",
)
def choose_schedule(mlir: Mlir, ret: Schedule.S) -> Schedule.D:
    """Choose a combined FuseFlow schedule"""
    return Schedule.D()


@Function(
    "ret.mlir = sched.mlir",
)
def finalize_schedule(sched: Schedule, ret: ScheduleChoice.S) -> ScheduleChoice.D:
    """Expose schedule choice without fixing stream parameters"""
    return ScheduleChoice.D()
